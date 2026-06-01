"""
DPO Trainer — Pairwise SDE-DPO for SD3 Flow Matching.

Single-timestep velocity-matching DPO loss (O(1) memory w.r.t. SDE steps).
Does NOT backpropagate through the full SDE trajectory.

Loss (per pair, following Wallace et al. Diffusion-DPO adapted for flow matching):
    Sample t ~ U(0,1), ε^w, ε^l ~ N(0,I)
    x_t = (1-t)·x_0 + t·ε          (rectified flow interpolation)
    u_t = ε - x_0                   (target velocity)
    ℓ_θ = ||u_t - v_θ(x_t, t, c)||²
    ℓ_ref = ||u_t - v_ref(x_t, t, c)||²
    L = -log σ(-β·T·[(ℓ_θ^w - ℓ_ref^w) - (ℓ_θ^l - ℓ_ref^l)])
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn.functional as F


def _velocity_mse(transformer, x_t, t, prompt_embeds, pooled_embeds, u_t):
    """Compute ||u_t - v(x_t, t, c)||² (mean over all dims)."""
    v_pred = transformer(
        hidden_states=x_t,
        encoder_hidden_states=prompt_embeds,
        pooled_projections=pooled_embeds,
        timestep=t.expand(x_t.shape[0]),
        return_dict=False,
    )[0]
    return torch.mean((u_t - v_pred) ** 2)


def dpo_update(
    transformer_trainable: torch.nn.Module,
    transformer_ref: torch.nn.Module,
    chain_data: List[dict],
    pairs: List[Tuple[int, int]],
    optimizer: torch.optim.Optimizer,
    beta: float = 1.0,
    num_inference_steps: int = 20,
    max_grad_norm: float = 5.0,
) -> dict:
    """Pairwise SDE-DPO update — single-timestep velocity matching.

    For each (winner, loser) pair:
        1. Extract final latent x_0 from SDE trajectory
        2. Sample random t, noise ε → rectified flow interpolation
        3. Compute velocity MSE for trainable and reference models
        4. DPO loss: -log σ(-β·T·[(ℓ_θ^w - ℓ_ref^w) - (ℓ_θ^l - ℓ_ref^l)])

    All pairs averaged → one backward → one optimizer step.

    Args:
        transformer_trainable: LoRA transformer (trainable).
        transformer_ref: frozen base transformer.
        chain_data: list of chain dicts with all_latents, prompt_embeds, pooled_embeds.
        pairs: list of (winner_idx, loser_idx) tuples.
        optimizer: AdamW optimizer.
        beta: DPO temperature (default 1.0 for velocity-loss scale).
        num_inference_steps: T, used as scaling factor in loss.
        max_grad_norm: gradient clipping threshold.

    Returns:
        metrics dict with loss, grad_norm, n_pairs, beta.
    """
    n_pairs = len(pairs)
    if n_pairs == 0:
        return {"loss": 0.0, "grad_norm": 0.0, "grad_clipped": False,
                "n_pairs": 0, "beta": beta}

    T = float(num_inference_steps)
    pair_losses = []

    for w_idx, l_idx in pairs:
        w_data = chain_data[w_idx]
        l_data = chain_data[l_idx]

        # Final latent x_0 (last element of SDE trajectory)
        x0_w = w_data["all_latents"][-1]
        x0_l = l_data["all_latents"][-1]

        # Prompt embeddings (conditional only, no CFG)
        pe_w = w_data["prompt_embeds"]
        pooled_w = w_data["pooled_embeds"]
        pe_l = l_data["prompt_embeds"]
        pooled_l = l_data["pooled_embeds"]

        device = x0_w.device
        model_dtype = next(transformer_trainable.parameters()).dtype

        # Cast latents to model dtype (bf16 pipeline, fp32 latents from sampling)
        x0_w = x0_w.to(dtype=model_dtype)
        x0_l = x0_l.to(dtype=model_dtype)
        pe_w = pe_w.to(dtype=model_dtype)
        pooled_w = pooled_w.to(dtype=model_dtype)
        pe_l = pe_l.to(dtype=model_dtype)
        pooled_l = pooled_l.to(dtype=model_dtype)

        # Sample random timestep t ~ U(0, 1)
        t = torch.rand(1, device=device, dtype=model_dtype)

        # Sample noise (in model dtype)
        eps_w = torch.randn_like(x0_w)
        eps_l = torch.randn_like(x0_l)

        # Rectified flow interpolation: x_t = (1-t)·x_0 + t·ε
        xt_w = (1 - t) * x0_w + t * eps_w
        xt_l = (1 - t) * x0_l + t * eps_l

        # Target velocity: u_t = ε - x_0
        ut_w = eps_w - x0_w
        ut_l = eps_l - x0_l

        # Velocity MSE — trainable model (WITH gradient)
        loss_w = _velocity_mse(transformer_trainable, xt_w, t, pe_w, pooled_w, ut_w)
        loss_l = _velocity_mse(transformer_trainable, xt_l, t, pe_l, pooled_l, ut_l)

        # Velocity MSE — reference model (NO gradient)
        with torch.no_grad():
            loss_ref_w = _velocity_mse(transformer_ref, xt_w, t, pe_w, pooled_w, ut_w)
            loss_ref_l = _velocity_mse(transformer_ref, xt_l, t, pe_l, pooled_l, ut_l)

        # DPO loss: -log σ(-β·T·[(ℓ_θ^w - ℓ_ref^w) - (ℓ_θ^l - ℓ_ref^l)])
        delta = (loss_w - loss_ref_w) - (loss_l - loss_ref_l)
        pair_loss = -F.logsigmoid(-beta * T * delta)
        pair_losses.append(pair_loss)

    # Mean over pairs
    loss = torch.stack(pair_losses).mean()
    loss.backward()

    # Gradient clipping (only LoRA params)
    params = [p for p in transformer_trainable.parameters() if p.requires_grad]
    grad_norm = torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
    grad_norm_val = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm)

    optimizer.step()
    optimizer.zero_grad()

    return {
        "loss": loss.item(),
        "grad_norm": grad_norm_val,
        "grad_clipped": grad_norm_val >= max_grad_norm * 0.99,
        "n_pairs": n_pairs,
        "beta": beta,
    }
