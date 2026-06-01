"""
随机性流匹配 SDE 采样器 (Stochastic Flow Match Scheduler).

基于 SD3 的 FlowMatchEulerDiscreteScheduler，注入 SDE 噪声使采样随机化。

采样公式 (参照 Flow-GRPO):
    x_{t-1} = x_t + f_θ(x_t, t, c) * dt + σ_t * √|dt| * ε

其中:
    f_θ = v_θ + σ_t²/(2t) * (x_t + (1-t) * v_θ)    # 修正后的漂移项
    σ_t = a * √(t / (1-t))                            # 噪声调度
    dt = sigma_next - sigma (< 0, 从噪声到清晰)       # 负步长
    ε ~ N(0, I)

转移概率:
    p_θ(x_{t-1} | x_t) ~ N(x_{t-1}; x_t + f_θ * dt, σ_t² * |dt|)

当 a=0 时，σ_t=0，退化为确定性流匹配采样 (等价于原始 FlowMatchEulerDiscreteScheduler)。
"""

import math
from typing import Optional, Tuple, Union

import torch

from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)


def _left_broadcast(t, shape):
    assert t.ndim <= len(shape)
    return t.reshape(t.shape + (1,) * (len(shape) - t.ndim)).broadcast_to(shape)


def _randn_tensor(shape, generator, device, dtype):
    """兼容旧版 diffusers.randn_tensor，新版已移除。处理 generator/device 不一致问题。"""
    if generator is not None and generator.device != device:
        gen = torch.Generator(device=device).manual_seed(generator.initial_seed())
    else:
        gen = generator
    return torch.randn(shape, generator=gen, device=device, dtype=dtype)


class StochasticFlowMatchScheduler(FlowMatchEulerDiscreteScheduler):
    """
    继承 FlowMatchEulerDiscreteScheduler，重写 step() 注入 SDE 噪声。

    Args:
        a: 噪声系数，控制 σ_t = a * √(t / (1-t)) 的幅度。
           0 = 确定性, ~0.7 = Flow-GRPO 推荐, 1.0 = 最大噪声。
        **kwargs: 传给父类 FlowMatchEulerDiscreteScheduler 的参数。
    """

    def __init__(self, a: float = 0.7, **kwargs):
        super().__init__(**kwargs)
        self.a = a

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        sample: torch.FloatTensor,
        generator: Optional[torch.Generator] = None,
        prev_sample: Optional[torch.FloatTensor] = None,
        return_dict: bool = True,
    ) -> Union[Tuple, "FlowMatchEulerDiscreteSchedulerOutput"]:
        """
        执行一步随机性流匹配 SDE 采样。

        Args:
            model_output: v_θ(x_t, t, c) — SD3 Transformer 输出的向量场。
            timestep: 当前离散时间步 (对应 self.timesteps 中的值)。
            sample: x_t — 当前潜变量。
            generator: 随机数生成器 (采样模式)。
            prev_sample: 预先给定的 x_{t-1} (训练模式，用于计算 log_prob)。
            return_dict: 是否返回 SchedulerOutput (保持兼容性)。

        Returns:
            (prev_sample, log_prob) — 上一步潜变量和对数概率。
        """
        # 1. 找到当前步在 timesteps 中的索引
        step_index = (self.timesteps == timestep).nonzero().item()
        sigma = self.sigmas[step_index]          # 当前噪声水平 = 连续时间 t
        sigma_next = self.sigmas[step_index + 1]  # 下一步噪声水平

        dt = sigma_next - sigma   # 负值 (从噪声到清晰)
        abs_dt = -dt              # 正步长 Δt

        # 2. 计算 σ_t = a * √(t / (1-t))
        #    当 t→0: σ_t → 0 (噪声消失，确定性收敛)
        #    当 t→1: σ_t → ∞，需 clamp 防止数值爆炸
        safe_one_minus_sigma = torch.clamp(1.0 - sigma, min=1e-4)
        sigma_t = self.a * torch.sqrt(sigma / safe_one_minus_sigma)
        sigma_t = torch.clamp(sigma_t, max=10.0)

        # 3. 计算修正后的漂移项 f_θ = v_θ + σ_t²/(2t) * (x_t + (1-t) * v_θ)
        #    注意: 从已 clamp 的 sigma_t² 推导 coeff，确保修正项也被 clamp 保护
        safe_sigma = torch.clamp(sigma, min=1e-4)
        correction = (sigma_t ** 2) / (2.0 * safe_sigma) * (
            sample + (1.0 - sigma) * model_output
        )
        f_theta = model_output + correction

        # 4. 计算均值: μ = x_t + f_θ * dt  (dt < 0)
        mean = sample + f_theta * dt

        # 5. 计算标准差: σ = σ_t * √(Δt)
        std = sigma_t * (abs_dt ** 0.5)
        std = _left_broadcast(std, sample.shape)

        # 6. 获取 prev_sample
        if prev_sample is not None and generator is not None:
            raise ValueError(
                "Cannot pass both generator and prev_sample. "
                "Use generator for sampling mode, prev_sample for training mode."
            )

        if prev_sample is None:
            # 采样模式: 生成随机噪声
            if generator is not None:
                noise = _randn_tensor(
                    sample.shape,
                    generator=generator,
                    device=sample.device,
                    dtype=sample.dtype,
                )
            else:
                noise = torch.randn_like(sample)
            prev_sample = mean + std * noise

        # 7. 计算 log_prob: log N(prev_sample; mean, std²)
        #    log p = -||prev_sample - mean||² / (2 * std²) - log(std) - log(√(2π))
        #    全程 float32: float16 下 std² 可能下溢 (a=0 时 1e-12)、sum 可能溢出
        std_safe = torch.clamp(std, min=1e-6).float()
        diff = (prev_sample.detach().float() - mean.float())
        log_prob = (
            -(diff ** 2) / (2 * std_safe ** 2)
            - torch.log(std_safe)
            - math.log(math.sqrt(2 * math.pi))
        )
        # 沿 batch 外的所有维度求和 (total log probability)
        log_prob = log_prob.sum(dim=tuple(range(1, log_prob.ndim)))

        return prev_sample.type(sample.dtype), log_prob
