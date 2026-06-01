import os


def load_prompts_from_file(file_path):
    """Load prompts from a text file, one prompt per line."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Prompt file not found: {file_path}")

    with open(file_path, "r") as f:
        return [line.strip() for line in f if line.strip()]


class PromptLoader:
    """顺序读取 prompt，到文件末尾返回 None。用于训练时的 epoch 遍历。"""

    def __init__(self, file_path, shuffle=False):
        self.prompts = load_prompts_from_file(file_path)
        self.shuffle = shuffle
        self.index = 0

    def get_next_prompt(self):
        if self.index >= len(self.prompts):
            return None
        prompt = self.prompts[self.index]
        self.index += 1
        return prompt

    def reset(self):
        self.index = 0

    def __len__(self):
        return len(self.prompts)
