import json
import torch
from torch.utils.data import Dataset
from tokenizer_utils import build_prompt, EOS


class DialogueDataset(Dataset):
    def __init__(self, path, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.samples = []

        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    # 校验格式
                    if "messages" not in item:
                        print(f"第 {line_num} 行缺少 messages 字段，跳过", flush=True)
                        continue
                    msgs = item["messages"]
                    if len(msgs) < 2:
                        print(f"第 {line_num} 行消息数不足 2，跳过", flush=True)
                        continue
                    if msgs[-1]["role"] != "assistant":
                        print(f"第 {line_num} 行最后一条不是 assistant，跳过", flush=True)
                        continue
                    self.samples.append(item)
                except json.JSONDecodeError as e:
                    print(f"第 {line_num} 行 JSON 解析失败: {e}，跳过", flush=True)
                    continue

        print(f"数据集加载完成，有效样本数: {len(self.samples)}", flush=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        messages = self.samples[idx]["messages"]

        # 拼完整对话（用于 input_ids）
        full_text = build_prompt(messages[:-1]) + messages[-1]["content"] + EOS

        # 拼 prompt 部分（用于 mask，只保留 Assistant 的 loss）
        prompt_text = build_prompt(messages[:-1])

        # tokenize 完整对话
        full_enc = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = full_enc["input_ids"].squeeze(0)

        # tokenize prompt 部分，用来算长度
        prompt_enc = self.tokenizer(
            prompt_text,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        prompt_len = prompt_enc["input_ids"].shape[1]

        # labels 默认等于 input_ids
        labels = input_ids.clone()

        # 把 prompt 部分 mask 掉，只对 Assistant 回答算 loss
        labels[:prompt_len] = -100

        # 只屏蔽尾部真正的 padding（不能按 pad_token_id 屏蔽，
        # 因为 pad 与 eos 同 id，会把答案末尾的 EOS 一起误伤）
        real_len = len(self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_len,
        )["input_ids"])
        labels[real_len:] = -100

        return {
            "input_ids": input_ids,
            "labels": labels,
        }