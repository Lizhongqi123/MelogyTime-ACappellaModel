import os

import torch
from config import Config
from model import ACappella
from tokenizer_utils import load_tokenizer, build_prompt, EOS


class Generator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.tokenizer = load_tokenizer(cfg.tokenizer_name)
        self.model = ACappella(cfg).to(cfg.device)
        # 优先加载 best（avg_loss 最低那一版），没有则退回主 ckpt。
        # 训练已改为断点续训，ckpt 现在是 dict{model, optimizer, epoch, ...}，
        # 这里要同时兼容裸 state_dict 的旧文件。
        ckpt_path = cfg.ckpt_path
        best_path = os.path.splitext(cfg.ckpt_path)[0] + "_best.pt"
        if os.path.exists(best_path):
            ckpt_path = best_path
            print(f"使用 best checkpoint: {best_path}")
        try:
            ck = torch.load(ckpt_path, map_location=cfg.device)
            state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
            self.model.load_state_dict(state)
            meta = ""
            if isinstance(ck, dict):
                meta = f"（epoch={ck.get('epoch')}, best_loss={ck.get('best_loss')}）"
            print(f"已加载 checkpoint {ckpt_path}{meta}")
        except FileNotFoundError:
            print("未找到 checkpoint，使用随机权重")
        except RuntimeError as e:
            # 与本版本结构不兼容（如缺少 RoPE 缓冲区），或是在旧对话模板下训练的
            # 权重（编码不匹配），生成只会是乱码，必须重新训练
            print(f"checkpoint 与当前模型结构不匹配，将使用随机权重（需要重新训练）: {e}")
        # 结束符集合：模板真正写入的 EOS（im_end）+ tokenizer 自带 eos（endoftext）都认
        self._eos_ids = {
            self.tokenizer.convert_tokens_to_ids(EOS),
            self.tokenizer.eos_token_id,
        }

    def _sample(self, logits):
        logits = logits / self.cfg.temperature
        probs = torch.softmax(logits, dim=-1)
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        mask = cumsum - sorted_probs > self.cfg.top_p
        sorted_probs[mask] = 0
        sorted_probs /= sorted_probs.sum()
        return sorted_idx[0, torch.multinomial(sorted_probs[0], 1)]

    @torch.no_grad()
    def chat(self, history):
        # 硬截断：历史滚长后必须截尾保留最近内容，否则 kv 缓存超出
        # RoPE 表覆盖范围（max_seq_len + max_new_tokens），增量解码会形状错乱崩溃
        max_ctx = self.cfg.max_seq_len - 8  # 给结尾 "Assistant: " 留位置
        prompt = build_prompt(history)
        enc = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        if len(enc) > max_ctx:
            enc = enc[-max_ctx:]  # 保尾截头：最近的话比久远的更重要
        output_ids = torch.tensor([enc], device=self.cfg.device)
        kv_caches = None
        L = output_ids.shape[1]
        # 总长上限同时受 RoPE 表长度约束
        limit = min(self.cfg.max_seq_len + self.cfg.max_new_tokens,
                    L + self.cfg.max_new_tokens)

        for _ in range(self.cfg.max_new_tokens):
            if output_ids.shape[1] >= limit:
                break
            cur = output_ids if kv_caches is None else output_ids[:, -1:]
            # pos_start：整段输入从 0 开始；增量解码时从已有序列长度-1 开始
            pos_start = 0 if kv_caches is None else output_ids.shape[1] - 1
            logits, kv_caches = self.model(cur, kv_caches, pos_start=pos_start)
            next_id = self._sample(logits[:, -1, :])
            if int(next_id) in self._eos_ids:
                break
            output_ids = torch.cat([output_ids, next_id.unsqueeze(0)], dim=1)

        new_tokens = output_ids[0, L:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)
