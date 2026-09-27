import math

import torch
import torch.nn as nn
from .block import Block


class ACappella(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm_f = nn.RMSNorm(cfg.dim)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)

        # RoPE 旋转表：覆盖训练窗口 + 增量生成预算
        # interleaved 布局，与 mla.apply_rope 的 x[..., ::2]/x[..., 1::2] 对应
        L = cfg.max_seq_len + cfg.max_new_tokens
        half = cfg.qk_rope_head_dim // 2
        inv = torch.exp(
            torch.arange(half, dtype=torch.float32) * -(math.log(10000.0) / half)
        )
        pos = torch.arange(L, dtype=torch.float32)
        ang = torch.outer(pos, inv)  # (L, half)
        self.register_buffer("rope_cos", ang.cos()[None, :, None, :])
        self.register_buffer("rope_sin", ang.sin()[None, :, None, :])

    def forward(self, input_ids, kv_caches=None, pos_start=0):
        T = input_ids.shape[1]
        # 按当前片段的位置切片，增量解码时 pos_start 会逐步后移
        rope_cache = (
            self.rope_cos[:, pos_start:pos_start + T],
            self.rope_sin[:, pos_start:pos_start + T],
        )
        x = self.tok_emb(input_ids)
        new_kvs = []
        for i, block in enumerate(self.blocks):
            cache = kv_caches[i] if kv_caches is not None else None
            x, new_kv = block(x, cache, rope_cache)
            new_kvs.append(new_kv)
        x = self.norm_f(x)
        return self.lm_head(x), new_kvs
