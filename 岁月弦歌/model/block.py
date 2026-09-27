import torch.nn as nn
from .mla import MLA
from .moe import MelogyTimeMoE


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = nn.RMSNorm(cfg.dim)
        self.attn = MLA(cfg)
        self.norm2 = nn.RMSNorm(cfg.dim)
        self.moe = MelogyTimeMoE(cfg)

    def forward(self, x, kv_cache=None, rope_cache=None):
        h, new_kv = self.attn(self.norm1(x), kv_cache, rope_cache)
        x = x + h
        x = x + self.moe(self.norm2(x))
        return x, new_kv