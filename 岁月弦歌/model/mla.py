import torch
import torch.nn as nn
import torch.nn.functional as F


def apply_rope(x, cos, sin):
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class MLA(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.qk_nope = cfg.qk_nope_head_dim
        self.qk_rope = cfg.qk_rope_head_dim
        self.q_head_dim = self.qk_nope + self.qk_rope
        self.kv_lora_rank = cfg.kv_lora_rank

        self.q_proj = nn.Linear(cfg.dim, self.n_heads * self.q_head_dim, bias=False)
        self.kv_down = nn.Linear(cfg.dim, self.kv_lora_rank, bias=False)
        self.k_up = nn.Linear(self.kv_lora_rank, self.n_heads * self.qk_nope, bias=False)
        self.v_up = nn.Linear(self.kv_lora_rank, self.n_heads * self.qk_nope, bias=False)
        self.k_rope = nn.Linear(cfg.dim, self.n_heads * self.qk_rope, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.qk_nope, cfg.dim, bias=False)

    def forward(self, x, kv_cache=None, rope_cache=None):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.q_head_dim)
        q_nope, q_rope = q.split([self.qk_nope, self.qk_rope], dim=-1)

        # 计算 latent 和 k_rope
        kv_latent = self.kv_down(x)
        k_rope = self.k_rope(x).view(B, T, self.n_heads, self.qk_rope)

        # 先按当前位置施加 RoPE，再进缓存：
        # 缓存里存的必须是已旋转好的 k_rope，否则增量生成时位置全错
        if rope_cache is not None:
            cos, sin = rope_cache
            q_rope = apply_rope(q_rope, cos, sin)
            k_rope = apply_rope(k_rope, cos, sin)

        # 缓存 (latent, k_rope)
        if kv_cache is not None:
            prev_latent, prev_k_rope = kv_cache
            kv_latent = torch.cat([prev_latent, kv_latent], dim=1)
            k_rope = torch.cat([prev_k_rope, k_rope], dim=1)

        new_cache = (kv_latent, k_rope)

        k_nope = self.k_up(kv_latent).view(B, -1, self.n_heads, self.qk_nope)
        v = self.v_up(kv_latent).view(B, -1, self.n_heads, self.qk_nope)

        q_nope = q_nope.transpose(1, 2)
        q_rope = q_rope.transpose(1, 2)
        k_nope_t = k_nope.transpose(1, 2)
        k_rope_t = k_rope.transpose(1, 2)
        v_t = v.transpose(1, 2)

        # 增量解码时 kv_len > q_len，is_causal=True 是左上对齐的，
        # 会把当前 query 错误地屏蔽掉大部分缓存。必须用右下对齐的真实因果掩码：
        # 第 i 行允许 attend 到 j <= i + (kv_len - T)
        kv_len = kv_latent.shape[1]
        if kv_len == T:
            attn_mask = None
            causal = True
        else:
            causal = False
            i = torch.arange(T, device=x.device).unsqueeze(1)
            j = torch.arange(kv_len, device=x.device).unsqueeze(0)
            attn_mask = j <= i + (kv_len - T)  # (T, kv_len)

        def sdpa(qh, kh):
            if causal:
                return F.scaled_dot_product_attention(qh, kh, v_t, is_causal=True)
            return F.scaled_dot_product_attention(qh, kh, v_t, attn_mask=attn_mask)

        attn_nope = sdpa(q_nope, k_nope_t)
        attn_rope = sdpa(q_rope, k_rope_t)
        out = (attn_nope + attn_rope).transpose(1, 2).contiguous()
        out = out.view(B, -1, self.n_heads * self.qk_nope)
        return self.o_proj(out), new_cache