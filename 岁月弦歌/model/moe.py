import torch
import torch.nn as nn


class Expert(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden, bias=False),
            nn.SiLU(),
            nn.Linear(hidden, dim, bias=False),
        )

    def forward(self, x):
        return self.net(x)


class MelogyTimeMoE(nn.Module):
    """细粒度路由专家 + 共享专家 + Sigmoid 门控 + 无辅助损失偏置"""
    def __init__(self, cfg):
        super().__init__()
        self.n_routed = cfg.n_routed_experts
        self.n_shared = cfg.n_shared_experts
        self.top_k = cfg.top_k
        self.routed_experts = nn.ModuleList([
            Expert(cfg.dim, cfg.expert_hidden) for _ in range(self.n_routed)
        ])
        self.shared_experts = nn.ModuleList([
            Expert(cfg.dim, cfg.expert_hidden) for _ in range(self.n_shared)
        ])
        self.gate = nn.Linear(cfg.dim, self.n_routed, bias=False)
        self.register_buffer("routing_bias", torch.zeros(self.n_routed))
        self.bias_update_rate = 0.001

    def forward(self, x):
        B, T, C = x.shape
        x_flat = x.view(-1, C)
        shared_out = sum(e(x_flat) for e in self.shared_experts)

        logits = self.gate(x_flat)
        scores = torch.sigmoid(logits)
        scores_sel = scores + self.routing_bias
        _, topk_idx = scores_sel.topk(self.top_k, dim=-1)
        weights = scores.gather(1, topk_idx)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-9)

        routed_out = torch.zeros_like(x_flat)
        for i in range(self.top_k):
            e_idx = topk_idx[:, i]
            w = weights[:, i].unsqueeze(-1)
            for e in range(self.n_routed):
                mask = (e_idx == e)
                if mask.any():
                    routed_out[mask] += w[mask] * self.routed_experts[e](x_flat[mask])

        if self.training:
            with torch.no_grad():
                count = torch.bincount(topk_idx.view(-1), minlength=self.n_routed).float()
                count = count / (B * T)
                self.routing_bias += self.bias_update_rate * (count.mean() - count)

        return (shared_out + routed_out).view(B, T, C)