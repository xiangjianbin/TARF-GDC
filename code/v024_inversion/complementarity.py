"""TARF and a paired, bidirectional cross-role branch (no depth tokens)."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class AnchoredResidualFusion(nn.Module):
    def __init__(self, width: int, role_width: int) -> None:
        super().__init__()
        joined = width + role_width
        self.residuals = nn.ModuleList([
            nn.Sequential(nn.Conv2d(joined, width, 1), nn.GELU(),
                          nn.Conv2d(width, width, 1)) for _ in range(2)
        ])
        self.gates = nn.ModuleList([nn.Conv2d(joined, 1, 1) for _ in range(2)])
        self.reset_special_initialization()
        self.last_gate_mean = torch.tensor(0.0)

    def reset_special_initialization(self) -> None:
        for residual in self.residuals:
            nn.init.zeros_(residual[-1].weight)
            nn.init.zeros_(residual[-1].bias)
        for gate in self.gates:
            nn.init.zeros_(gate.bias)

    def forward(self, anchor, spin1, spin2):
        joins = [torch.cat([anchor, role], dim=1) for role in (spin1, spin2)]
        gates = [torch.sigmoid(layer(value)) for layer, value in zip(self.gates, joins)]
        updates = [gate * residual(value) for gate, residual, value in
                   zip(gates, self.residuals, joins)]
        self.last_gate_mean = torch.stack([g.detach().float().mean() for g in gates]).mean()
        return anchor + 0.25 * (updates[0] + updates[1])


class RoleTokens(nn.Module):
    """33x33 observations -> 8x8 overlapping patches, preserving role pairs."""
    def __init__(self, role_width: int, dim: int = 32) -> None:
        super().__init__()
        self.projects = nn.ModuleList([nn.Conv2d(role_width, dim, 1) for _ in range(3)])
        self.roles = nn.Embedding(3, dim)
        self.position = nn.Linear(2, dim, bias=False)
        self.coverage = nn.Linear(1, dim, bias=False)
        # Explicitly expose spin-1/spin-2 magnitude and contrast.  In the old
        # branch these quantities were only implicit in two independent
        # convolutions, making the Transformer easy to bypass.
        self.pair_project = nn.Conv2d(2, dim, 1)
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, 8), torch.linspace(-1, 1, 8), indexing='ij')
        self.register_buffer('coordinates', torch.stack([xx, yy], dim=-1).reshape(64, 2))

    def forward(self, features, mask, role_image=None):
        if mask.shape[-2:] != (33, 33):
            raise ValueError('This frozen patch layout requires 33x33 receivers')
        coverage = F.avg_pool2d(mask.float(), 5, stride=4)
        tokens = []
        position = self.position(self.coordinates.to(features[0].dtype))
        fraction = coverage.flatten(2).transpose(1, 2)
        pair_tokens = None
        if role_image is not None:
            pair = torch.cat([
                (role_image[:, 1:3].square().sum(dim=1, keepdim=True) + 1e-8).sqrt(),
                (role_image[:, 3:5].square().sum(dim=1, keepdim=True) + 1e-8).sqrt(),
            ], dim=1)
            pair = F.avg_pool2d(pair * mask.to(pair.dtype), 5, stride=4)
            pair = pair / coverage.to(pair.dtype).clamp_min(1e-6)
            pair_tokens = self.pair_project(pair).flatten(2).transpose(1, 2)
        for index, (project, feature) in enumerate(zip(self.projects, features)):
            pooled = F.avg_pool2d(feature * mask.to(feature.dtype), 5, stride=4)
            pooled = pooled / coverage.to(feature.dtype).clamp_min(1e-6)
            token = project(pooled).flatten(2).transpose(1, 2)
            token = token + self.roles.weight[index] + position + self.coverage(fraction.to(token.dtype))
            if pair_tokens is not None:
                token = token + pair_tokens
            tokens.append(token)
        return tokens, fraction[..., 0] > 0


class CrossRoleBlock(nn.Module):
    """Each role attends to the other two roles on the shared 8x8 grid."""
    def __init__(self, dim=32, heads=4):
        super().__init__()
        self.heads, self.head_dim = heads, dim // heads
        self.q_norm, self.kv_norm = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.q, self.k, self.v, self.out = [nn.Linear(dim, dim) for _ in range(4)]
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, 64), nn.GELU(), nn.Linear(64, dim))
        self.relative_bias = nn.Embedding(15 * 15, heads)
        yy, xx = torch.meshgrid(torch.arange(8), torch.arange(8), indexing='ij')
        coords = torch.stack([yy, xx], dim=-1).reshape(64, 2)
        offset = coords[:, None] - coords[None, :]
        index = (offset[..., 0] + 7) * 15 + offset[..., 1] + 7
        self.register_buffer('relative_index', index.repeat(1, 2))
        self.reset_special_initialization()

    def reset_special_initialization(self):
        nn.init.zeros_(self.relative_bias.weight)

    def _attend(self, query_tokens, key_tokens, valid):
        batch = query_tokens.shape[0]
        query = self.q_norm(query_tokens)
        aux = self.kv_norm(torch.cat(key_tokens, dim=1))
        def heads(value):
            return value.reshape(batch, -1, self.heads, self.head_dim).transpose(1, 2)
        q, k, v = heads(self.q(query)), heads(self.k(aux)), heads(self.v(aux))
        keys_valid = valid.repeat(1, 2)
        # A dummy valid key prevents undefined all-masked softmax. The whole
        # sample update is zeroed by RelationBranch for that case.
        safe_keys = keys_valid.clone()
        safe_keys[:, 0] |= ~safe_keys.any(dim=1)
        bias = self.relative_bias(self.relative_index).permute(2, 0, 1)
        logits = (q.float() @ k.float().transpose(-1, -2)) * self.head_dim**-0.5
        logits = logits + bias.float().unsqueeze(0)
        logits = logits.masked_fill(~safe_keys[:, None, None, :], float('-inf'))
        attention = logits.softmax(dim=-1).to(v.dtype)
        mixed = (attention @ v).transpose(1, 2).reshape(batch, 64, -1)
        result = query_tokens + self.out(mixed)
        return (result + self.ffn(self.ffn_norm(result))) * valid[..., None]

    def forward(self, tokens, valid):
        outputs = []
        for index in range(3):
            outputs.append(self._attend(tokens[index],
                                        [tokens[j] for j in range(3) if j != index], valid))
        # Keep all three role updates until the decoder projection, rather
        # than reducing to the scalar role before attention.
        return torch.cat(outputs, dim=-1)


class DilatedRoleBlock(nn.Module):
    """Capacity control: dense 15x15 receptive field on the same 8x8 grid."""
    def __init__(self, target_parameters: int, dim=32):
        super().__init__()
        # Counts include biases and a pointwise residual FFN. Search is only
        # integer capacity matching before training, never outcome tuning.
        def count(hidden, expansion):
            return (3*dim*hidden*9+hidden + hidden*hidden*9+hidden
                    + hidden*dim*9+dim + dim*expansion+expansion+expansion*dim+dim)
        candidates = [(abs(count(h, e)-target_parameters), h, e)
                      for h in range(4, 33) for e in range(4, 65)]
        _, hidden, expansion = min(candidates)
        self.widths = (hidden, expansion)
        self.spatial = nn.Sequential(
            nn.Conv2d(3*dim, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=2, dilation=2), nn.GELU(),
            nn.Conv2d(hidden, dim, 3, padding=4, dilation=4),
        )
        self.ffn = nn.Sequential(nn.Conv2d(dim, expansion, 1), nn.GELU(), nn.Conv2d(expansion, dim, 1))

    def forward(self, tokens, valid):
        grids = [token.transpose(1, 2).reshape(token.shape[0], 32, 8, 8)
                 * valid[:, None].reshape(-1, 1, 8, 8) for token in tokens]
        result = grids[0] + self.spatial(torch.cat(grids, dim=1))
        return (result + self.ffn(result)).flatten(2).transpose(1, 2)


class RelationBranch(nn.Module):
    def __init__(self, role_width: int, output_width: int, kind: str):
        super().__init__()
        self.tokens = RoleTokens(role_width)
        if kind == 'transformer':
            self.block = CrossRoleBlock()
        elif kind == 'conv':
            target = (sum(p.numel() for p in CrossRoleBlock().parameters())
                      + 96 * 32 + 32)  # matched transformer merge projection
            self.block = DilatedRoleBlock(target)
        else:
            raise ValueError(kind)
        self.kind = kind
        self.merge = nn.Identity() if kind == 'conv' else nn.Linear(96, 32)
        self.project = nn.Conv2d(32, output_width, 1)
        self.gate = nn.Conv2d(output_width, output_width, 1)
        self.branch_on = True
        self.reset_special_initialization()
        self.last_gate_mean = torch.tensor(0.0)
        self.last_update_rms = torch.tensor(0.0)

    def reset_special_initialization(self):
        # A zero output projection makes Q/K/V/FFN receive no gradient at
        # step 1.  A small Xavier projection keeps the branch trainable from
        # the beginning without destabilising the frozen backbone.
        nn.init.xavier_uniform_(self.project.weight, gain=0.1)
        nn.init.zeros_(self.project.bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -2.0)

    def forward(self, features, mask, role_image=None):
        if not self.branch_on:
            return features[0].new_zeros(features[0].shape[0], self.project.out_channels, 8, 8)
        tokens, valid = self.tokens(features, mask, role_image)
        result = self.block(tokens, valid)
        result = self.merge(result).transpose(1, 2).reshape(-1, 32, 8, 8)
        update = self.project(result)
        gate = torch.sigmoid(self.gate(update))
        self.last_gate_mean = gate.detach().float().mean()
        self.last_update_rms = update.detach().float().square().mean().sqrt()
        return 0.25 * gate * update * valid.any(dim=1)[:, None, None, None]
