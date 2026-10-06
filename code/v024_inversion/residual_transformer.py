"""One bounded density-space refinement from noise-weighted 3-D residual evidence.

No label, clean observation, historical inverse, or frozen network is an input.
Physical buffers are deterministic functions of the fixed operator/noise scales.
"""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F
from .deterministic_resize import resize_trilinear3d


class VolumeAttention(nn.Module):
    def __init__(self, dim=32, heads=4):
        super().__init__()
        if dim % heads:
            raise ValueError('dim must be divisible by heads')
        self.heads, self.head_dim = heads, dim // heads
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3*dim)
        self.out = nn.Linear(dim, dim)
        self.ffn = nn.Sequential(nn.Linear(dim, 2*dim), nn.GELU(), nn.Linear(2*dim, dim))

    def forward(self, tokens):
        batch, length, dim = tokens.shape
        qkv = self.qkv(self.norm1(tokens)).reshape(batch, length, 3, self.heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        logits = (q.float() @ k.float().transpose(-1, -2)) * self.head_dim**-.5
        attention = logits.softmax(-1).to(v.dtype)
        mixed = (attention @ v).transpose(1, 2).reshape(batch, length, dim)
        tokens = tokens + self.out(mixed)
        return tokens + self.ffn(self.norm2(tokens))


class PhysicalResidualTransformer(nn.Module):
    def __init__(self, operator, role_noise_scales, width=16, dim=32, heads=4,
                 blocks=1, correction_bound=.25, density_scale=.85,
                 sensitivity_floor=1e-3, support_gated=False):
        super().__init__()
        if operator is None:
            raise ValueError('Residual refinement requires the actual GravityOperator')
        if tuple(operator.density_shape) != (16,32,32):
            raise ValueError('Registered volume layout is 16x32x32')
        if len(role_noise_scales) != 5 or min(role_noise_scales) <= 0:
            raise ValueError('Five positive physical role noise standard deviations required')
        if correction_bound <= 0 or density_scale <= 0 or not 0 < sensitivity_floor < 1:
            raise ValueError('Invalid refinement scales')
        self.density_shape = tuple(operator.density_shape)
        self.dim, self.width = dim, width
        self.correction_bound, self.density_scale = float(correction_bound), float(density_scale)
        self.support_gated = bool(support_gated)
        self.branch_on = True
        self.collect_diagnostics = False
        self.last_diagnostics = {}
        # K_c = G_role,c / sigma_c. No learned/calibrated/test-derived buffer.
        with torch.no_grad():
            g = operator.g6.detach().float().reshape(operator.n_receivers,6,-1)
            sigma = torch.tensor(role_noise_scales, dtype=torch.float32, device=g.device)
            kernel = torch.einsum('jc,rcv->jrv', operator.role_transform.float(), g)
            kernel.div_(sigma[:,None,None])
            # A fixed full-survey Jacobi preconditioner; mask enters residual.
            diagonal = kernel.square().sum((0,1))
            diagonal = diagonal.clamp_min(diagonal.max()*sensitivity_floor)
            norm_sigma = sigma / operator.role_scales.float()
        self.register_buffer('weighted_kernel', kernel.contiguous(), persistent=False)
        self.register_buffer('diagonal', diagonal, persistent=False)
        self.register_buffer('normalized_sigma', norm_sigma, persistent=False)
        self.local = nn.Sequential(nn.Conv3d(6,width,3,padding=1,bias=False),
                                   nn.GroupNorm(4,width), nn.GELU())
        self.down = nn.Conv3d(width,dim,4,stride=4,bias=False)
        self.position = nn.Linear(3,dim,bias=False)
        zz,yy,xx = torch.meshgrid(torch.linspace(-1,1,4),torch.linspace(-1,1,8),
                                 torch.linspace(-1,1,8),indexing='ij')
        self.register_buffer('coordinates',torch.stack([xx,yy,zz],-1).reshape(256,3))
        self.blocks = nn.ModuleList([VolumeAttention(dim,heads) for _ in range(blocks)])
        self.up_project = nn.Conv3d(dim,width,1)
        self.fuse = nn.Sequential(nn.Conv3d(2*width,width,3,padding=1,bias=False),
                                  nn.GroupNorm(4,width),nn.GELU())
        self.output = nn.Conv3d(width,1,3,padding=1)
        self.reset_special_initialization()

    def reset_special_initialization(self):
        # Exactly preserve rho0 at initialization. Only ONE zero multiplier:
        # output receives gradients on step one; upstream opens after that.
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def weighted_backprojection(self, residual, mask):
        with torch.autocast(device_type=residual.device.type,enabled=False):
            whitened = residual.float() / self.normalized_sigma[None,None,:]
            whitened = whitened * mask.float().unsqueeze(-1)
            # b_j = D^-1 K_j^T (r_j / sigma_j), in density units.
            evidence = torch.bmm(whitened.permute(2,0,1).contiguous(), self.weighted_kernel)
            evidence = evidence.permute(1,0,2) / self.diagonal[None,None,:]
            evidence = evidence.reshape(residual.shape[0],5,*self.density_shape)
        return evidence, whitened

    def forward(self, initial, prepared, operator):
        rho0 = initial['density']
        if not self.branch_on:
            return initial
        with torch.autocast(device_type=rho0.device.type,enabled=False):
            predicted = operator.density_to_normalized(rho0.float(),mode='roles')
            residual = prepared['observed_roles'].float()-predicted
            if getattr(self, 'observation_only', False):
                # Table 6 intervention: do not subtract the initial response.
                residual = prepared['observed_roles'].float()
            evidence, whitened = self.weighted_backprojection(residual,prepared['receiver_mask'])
            # Fixed, monotone compression; no sample labels or outcome tuning.
            features = torch.cat([rho0.float()/self.density_scale,
                                  torch.asinh(evidence/self.density_scale)],1)
        local = self.local(features)
        coarse = self.down(local)
        tokens = coarse.flatten(2).transpose(1,2) + self.position(self.coordinates.to(coarse.dtype))
        for block in self.blocks:
            tokens = block(tokens)
        coarse = tokens.transpose(1,2).reshape(-1,self.dim,4,8,8)
        volume = resize_trilinear3d(self.up_project(coarse),self.density_shape)
        raw = self.output(self.fuse(torch.cat([local,volume],1)))
        # Float32 summation avoids losing a small update when rho0 is bf16.
        valid = prepared['receiver_mask'].bool().any(1).view(-1,1,1,1,1)
        raw_delta = self.correction_bound * torch.tanh(raw.float()) * valid
        if self.support_gated:
            if 'support' not in initial or initial['support'].shape != rho0.shape:
                raise ValueError('Gated correction requires the initial predicted GLIN support')
            gate = initial['support'].detach().float()
            delta = gate * raw_delta
        else:
            gate = torch.ones_like(rho0, dtype=torch.float32)
            delta = raw_delta
        density = rho0.float() + delta
        result = dict(density=density,initial_density=rho0,density_correction=delta,
                      ungated_density_correction=raw_delta, correction_gate=gate)
        # GLIN describes the INITIAL prediction, not the final additive output.
        for key in ('support','support_logits','signed_density'):
            if key in initial:
                result['initial_'+key] = initial[key]
        if self.collect_diagnostics:
            with torch.no_grad():
                def rms(value): return value.detach().float().square().mean().sqrt()
                self.last_diagnostics = dict(
                    residual_noise_rms=float(rms(whitened)),
                    backprojection_rms=float(rms(evidence)),
                    initial_density_rms=float(rms(rho0)),
                    correction_rms=float(rms(delta)),
                    ungated_correction_rms=float(rms(raw_delta)),
                    support_gate_mean=float(gate.mean()),
                    gate_bound_violation=float((delta.abs()-self.correction_bound*gate).clamp_min(0).max()),
                    correction_to_initial_rms=float(rms(delta)/rms(rho0).clamp_min(1e-12)),
                    correction_abs_max=float(delta.detach().abs().max()),
                    correction_saturation_fraction=float((raw.detach().abs()>2).float().mean()))
        return result
