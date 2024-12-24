# --------------------------------------------------------
# Elastic Attention Transformer for Robust Single-Branch Object Tracking
# Modified by Lishen Wang
# --------------------------------------------------------
import math
import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple

from lib.models.layers.eat_blocks import *


class LayerScale(nn.Module):
    """ Implements LayerScale to normalize feature weights. """

    def __init__(self, dim: int, inplace: bool = False, init_values: float = 1e-5):
        super().__init__()
        self.inplace = inplace
        self.weight = nn.Parameter(torch.ones(dim) * init_values)

    def forward(self, x):
        if self.inplace:
            return x.mul_(self.weight.view(-1, 1, 1))
        else:
            return x * self.weight.view(-1, 1, 1)


class TransformerStage(nn.Module):
    """ Transformer Stage with support for dual embedding and multi-type attention. """

    def __init__(self,
                 fmap_size,
                 window_size,
                 ns_per_pt,
                 dim_in,
                 dim_embed,
                 depths,
                 stage_spec,
                 n_groups,
                 use_pe,
                 sr_ratio,
                 heads,
                 heads_q,
                 stride,
                 offset_range_factor,
                 dwc_pe,
                 no_off,
                 fixed_pe,
                 attn_drop,
                 proj_drop,
                 expansion,
                 drop,
                 drop_path_rate,
                 ksize,
                 layer_scale_value,
                 use_lpu,
                 log_cpb,
                 dual_embedding=False):

        super().__init__()
        fmap_size = to_2tuple(fmap_size)
        self.depths = depths
        self.dual_embedding = dual_embedding
        hc = dim_embed // heads
        assert dim_embed == heads * hc, "Embedding dimension must be divisible by number of heads."

        self.proj = nn.Conv2d(dim_in, dim_embed, 1, 1, 0) if dim_in != dim_embed else nn.Identity()
        self.stage_spec = stage_spec
        self.use_lpu = use_lpu

        if dual_embedding:
            self.global_proj = nn.Linear(dim_embed, dim_embed)
            self.local_proj = nn.Linear(dim_embed, dim_embed)
            self.global_pos_emb = nn.Parameter(torch.zeros(1, fmap_size[0] * fmap_size[1], dim_embed))
            self.local_pos_emb = nn.Parameter(torch.zeros(1, fmap_size[0] * fmap_size[1], dim_embed))
            nn.init.trunc_normal_(self.global_pos_emb, std=0.02)
            nn.init.trunc_normal_(self.local_pos_emb, std=0.02)

        self.ln_cnvnxt = nn.ModuleDict(
            {str(d): LayerNormProxy(dim_embed) for d in range(depths) if stage_spec[d] == 'X'}
        )
        self.layer_norms = nn.ModuleList(
            [LayerNormProxy(dim_embed) if stage_spec[d // 2] != 'X' else nn.Identity() for d in range(2 * depths)]
        )

        mlp_fn = TransformerMLP
        self.mlps = nn.ModuleList(
            [
                mlp_fn(dim_embed, expansion, drop) for _ in range(depths)
            ]
        )
        self.attns = nn.ModuleList()
        self.drop_path = nn.ModuleList()
        self.layer_scales = nn.ModuleList(
            [
                LayerScale(dim_embed, init_values=layer_scale_value) if layer_scale_value > 0.0 else nn.Identity()
                for _ in range(2 * depths)
            ]
        )
        self.local_perception_units = nn.ModuleList(
            [
                nn.Conv2d(dim_embed, dim_embed, kernel_size=3, stride=1, padding=1,
                          groups=dim_embed) if use_lpu else nn.Identity()
                for _ in range(depths)
            ]
        )

        for i in range(depths):
            if stage_spec[i] == 'L':
                self.attns.append(
                    LocalAttention(dim_embed, heads, window_size, attn_drop, proj_drop)
                )
            elif stage_spec[i] == 'E':
                self.attns.append(
                    EAttentionBaseline(fmap_size, fmap_size, heads,
                                       hc, n_groups, attn_drop, proj_drop,
                                       stride, offset_range_factor, use_pe, dwc_pe,
                                       no_off, fixed_pe, ksize, log_cpb)
                )
            elif stage_spec[i] == 'S':
                shift_size = math.ceil(window_size / 2)
                self.attns.append(
                    ShiftWindowAttention(dim_embed, heads, window_size, attn_drop, proj_drop, shift_size, fmap_size)
                )
            else:
                raise NotImplementedError(f'Spec: {stage_spec[i]} is not supported.')

            self.drop_path.append(DropPath(drop_path_rate[i]) if drop_path_rate[i] > 0.0 else nn.Identity())

    def forward(self, x):
        x = self.proj(x)

        if self.dual_embedding:
            B, C, H, W = x.size()
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
            global_emb = self.global_proj(x) + self.global_pos_emb
            local_emb = self.local_proj(x) + self.local_pos_emb
            x = global_emb + local_emb  # Combine embeddings
            x = x.transpose(1, 2).view(B, C, H, W)  # BNC -> BCHW

        for d in range(self.depths):

            if self.use_lpu:
                x0 = x
                x = self.local_perception_units[d](x.contiguous())
                x = x + x0

            if self.stage_spec[d] == 'X':
                x0 = x
                x = self.attns[d](x)
                x = self.mlps[d](self.ln_cnvnxt[str(d)](x))
                x = self.drop_path[d](x) + x0
            else:
                x0 = x
                x, pos, ref = self.attns[d](self.layer_norms[2 * d](x))
                x = self.layer_scales[2 * d](x)
                x = self.drop_path[d](x) + x0
                x0 = x
                x = self.mlps[d](self.layer_norms[2 * d + 1](x))
                x = self.layer_scales[2 * d + 1](x)
                x = self.drop_path[d](x) + x0

        return x