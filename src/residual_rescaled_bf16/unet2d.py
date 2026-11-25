"""Config-driven Diffusers-0.0.4 Exact UNet2DModel for Pretrained DDPM Checkpoints.

Supports exact `strict=True` weight loading for official HuggingFace checkpoints including:
  - `google/ddpm-cifar10-32`   (32x32,  35.75M parameters)
  - `google/ddpm-celebahq-256` (256x256, 113.67M parameters)
with mixed-precision FP32 GroupNorm(32, eps=1e-6) and Flash/SDPA self-attention.
"""

from __future__ import annotations

import json
import math
import os
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int = 128,
    flip_sin_to_cos: bool = False,
    freq_shift: float = 1.0,
) -> torch.Tensor:
    """Sinusoidal positional timestep embedding matching diffusers UNet2DModel."""
    half_dim = embedding_dim // 2
    exponent = -math.log(10000.0) * torch.arange(
        0, half_dim, dtype=torch.float32, device=timesteps.device
    ) / (half_dim - freq_shift)
    emb = timesteps[:, None].float() * torch.exp(exponent)[None, :]
    if flip_sin_to_cos:
        return torch.cat([torch.cos(emb), torch.sin(emb)], dim=-1)
    return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


class FP32GroupNorm(nn.GroupNorm):
    """Executes GroupNorm statistics in FP32 and casts back to activation dtype."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.group_norm(
            x.float(),
            self.num_groups,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        ).to(x.dtype)


class TimestepEmbedding(nn.Module):
    def __init__(self, channel: int = 128, time_embed_dim: int = 512):
        super().__init__()
        self.linear_1 = nn.Linear(channel, time_embed_dim)
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        return self.linear_2(F.silu(self.linear_1(sample)))


class ResnetBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int = 512,
        eps: float = 1e-6,
        groups: int = 32,
    ):
        super().__init__()
        self.norm1 = FP32GroupNorm(groups, in_channels, eps=eps)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.time_emb_proj = nn.Linear(temb_channels, out_channels)
        self.norm2 = FP32GroupNorm(groups, out_channels, eps=eps)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv_shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
            if in_channels != out_channels
            else None
        )

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_emb_proj(F.silu(temb))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        if self.conv_shortcut is not None:
            x = self.conv_shortcut(x)
        return x + h


class AttentionBlock2D(nn.Module):
    def __init__(self, channels: int, groups: int = 32, eps: float = 1e-6):
        super().__init__()
        self.group_norm = FP32GroupNorm(groups, channels, eps=eps)
        self.query = nn.Linear(channels, channels)
        self.key = nn.Linear(channels, channels)
        self.value = nn.Linear(channels, channels)
        self.proj_attn = nn.Linear(channels, channels)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, C, H, W = hidden_states.shape
        h = self.group_norm(hidden_states).reshape(B, C, H * W).transpose(1, 2)
        attn_out = F.scaled_dot_product_attention(self.query(h), self.key(h), self.value(h))
        return hidden_states + self.proj_attn(attn_out).transpose(1, 2).reshape(B, C, H, W)


class Downsample2D(nn.Module):
    def __init__(self, channels: int, padding: int = 0):
        super().__init__()
        self.padding = padding
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.padding == 0:
            x = F.pad(x, (0, 1, 0, 1), mode="constant", value=0.0)
        return self.conv(x)


class Upsample2D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.interpolate(x, scale_factor=2.0, mode="nearest"))


class DownBlock2D(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        temb_ch: int = 512,
        num_layers: int = 2,
        add_downsample: bool = True,
        add_attn: bool = False,
        norm_eps: float = 1e-6,
        norm_groups: int = 32,
        downsample_padding: int = 0,
    ):
        super().__init__()
        self.resnets = nn.ModuleList(
            [
                ResnetBlock2D(
                    in_ch if i == 0 else out_ch,
                    out_ch,
                    temb_channels=temb_ch,
                    eps=norm_eps,
                    groups=norm_groups,
                )
                for i in range(num_layers)
            ]
        )
        self.attentions = (
            nn.ModuleList(
                [AttentionBlock2D(out_ch, groups=norm_groups, eps=norm_eps) for _ in range(num_layers)]
            )
            if add_attn
            else None
        )
        self.downsamplers = (
            nn.ModuleList([Downsample2D(out_ch, padding=downsample_padding)])
            if add_downsample
            else None
        )

    def forward(
        self, hidden_states: torch.Tensor, temb: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        output_states: tuple[torch.Tensor, ...] = ()
        for i, resnet in enumerate(self.resnets):
            hidden_states = resnet(hidden_states, temb)
            if self.attentions is not None:
                hidden_states = self.attentions[i](hidden_states)
            output_states = output_states + (hidden_states,)
        if self.downsamplers is not None:
            for ds in self.downsamplers:
                hidden_states = ds(hidden_states)
            output_states = output_states + (hidden_states,)
        return hidden_states, output_states


class UNetMidBlock2D(nn.Module):
    def __init__(
        self,
        channels: int,
        temb_ch: int = 512,
        norm_eps: float = 1e-6,
        norm_groups: int = 32,
    ):
        super().__init__()
        self.resnets = nn.ModuleList(
            [
                ResnetBlock2D(channels, channels, temb_channels=temb_ch, eps=norm_eps, groups=norm_groups),
                ResnetBlock2D(channels, channels, temb_channels=temb_ch, eps=norm_eps, groups=norm_groups),
            ]
        )
        self.attentions = nn.ModuleList([AttentionBlock2D(channels, groups=norm_groups, eps=norm_eps)])

    def forward(self, hidden_states: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        hidden_states = self.attentions[0](self.resnets[0](hidden_states, temb))
        return self.resnets[1](hidden_states, temb)


class UpBlock2D(nn.Module):
    def __init__(
        self,
        prev_out_ch: int,
        out_ch: int,
        skip_chs: Sequence[int],
        temb_ch: int = 512,
        add_upsample: bool = True,
        add_attn: bool = False,
        norm_eps: float = 1e-6,
        norm_groups: int = 32,
    ):
        super().__init__()
        self.resnets = nn.ModuleList(
            [
                ResnetBlock2D(
                    (prev_out_ch if i == 0 else out_ch) + skip_chs[i],
                    out_ch,
                    temb_channels=temb_ch,
                    eps=norm_eps,
                    groups=norm_groups,
                )
                for i in range(len(skip_chs))
            ]
        )
        self.attentions = (
            nn.ModuleList(
                [AttentionBlock2D(out_ch, groups=norm_groups, eps=norm_eps) for _ in range(len(skip_chs))]
            )
            if add_attn
            else None
        )
        self.upsamplers = nn.ModuleList([Upsample2D(out_ch)]) if add_upsample else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        res_samples: tuple[torch.Tensor, ...],
        temb: torch.Tensor,
    ) -> torch.Tensor:
        for i, resnet in enumerate(self.resnets):
            res_hidden = res_samples[-1]
            res_samples = res_samples[:-1]
            hidden_states = resnet(torch.cat([hidden_states, res_hidden], dim=1), temb)
            if self.attentions is not None:
                hidden_states = self.attentions[i](hidden_states)
        if self.upsamplers is not None:
            for us in self.upsamplers:
                hidden_states = us(hidden_states)
        return hidden_states


class DiffusersUNet2D(nn.Module):
    """Exact config-driven UNet2DModel compatible with HuggingFace DDPM checkpoints."""

    def __init__(
        self,
        sample_size: int = 32,
        in_channels: int = 3,
        out_channels: int = 3,
        block_out_channels: Sequence[int] = (128, 256, 256, 256),
        down_block_types: Sequence[str] = (
            "DownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
        ),
        up_block_types: Sequence[str] = (
            "UpBlock2D",
            "UpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
        ),
        layers_per_block: int = 2,
        downsample_padding: int = 0,
        flip_sin_to_cos: bool = False,
        freq_shift: float = 1.0,
        norm_eps: float = 1e-6,
        norm_num_groups: int = 32,
    ):
        super().__init__()
        self.sample_size = sample_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_ch = block_out_channels[0]
        self.temb_ch = self.base_ch * 4
        self.flip_sin_to_cos = flip_sin_to_cos
        self.freq_shift = freq_shift

        self.conv_in = nn.Conv2d(in_channels, self.base_ch, kernel_size=3, padding=1)
        self.time_embedding = TimestepEmbedding(self.base_ch, self.temb_ch)

        self.down_blocks = nn.ModuleList()
        skip_channels: list[int] = [self.base_ch]
        out_ch = self.base_ch
        for i, db_type in enumerate(down_block_types):
            in_ch = out_ch
            out_ch = block_out_channels[i]
            is_final = i == len(block_out_channels) - 1
            add_attn = "Attn" in db_type
            self.down_blocks.append(
                DownBlock2D(
                    in_ch=in_ch,
                    out_ch=out_ch,
                    temb_ch=self.temb_ch,
                    num_layers=layers_per_block,
                    add_downsample=not is_final,
                    add_attn=add_attn,
                    norm_eps=norm_eps,
                    norm_groups=norm_num_groups,
                    downsample_padding=downsample_padding,
                )
            )
            for _ in range(layers_per_block):
                skip_channels.append(out_ch)
            if not is_final:
                skip_channels.append(out_ch)

        self.mid_block = UNetMidBlock2D(
            block_out_channels[-1],
            temb_ch=self.temb_ch,
            norm_eps=norm_eps,
            norm_groups=norm_num_groups,
        )

        self.up_blocks = nn.ModuleList()
        rev_out = list(reversed(block_out_channels))
        prev_out_ch = rev_out[0]
        for i, ub_type in enumerate(up_block_types):
            out_ch = rev_out[i]
            is_final = i == len(block_out_channels) - 1
            add_attn = "Attn" in ub_type
            skips_for_block = [skip_channels.pop() for _ in range(layers_per_block + 1)]
            self.up_blocks.append(
                UpBlock2D(
                    prev_out_ch=prev_out_ch,
                    out_ch=out_ch,
                    skip_chs=skips_for_block,
                    temb_ch=self.temb_ch,
                    add_upsample=not is_final,
                    add_attn=add_attn,
                    norm_eps=norm_eps,
                    norm_groups=norm_num_groups,
                )
            )
            prev_out_ch = out_ch

        self.conv_norm_out = FP32GroupNorm(norm_num_groups, block_out_channels[0], eps=norm_eps)
        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, kernel_size=3, padding=1)

    @classmethod
    def from_pretrained_dir(cls, ckpt_dir: str) -> "DiffusersUNet2D":
        """Instantiate and load weights (`strict=True`) from a local HuggingFace checkpoint directory."""
        cfg_path = os.path.join(ckpt_dir, "config.json")
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        model = cls(
            sample_size=cfg["sample_size"],
            in_channels=cfg.get("in_channels", 3),
            out_channels=cfg.get("out_channels", 3),
            block_out_channels=cfg["block_out_channels"],
            down_block_types=cfg["down_block_types"],
            up_block_types=cfg["up_block_types"],
            layers_per_block=cfg.get("layers_per_block", 2),
            downsample_padding=cfg.get("downsample_padding", 0),
            flip_sin_to_cos=cfg.get("flip_sin_to_cos", False),
            freq_shift=float(cfg.get("freq_shift", 1.0)),
            norm_eps=float(cfg.get("norm_eps", 1e-6)),
            norm_num_groups=int(cfg.get("norm_num_groups", 32)),
        )
        st_path = os.path.join(ckpt_dir, "diffusion_pytorch_model.safetensors")
        bin_path = os.path.join(ckpt_dir, "diffusion_pytorch_model.bin")
        if os.path.exists(st_path):
            from safetensors.torch import load_file

            sd = load_file(st_path)
        elif os.path.exists(bin_path):
            sd = torch.load(bin_path, map_location="cpu", weights_only=True)
        else:
            raise FileNotFoundError(f"No model weights found in {ckpt_dir}")
        model.load_state_dict(sd, strict=True)
        return model

    def forward(self, sample: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        dtp = sample.dtype
        emb = self.time_embedding(
            get_timestep_embedding(
                timesteps,
                self.base_ch,
                flip_sin_to_cos=self.flip_sin_to_cos,
                freq_shift=self.freq_shift,
            ).to(dtp)
        )
        sample = self.conv_in(sample)
        down_block_res_samples: tuple[torch.Tensor, ...] = (sample,)
        for downsample_block in self.down_blocks:
            sample, res_samples = downsample_block(sample, emb)
            down_block_res_samples = down_block_res_samples + res_samples
        sample = self.mid_block(sample, emb)
        for upsample_block in self.up_blocks:
            n_res = len(upsample_block.resnets)
            res_samples = down_block_res_samples[-n_res:]
            down_block_res_samples = down_block_res_samples[:-n_res]
            sample = upsample_block(sample, res_samples, emb)
        return self.conv_out(F.silu(self.conv_norm_out(sample)))
