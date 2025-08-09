import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.embeddings import TimestepEmbedding, Timesteps


class SPADEGroupNorm(nn.Module):
    def __init__(self, norm_nc, label_nc, eps=1e-5, groups=32, hidden=128):
        super().__init__()
        self.norm = nn.GroupNorm(groups, norm_nc, affine=False, eps=eps)
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, hidden, kernel_size=3, padding=1), nn.ReLU(inplace=True)
        )
        self.mlp_gamma = nn.Conv2d(hidden, norm_nc, kernel_size=3, padding=1)
        self.mlp_beta = nn.Conv2d(hidden, norm_nc, kernel_size=3, padding=1)

    def forward(self, x, seg):
        x = self.norm(x)
        seg = F.interpolate(seg, size=x.shape[2:], mode="nearest")
        h = self.mlp_shared(seg)
        gamma = self.mlp_gamma(h)
        beta = self.mlp_beta(h)
        return x * (1 + gamma) + beta


class SiLU(nn.Module):
    def forward(self, x):
        return F.silu(x)


class SDMResBlock(nn.Module):
    """
    SPADEでsegを注入。timestep埋め込みは線形で注入（FiLM/加算の両対応）。
    """

    def __init__(
        self,
        in_channels,
        emb_channels,
        out_channels=None,
        c_channels=19,
        dropout=0.0,
        use_scale_shift_norm=False,
        up=False,
        down=False,
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        self.use_scale_shift_norm = use_scale_shift_norm
        self.up = up
        self.down = down

        self.in_norm = SPADEGroupNorm(in_channels, c_channels)
        self.in_act = SiLU()
        self.in_conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        if up:
            self.h_up = nn.Upsample(scale_factor=2, mode="nearest")
            self.x_up = nn.Upsample(scale_factor=2, mode="nearest")
        elif down:
            self.h_up = nn.AvgPool2d(2)
            self.x_up = nn.AvgPool2d(2)
        else:
            self.h_up = self.x_up = nn.Identity()

        emb_out = out_channels * (2 if use_scale_shift_norm else 1)
        self.emb = nn.Sequential(SiLU(), nn.Linear(emb_channels, emb_out))

        self.out_norm = SPADEGroupNorm(out_channels, c_channels)
        self.out_act = SiLU()
        self.dropout = nn.Dropout(dropout)
        self.out_conv = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self.skip = (
            nn.Identity()
            if out_channels == in_channels
            else nn.Conv2d(in_channels, out_channels, 1)
        )

    def forward(self, x, seg, emb):
        h = self.in_norm(x, seg)
        h = self.in_act(h)
        if self.up is not False or self.down is not False:
            h = self.h_up(h)
            x = self.x_up(x)
        h = self.in_conv(h)

        emb_out = self.emb(emb).type(h.dtype)
        while emb_out.ndim < h.ndim:
            emb_out = emb_out[..., None, None]
        if self.use_scale_shift_norm:
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = self.out_norm(h, seg)
            h = h * (1 + scale) + shift
            h = self.out_act(h)
        else:
            h = h + emb_out
            h = self.out_norm(h, seg)
            h = self.out_act(h)

        h = self.dropout(h)
        h = self.out_conv(h)
        return self.skip(x) + h


class ResBlock(nn.Module):
    def __init__(self, in_ch, emb_ch, out_ch=None, dropout=0.0, up=False, down=False):
        super().__init__()
        out_ch = out_ch or in_ch
        self.up, self.down = up, down
        self.norm1 = nn.GroupNorm(32, in_ch)
        self.act1 = SiLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        if up:
            self.h_up = nn.Upsample(scale_factor=2, mode="nearest")
            self.x_up = nn.Upsample(scale_factor=2, mode="nearest")
        elif down:
            self.h_up = nn.AvgPool2d(2)
            self.x_up = nn.AvgPool2d(2)
        else:
            self.h_up = self.x_up = nn.Identity()

        self.emb = nn.Sequential(SiLU(), nn.Linear(emb_ch, out_ch))
        self.norm2 = nn.GroupNorm(32, out_ch)
        self.act2 = SiLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Identity() if out_ch == in_ch else nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x, emb):
        h = self.norm1(x)
        h = self.act1(h)
        if self.up or self.down:
            h = self.h_up(h)
            x = self.x_up(x)
        h = self.conv1(h)
        e = self.emb(emb).type(h.dtype)
        while e.ndim < h.ndim:
            e = e[..., None, None]
        h = h + e
        h = self.norm2(h)
        h = self.act2(h)
        h = self.dropout(h)
        h = self.conv2(h)
        return self.skip(x) + h


class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv(self.norm(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = q.reshape(b, self.num_heads, c // self.num_heads, h * w).permute(0, 1, 3, 2)
        k = k.reshape(b, self.num_heads, c // self.num_heads, h * w).permute(0, 1, 3, 2)
        v = v.reshape(b, self.num_heads, c // self.num_heads, h * w).permute(0, 1, 3, 2)
        attn = torch.matmul(q, k.transpose(-1, -2)) * (c // self.num_heads) ** -0.5
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v)
        out = out.permute(0, 1, 3, 2).reshape(b, c, h, w)
        out = self.proj_out(out)
        return x + out


class SDMUNet(nn.Module):
    """
    diffusers互換のSDM UNet:
      forward(x, timesteps, seg) -> pred_eps
    """

    def __init__(
        self,
        in_channels=3,
        base_channels=128,
        channel_mult=(1, 2, 4, 8),
        num_res_blocks=2,
        num_classes=19,
        dropout=0.1,
        use_scale_shift_norm=False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.time_proj = Timesteps(
            base_channels, flip_sin_to_cos=True, downscale_freq_shift=0
        )
        self.time_embed = TimestepEmbedding(base_channels, base_channels * 4)
        self.in_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        ch = base_channels
        emb_ch = base_channels * 4

        # Down (SPADE)
        self.down_blocks = nn.ModuleList()
        self.down_chs = [ch]
        for i, mult in enumerate(channel_mult):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                self.down_blocks.append(
                    SDMResBlock(
                        ch,
                        emb_ch,
                        out_channels=out_ch,
                        c_channels=num_classes,
                        dropout=dropout,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                )
                ch = out_ch
                self.down_chs.append(ch)
            if i != len(channel_mult) - 1:
                self.down_blocks.append(
                    SDMResBlock(
                        ch,
                        emb_ch,
                        c_channels=num_classes,
                        dropout=dropout,
                        use_scale_shift_norm=use_scale_shift_norm,
                        down=True,
                    )
                )
                self.down_chs.append(ch)

        # Middle (SPADE)
        self.mid_block1 = SDMResBlock(
            ch,
            emb_ch,
            c_channels=num_classes,
            dropout=dropout,
            use_scale_shift_norm=use_scale_shift_norm,
        )
        self.mid_attn = AttentionBlock(ch)
        self.mid_block2 = SDMResBlock(
            ch,
            emb_ch,
            c_channels=num_classes,
            dropout=dropout,
            use_scale_shift_norm=use_scale_shift_norm,
        )

        # Up (SPADE)
        self.up_blocks = nn.ModuleList()
        for i, mult in list(enumerate(channel_mult))[::-1]:
            for j in range(num_res_blocks + 1):
                skip_ch = self.down_chs.pop()
                self.up_blocks.append(
                    SDMResBlock(
                        ch + skip_ch,
                        emb_ch,
                        out_channels=base_channels * mult,
                        c_channels=num_classes,
                        dropout=dropout,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                )
                ch = base_channels * mult
            if i != 0:
                self.up_blocks.append(
                    SDMResBlock(
                        ch,
                        emb_ch,
                        c_channels=num_classes,
                        dropout=dropout,
                        use_scale_shift_norm=use_scale_shift_norm,
                        up=True,
                    )
                )

        self.out_norm = nn.GroupNorm(32, ch)
        self.out_act = SiLU()
        self.out_conv = nn.Conv2d(ch, in_channels, 3, padding=1)

    def _downseg(self, seg, h):
        # segをhの解像度にnearestで合わせる
        return F.interpolate(seg, size=h.shape[2:], mode="nearest")

    def forward(self, x, timesteps, seg):
        """
        x: [B,3,H,W], timesteps: [B], seg: [B,Cs,H,W] (one-hot)
        返り値: 予測ε（xと同shape）
        """
        t = self.time_proj(timesteps)
        emb = self.time_embed(t)

        h = self.in_conv(x)
        skips = [h]
        for blk in self.down_blocks:
            h = blk(h, self._downseg(seg, h), emb)
            skips.append(h)

        h = self.mid_block1(h, self._downseg(seg, h), emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, self._downseg(seg, h), emb)

        for blk in self.up_blocks:
            if blk.up is False:
                skip = skips.pop()
                h = torch.cat([h, skip], dim=1)
            h = blk(h, self._downseg(seg, h), emb)

        h = self.out_norm(h)
        h = self.out_act(h)
        return self.out_conv(h)
