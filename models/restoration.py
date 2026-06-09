from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.coords import make_voxel_grid
from models.inr_primitives import DeltaNBoundedHead, FiLMLayer, FourierFeatures

class IdentityRestorer(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

class SliceAutoEncoder2D(nn.Module):

    def __init__(self, base_channels: int = 32):
        super().__init__()
        c = base_channels
        self.encoder = nn.Sequential(
            nn.Conv2d(1, c, 3, padding=1),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(c, 2 * c, 3, padding=1),
            nn.BatchNorm2d(2 * c),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2 * c, c, 2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(c, c, 2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, 1, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, ch, z, h, w = x.shape
        y = x.permute(0, 2, 1, 3, 4).reshape(b * z, ch, h, w)
        y = self.decoder(self.encoder(y))
        y = y.reshape(b, z, 1, h, w).permute(0, 2, 1, 3, 4)
        return y.sigmoid()

class ConvBlock3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class SkipAutoEncoder3D(nn.Module):

    def __init__(self, base_channels: int = 16):
        super().__init__()
        c = base_channels
        self.enc1 = ConvBlock3D(1, c)
        self.enc2 = ConvBlock3D(c, 2 * c)
        self.mid = ConvBlock3D(2 * c, 4 * c)
        self.up2 = nn.ConvTranspose3d(4 * c, 2 * c, 2, stride=2)
        self.dec2 = ConvBlock3D(4 * c, 2 * c)
        self.up1 = nn.ConvTranspose3d(2 * c, c, 2, stride=2)
        self.dec1 = ConvBlock3D(2 * c, c)
        self.out = nn.Conv3d(c, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool3d(e1, 2))
        m = self.mid(F.max_pool3d(e2, 2))
        d2 = self.up2(m)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.out(d1).sigmoid()

class SmallUNet3D(nn.Module):

    def __init__(self, base_channels: int = 16):
        super().__init__()
        c = base_channels
        self.enc1 = ConvBlock3D(1, c)
        self.enc2 = ConvBlock3D(c, 2 * c)
        self.enc3 = ConvBlock3D(2 * c, 4 * c)
        self.mid = ConvBlock3D(4 * c, 8 * c)
        self.up3 = nn.ConvTranspose3d(8 * c, 4 * c, 2, stride=2)
        self.dec3 = ConvBlock3D(8 * c, 4 * c)
        self.up2 = nn.ConvTranspose3d(4 * c, 2 * c, 2, stride=2)
        self.dec2 = ConvBlock3D(4 * c, 2 * c)
        self.up1 = nn.ConvTranspose3d(2 * c, c, 2, stride=2)
        self.dec1 = ConvBlock3D(2 * c, c)
        self.out = nn.Conv3d(c, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool3d(e1, 2))
        e3 = self.enc3(F.max_pool3d(e2, 2))
        m = self.mid(F.max_pool3d(e3, 2))
        d3 = self.dec3(torch.cat([self.up3(m), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out(d1).sigmoid()

class StackViTEncoder(nn.Module):

    def __init__(
        self,
        in_chans: int,
        img_size: int,
        patch_size: int,
        latent_dim: int,
        embed_dim: int = 192,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError(f"img_size={img_size} must be divisible by patch_size={patch_size}")
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, patch_size, stride=patch_size)
        num_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.latent_proj = nn.Linear(embed_dim, latent_dim)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(h.shape[0], -1, -1)
        h = torch.cat([cls, h], dim=1) + self.pos_embed
        h = self.blocks(h)
        return self.latent_proj(self.norm(h[:, 0]))

class LocalFiLMBranch(nn.Module):

    def __init__(self, in_dim: int, cond_dim: int, hidden: int, depth: int, out_dim: int):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(depth):
            layers.append(FiLMLayer(d, hidden, cond_dim))
            d = hidden
        self.layers = nn.ModuleList(layers)
        self.head = nn.Linear(hidden, out_dim)

    def forward(self, feats: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        h = feats
        for layer in self.layers:
            h = layer(h, z)
        return self.head(h)

class VolumeMultiHeadINR(nn.Module):

    def __init__(
        self,
        latent_dim: int,
        hidden: int = 128,
        depth: int = 6,
        fourier_features: int = 32,
        fourier_sigma: float = 2.0,
        delta_n_max: float = 1.0,
        local_dim: int = 0,
    ):
        super().__init__()
        z_split = latent_dim // 2
        self.fourier = FourierFeatures(3, fourier_features, fourier_sigma)
        self.delta_n_head = DeltaNBoundedHead(delta_n_max, scale=3.0)
        feat_dim = self.fourier.out_dim + int(local_dim)
        self.branch_aux = LocalFiLMBranch(feat_dim, z_split, hidden, depth, 1)
        nn.init.zeros_(self.branch_aux.head.bias)
        self.branch_rho = LocalFiLMBranch(feat_dim, latent_dim - z_split, hidden, depth, 1)
        nn.init.constant_(self.branch_rho.head.bias, -1.0)

    def forward(
        self,
        coords: torch.Tensor,
        z: torch.Tensor,
        local_feats: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z_ri, z_rho = z.chunk(2, dim=-1)
        feats = self.fourier(coords)
        if local_feats is not None:
            feats = torch.cat([feats, local_feats], dim=-1)
        aux = self.delta_n_head(self.branch_aux(feats, z_ri))
        rho = F.softplus(self.branch_rho(feats, z_rho))
        return aux, rho

class ViTMultiHeadINRRestorer(nn.Module):

    def __init__(
        self,
        Nz: int = 64,
        Ny: int = 64,
        Nx: int = 64,
        latent_dim: int = 256,
        patch_size: int = 8,
        vit_model_name: str = "stack_vit_tiny",
        vit_embed_dim: int = 192,
        vit_depth: int = 4,
        vit_heads: int = 4,
        inr_hidden: int = 128,
        inr_depth: int = 6,
        fourier_features: int = 32,
        fourier_sigma: float = 2.0,
        delta_n_max: float = 1.0,
        decode_chunk: int = 65536,
        local_feature_dim: int = 8,
        local_feature_dropout: float = 0.0,
    ):
        super().__init__()
        self.Nz, self.Ny, self.Nx = int(Nz), int(Ny), int(Nx)
        self.decode_chunk = int(decode_chunk)
        self.local_feature_dropout = float(local_feature_dropout)
        if vit_model_name not in {"stack_vit_tiny", "pure_torch_vit"}:
            raise ValueError(
                f"unsupported vit_model_name={vit_model_name!r}; use 'stack_vit_tiny' "
                "to avoid timm/torchvision dependency issues"
            )
        self.encoder = StackViTEncoder(
            in_chans=self.Nz,
            img_size=self.Ny,
            patch_size=patch_size,
            latent_dim=latent_dim,
            embed_dim=vit_embed_dim,
            depth=vit_depth,
            num_heads=vit_heads,
        )
        self.local_feature_dim = int(local_feature_dim)
        self.local_stem = nn.Sequential(
            nn.Conv3d(1, self.local_feature_dim, 3, padding=1),
            nn.GroupNorm(1, self.local_feature_dim),
            nn.GELU(),
            nn.Conv3d(self.local_feature_dim, self.local_feature_dim, 3, padding=1),
            nn.GELU(),
        )
        self.decoder = VolumeMultiHeadINR(
            latent_dim=latent_dim,
            hidden=inr_hidden,
            depth=inr_depth,
            fourier_features=fourier_features,
            fourier_sigma=fourier_sigma,
            delta_n_max=delta_n_max,
            local_dim=self.local_feature_dim + 1,
        )
        grid = make_voxel_grid(self.Nz, self.Ny, self.Nx, device="cpu")
        self.register_buffer("voxel_grid", grid, persistent=False)
        self.last_aux_delta: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        stack = x[:, 0]
        z = self.encoder(stack)
        B = stack.shape[0]
        coords = self.voxel_grid.to(x.device).unsqueeze(0).expand(B, -1, -1)
        local_volume = torch.cat([x, self.local_stem(x)], dim=1)

        aux_chunks, rho_chunks = [], []
        for c in coords.split(self.decode_chunk, dim=1):
            local_feats = self._sample_local_features(local_volume, c)
            if self.training and self.local_feature_dropout > 0:

                stem = local_feats[..., 1:]
                stem = F.dropout(stem, p=self.local_feature_dropout, training=True)
                local_feats = torch.cat([local_feats[..., :1], stem], dim=-1)
            aux, rho = self.decoder(c, z, local_feats)
            aux_chunks.append(aux)
            rho_chunks.append(rho)
        aux_delta = torch.cat(aux_chunks, dim=1).view(B, self.Nz, self.Ny, self.Nx)
        rho = torch.cat(rho_chunks, dim=1).view(B, self.Nz, self.Ny, self.Nx)
        self.last_aux_delta = aux_delta
        corrected = 1.0 - torch.exp(-rho.clamp_min(0.0))
        return corrected.unsqueeze(1)

    @staticmethod
    def _sample_local_features(feature_volume: torch.Tensor, coords_zyx: torch.Tensor) -> torch.Tensor:

        grid = coords_zyx[..., [2, 1, 0]].view(coords_zyx.shape[0], coords_zyx.shape[1], 1, 1, 3)
        sampled = F.grid_sample(feature_volume, grid, mode="bilinear", align_corners=True)
        return sampled[:, :, :, 0, 0].transpose(1, 2).contiguous()

def build_restoration_model(
    name: str,
    base_channels: int = 16,
    Nz: int = 64,
    Ny: int = 64,
    Nx: int = 64,
    **kwargs,
) -> nn.Module:
    name = name.lower()
    if name in {"identity", "none"}:
        return IdentityRestorer()
    if name in {"slice_autoencoder_2d", "ae2d", "paper_ae"}:
        return SliceAutoEncoder2D(base_channels=max(base_channels, 16))
    if name in {"skip_autoencoder_3d", "skip_ae3d", "ours"}:
        return SkipAutoEncoder3D(base_channels=base_channels)
    if name in {"unet3d", "small_unet_3d"}:
        return SmallUNet3D(base_channels=base_channels)
    if name in {"vit_multihead_inr", "vit_inr", "vit_mhinr"}:
        return ViTMultiHeadINRRestorer(
            Nz=Nz,
            Ny=Ny,
            Nx=Nx,
            latent_dim=int(kwargs.get("latent_dim", 256)),
            patch_size=int(kwargs.get("patch_size", 8)),
            vit_model_name=str(kwargs.get("vit_model_name", "stack_vit_tiny")),
            vit_embed_dim=int(kwargs.get("vit_embed_dim", 192)),
            vit_depth=int(kwargs.get("vit_depth", 4)),
            vit_heads=int(kwargs.get("vit_heads", 4)),
            inr_hidden=int(kwargs.get("inr_hidden", 128)),
            inr_depth=int(kwargs.get("inr_depth", 6)),
            fourier_features=int(kwargs.get("fourier_features", 32)),
            fourier_sigma=float(kwargs.get("fourier_sigma", 2.0)),
            delta_n_max=float(kwargs.get("delta_n_max", 1.0)),
            decode_chunk=int(kwargs.get("decode_chunk", 65536)),
            local_feature_dim=int(kwargs.get("local_feature_dim", 8)),
            local_feature_dropout=float(kwargs.get("local_feature_dropout", 0.0)),
        )
    raise ValueError(f"unknown restoration model: {name}")
