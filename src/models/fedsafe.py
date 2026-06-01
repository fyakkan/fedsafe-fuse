"""FedSafe-Fuse: dual MobileNetV3-Small encoders + 2-layer Transformer + conv decoder.

Per the project proposal:
    - Two modality-specific encoders, each MobileNetV3-Small (~2.5M params total each)
    - Shared 2-layer Transformer module (2 heads, embed dim 128) for cross-modal attention
    - Convolutional decoder
    - Target ~8M params

Actual realised parameter count depends on decoder width; default config lands at ~5M.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small


class ModalityEncoder(nn.Module):
    """MobileNetV3-Small features adapted to 1-channel input.

    MobileNetV3-Small downsamples by 32x, so a 128x128 input yields 4x4 features.
    We bilinearly upsample to 8x8 to give the cross-modal Transformer 64 tokens
    per modality (matching the proposal). The upsample is parameter-free.
    """

    def __init__(self, embed_dim: int = 128):
        super().__init__()
        base = mobilenet_v3_small(weights=None)
        # Replace first conv to accept a single channel (medical grayscale)
        base.features[0][0] = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.features = base.features  # (B, 576, 4, 4) for 128x128 input
        self.proj = nn.Conv2d(576, embed_dim, kernel_size=1)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x)
        h = self.proj(h)
        return self.upsample(h)  # (B, embed_dim, 8, 8)


class CrossModalTransformer(nn.Module):
    """2-layer Transformer encoder over concatenated MRI/PET tokens.

    Input feature maps are flattened to 64 spatial tokens per modality, given a
    learned modality embedding, concatenated to 128 tokens, given a learned
    positional embedding, and passed through `num_layers` transformer encoder layers.
    Output is averaged across the two modality halves and reshaped back to a
    spatial feature map for the decoder.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 2,
        ff_dim: int = 256,
        dropout: float = 0.1,
        spatial_tokens_per_modality: int = 64,
    ):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        total_tokens = spatial_tokens_per_modality * 2
        self.pos_embed = nn.Parameter(torch.zeros(1, total_tokens, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.mod_embed = nn.Embedding(2, embed_dim)  # 0 = MRI, 1 = PET
        self.spatial_tokens = spatial_tokens_per_modality

    def forward(self, mri_feat: torch.Tensor, pet_feat: torch.Tensor) -> torch.Tensor:
        # mri_feat, pet_feat: (B, C, H, W) with H*W = self.spatial_tokens
        B, C, H, W = mri_feat.shape
        mri_tokens = mri_feat.flatten(2).transpose(1, 2)  # (B, HW, C)
        pet_tokens = pet_feat.flatten(2).transpose(1, 2)
        mri_tokens = mri_tokens + self.mod_embed.weight[0]
        pet_tokens = pet_tokens + self.mod_embed.weight[1]
        tokens = torch.cat([mri_tokens, pet_tokens], dim=1) + self.pos_embed
        out = self.transformer(tokens)
        mri_out = out[:, : self.spatial_tokens].transpose(1, 2).reshape(B, C, H, W)
        pet_out = out[:, self.spatial_tokens :].transpose(1, 2).reshape(B, C, H, W)
        return 0.5 * (mri_out + pet_out)


class FusionDecoder(nn.Module):
    """Progressive bilinear upsample + conv decoder: 8x8 -> 16 -> 32 -> 64 -> 128."""

    def __init__(self, embed_dim: int = 128, base_ch: int = 256):
        super().__init__()
        c0, c1, c2, c3 = base_ch, base_ch, base_ch // 2, base_ch // 4
        self.up1 = self._block(embed_dim, c0)
        self.up2 = self._block(c0, c1)
        self.up3 = self._block(c1, c2)
        self.up4 = self._block(c2, c3)
        self.final = nn.Conv2d(c3, 1, kernel_size=1)

    @staticmethod
    def _block(in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.up1(x)
        h = self.up2(h)
        h = self.up3(h)
        h = self.up4(h)
        return torch.sigmoid(self.final(h))


class FedSafeFuse(nn.Module):
    """End-to-end FedSafe-Fuse model. Inputs: two (B, 1, 128, 128) tensors. Output: (B, 1, 128, 128)."""

    def __init__(
        self,
        embed_dim: int = 128,
        transformer_layers: int = 2,
        transformer_heads: int = 2,
        transformer_ff: int = 256,
        decoder_base_ch: int = 256,
    ):
        super().__init__()
        self.mri_encoder = ModalityEncoder(embed_dim)
        self.pet_encoder = ModalityEncoder(embed_dim)
        self.fusion = CrossModalTransformer(
            embed_dim=embed_dim,
            num_layers=transformer_layers,
            num_heads=transformer_heads,
            ff_dim=transformer_ff,
        )
        self.decoder = FusionDecoder(embed_dim=embed_dim, base_ch=decoder_base_ch)

    def forward(self, mri: torch.Tensor, pet: torch.Tensor) -> torch.Tensor:
        mri_feat = self.mri_encoder(mri)
        pet_feat = self.pet_encoder(pet)
        fused_feat = self.fusion(mri_feat, pet_feat)
        return self.decoder(fused_feat)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    m = FedSafeFuse()
    n = count_parameters(m)
    print(f"FedSafe-Fuse trainable params: {n:,} ({n / 1e6:.2f}M)")
    x = torch.randn(2, 1, 128, 128)
    y = m(x, x)
    print(f"Input: {x.shape} -> Output: {y.shape}")
