"""
Multimodal Cyclone Intensity Decay Model
=========================================
Three input branches — one per TCND pillar — fused for regression.

Architecture:
                         ┌─────────────────┐
  Data_1d  (T, F1) ─────▶  LSTM Encoder   ├──▶ h1 (128)  ─┐
                         └─────────────────┘                │
                         ┌─────────────────┐                │
  Data_3d  (T,C,H,W) ───▶  CNN + Temporal ├──▶ h2 (128)  ─┼──▶ Fusion MLP ──▶ [wind_24h, wind_48h]
                         └─────────────────┘                │
                         ┌─────────────────┐                │
  Env-Data (E,) ─────────▶  MLP Encoder   ├──▶ h3 (64)   ─┘
                         └─────────────────┘

Targets: wind speed 24 h and 48 h after landfall (knots, regression).
"""

import torch
import torch.nn as nn


# ── Branch 1: LSTM over Data_1d tabular sequence ──────────────────────────────
class TabularLSTM(nn.Module):
    """Encodes a (T, F1) sequence of IBTrACS features."""

    def __init__(self, input_size: int = 4, hidden_size: int = 128,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.out_size = hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, T, F1)
        _, (h_n, _) = self.lstm(x)
        return h_n[-1]   # (B, hidden_size)


# ── Branch 2: CNN over Data_3d spatial patches ────────────────────────────────
class SpatialCNN(nn.Module):
    """
    Processes a (T, C, H, W) sequence of 20°x20° gridded patches.
    Each timestep is encoded with a shared CNN; temporal context is
    pooled across steps.
    """

    def __init__(self, in_channels: int = 5, out_size: int = 128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),           nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),   # -> (B*T, 64, 4, 4)
            nn.Flatten(),                   # -> (B*T, 1024)
        )
        self.proj = nn.Sequential(
            nn.Linear(1024, out_size), nn.ReLU(),
        )
        self.out_size = out_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, T, C, H, W)
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feat = self.cnn(x)           # (B*T, 1024)
        feat = self.proj(feat)       # (B*T, out_size)
        feat = feat.view(B, T, -1)   # (B, T, out_size)
        return feat.mean(dim=1)      # temporal mean pool -> (B, out_size)


# ── Branch 3: MLP over Env-Data feature vector ────────────────────────────────
class EnvMLP(nn.Module):
    """Encodes a (E,) pre-calculated environmental feature vector."""

    def __init__(self, input_size: int = 32, out_size: int = 64,
                 dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, out_size),   nn.ReLU(),
        )
        self.out_size = out_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)   # (B, out_size)


# ── Fusion head ───────────────────────────────────────────────────────────────
class FusionHead(nn.Module):
    def __init__(self, in_size: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_size, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 64),      nn.ReLU(),
            nn.Linear(64, 2),        # [wind_24h, wind_48h]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)   # (B, 2)


# ── Full multimodal model ─────────────────────────────────────────────────────
class CycloneDecayModel(nn.Module):
    """
    Multimodal model for post-landfall intensity decay prediction.

    Args:
        tabular_features : number of Data_1d features per timestep (default 4)
        cnn_channels     : number of channels in Data_3d patches   (default 5)
        env_features     : size of Env-Data feature vector         (default 32)
        use_3d           : include the CNN branch                  (default True)
        use_env          : include the Env-Data branch             (default True)
    """

    def __init__(self,
                 tabular_features: int = 4,
                 cnn_channels: int = 5,
                 env_features: int = 32,
                 use_3d: bool = True,
                 use_env: bool = True,
                 dropout: float = 0.3):
        super().__init__()
        self.use_3d  = use_3d
        self.use_env = use_env

        self.branch_1d  = TabularLSTM(tabular_features, 128, dropout=dropout)
        fusion_in = self.branch_1d.out_size

        if use_3d:
            self.branch_3d = SpatialCNN(cnn_channels, 128)
            fusion_in += self.branch_3d.out_size

        if use_env:
            self.branch_env = EnvMLP(env_features, 64, dropout=dropout)
            fusion_in += self.branch_env.out_size

        self.head = FusionHead(fusion_in, dropout=dropout)

    def forward(self,
                x_1d: torch.Tensor,
                x_3d: torch.Tensor | None = None,
                x_env: torch.Tensor | None = None) -> torch.Tensor:
        parts = [self.branch_1d(x_1d)]

        if self.use_3d and x_3d is not None:
            parts.append(self.branch_3d(x_3d))

        if self.use_env and x_env is not None:
            parts.append(self.branch_env(x_env))

        fused = torch.cat(parts, dim=-1)
        return self.head(fused)   # (B, 2) → [wind_24h, wind_48h]


# ── Quick sanity check ────────────────────────────────────────────────────────
if __name__ == "__main__":
    B, T = 4, 8
    model = CycloneDecayModel(tabular_features=4, cnn_channels=5, env_features=32)
    print(model)
    print(f"\nTotal params: {sum(p.numel() for p in model.parameters()):,}")

    x1d  = torch.randn(B, T, 4)
    x3d  = torch.randn(B, T, 5, 80, 80)   # 20° / 0.25° = 80 grid cells
    xenv = torch.randn(B, 32)

    out = model(x1d, x3d, xenv)
    print(f"Output shape : {out.shape}")   # (4, 2)
