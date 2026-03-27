"""
U-FNO for Cyclone Prediction
=============================
Adapted from Gege Wen et al., "U-FNO — An enhanced Fourier neural operator
based deep learning model for multiphase flow" (2022).
  Repository : https://github.com/gegewen/ufno

Supports two tasks
──────────────────
  landfall  — binary classification: will this storm make landfall?
              n_outputs=1, BCEWithLogitsLoss, output is a logit
  decay     — regression: wind speed at 24h and 48h
              n_outputs=2, LpLoss, output is [wind_24h, wind_48h]

Architecture overview
─────────────────────
Two input modes:

  spatial (default)
    Primary : Data_3d patches  (B, T, C_sp, H, W)
    Optional: tabular features (B, F_tab) injected via FiLM at every block

  tabular-only
    Tabular features (B, F_tab) projected onto a (G × G) pseudo-grid
    and processed by the identical UFNO stack.

Landfall embedding injection
─────────────────────────────
When training the decay model, a frozen landfall model provides a
learned embedding (B, width) via extract_embedding(). This is injected
into the decay model's UFNO stack at blocks 3–5 via dedicated FiLM
layers — separate from the tabular FiLM — so landfall information
modulates the spatial field, not just the input features.

UFNO block (2-D, repeated 6×)
  SpectralConv2d   — Fourier modes over (H, W)
  + W_local        — pointwise Conv2d for local mixing
  + UNet2d         — U-shaped skip-connection branch (at blocks 3–5)
  + FiLM (tab)     — affine conditioning from tabular features
  + FiLM (lf)      — affine conditioning from landfall embedding (blocks 3–5 only)
  + GELU activation

Prediction head
  Global average pool → Linear(width → 128) → GELU → Dropout(0.3)
  → Linear(128 → n_outputs)
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Learnable relative Lp loss (matches Gege Wen's lploss.py) ─────────────────
class LpLoss(nn.Module):
    """Relative Lp-norm loss: ||pred - target||_p / ||target||_p."""

    def __init__(self, p: int = 2, reduction: str = "mean"):
        super().__init__()
        self.p = p
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = torch.norm(pred - target, p=self.p, dim=-1)
        base = torch.norm(target,        p=self.p, dim=-1).clamp(min=1e-8)
        loss = diff / base
        return loss.mean() if self.reduction == "mean" else loss.sum()


# ── 2-D Spectral Convolution (core UFNO operation) ────────────────────────────
class SpectralConv2d(nn.Module):
    """
    2-D Fourier integral operator.

    For each of the four quadrant combinations of (modes1, modes2) frequency
    indices, applies a learnable complex weight tensor, then reconstructs
    via inverse real FFT.  Mirrors the 3-D implementation in ufno.py but
    operates over (H, W) spatial dimensions.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 modes1: int = 12, modes2: int = 12):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1   # Fourier modes along H
        self.modes2 = modes2   # Fourier modes along W

        scale = 1.0 / (in_channels * out_channels)
        shape = (in_channels, out_channels, modes1, modes2)
        # Four weight tensors for the four frequency quadrants
        for i, name in enumerate(("w1", "w2", "w3", "w4")):
            wr = nn.Parameter(scale * torch.randn(*shape))
            wi = nn.Parameter(scale * torch.randn(*shape))
            self.register_parameter(f"{name}_r", wr)
            self.register_parameter(f"{name}_i", wi)

    def _mul(self, inp: torch.Tensor, wr: nn.Parameter,
             wi: nn.Parameter) -> torch.Tensor:
        """Complex multiply: (B, in_ch, m1, m2) × (in_ch, out_ch, m1, m2)."""
        r = torch.einsum("bimn,iomn->bomn", inp.real, wr) \
          - torch.einsum("bimn,iomn->bomn", inp.imag, wi)
        i = torch.einsum("bimn,iomn->bomn", inp.real, wi) \
          + torch.einsum("bimn,iomn->bomn", inp.imag, wr)
        return torch.complex(r, i)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, in_channels, H, W)
        B, C, H, W = x.shape
        x_ft = torch.fft.rfft2(x)   # (B, C, H, W//2+1)

        m1, m2 = self.modes1, self.modes2
        out_ft = torch.zeros(B, self.out_channels, H, W // 2 + 1,
                             dtype=torch.cfloat, device=x.device)

        # Low-H / Low-W
        out_ft[:, :, :m1, :m2] = self._mul(
            x_ft[:, :, :m1, :m2], self.w1_r, self.w1_i)
        # High-H / Low-W
        out_ft[:, :, -m1:, :m2] = self._mul(
            x_ft[:, :, -m1:, :m2], self.w2_r, self.w2_i)
        # Low-H / High-W
        out_ft[:, :, :m1, -m2:] = self._mul(
            x_ft[:, :, :m1, -m2:], self.w3_r, self.w3_i)
        # High-H / High-W
        out_ft[:, :, -m1:, -m2:] = self._mul(
            x_ft[:, :, -m1:, -m2:], self.w4_r, self.w4_i)

        return torch.fft.irfft2(out_ft, s=(H, W))   # (B, out_channels, H, W)


# ── Mini 2-D U-Net (local spatial refinement) ─────────────────────────────────
class UNet2d(nn.Module):
    """
    3-level encoder-decoder with skip connections.
    Downsamples via strided Conv2d, upsamples via ConvTranspose2d.
    Mirrors the U_net module in Gege Wen's ufno.py.
    """

    def __init__(self, channels: int, kernel_size: int = 3,
                 dropout: float = 0.0):
        super().__init__()
        pad = kernel_size // 2

        def _down(c_in, c_out):
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, kernel_size, stride=2, padding=pad),
                nn.BatchNorm2d(c_out),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(dropout),
            )

        def _up(c_in, c_out):
            return nn.ConvTranspose2d(c_in, c_out, kernel_size=4,
                                      stride=2, padding=1)

        C = channels
        self.d1 = _down(C,     C * 2)
        self.d2 = _down(C * 2, C * 4)
        self.d3 = _down(C * 4, C * 8)

        self.u3 = _up(C * 8,       C * 4)
        self.u2 = _up(C * 4 + C * 4, C * 2)
        self.u1 = _up(C * 2 + C * 2, C)
        self.out = nn.Conv2d(C + C, C, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure spatial dims are divisible by 8 (3 stride-2 downs)
        H, W = x.shape[-2], x.shape[-1]
        ph = (8 - H % 8) % 8
        pw = (8 - W % 8) % 8
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph))

        s1 = self.d1(x)
        s2 = self.d2(s1)
        s3 = self.d3(s2)

        u  = self.u3(s3)
        u  = self.u2(torch.cat([u, s2], dim=1))
        u  = self.u1(torch.cat([u, s1], dim=1))
        u  = self.out(torch.cat([u,  x], dim=1))

        # Crop back if we padded
        return u[:, :, :H, :W]


# ── FiLM conditioning (tabular → scale + shift per channel) ───────────────────
class FiLM(nn.Module):
    """
    Feature-wise Linear Modulation.
    Projects tabular feature vector to (scale, shift) for each channel.
    """

    def __init__(self, tab_dim: int, n_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(tab_dim, n_channels * 2),
        )

    def forward(self, x: torch.Tensor,
                tab: Optional[torch.Tensor]) -> torch.Tensor:
        # x   : (B, C, H, W)
        # tab : (B, tab_dim)  or None
        if tab is None:
            return x
        gamma_beta = self.net(tab)             # (B, 2*C)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        gamma = gamma[:, :, None, None] + 1.0  # broadcast over H, W
        beta  = beta[:, :, None, None]
        return gamma * x + beta


# ── Main UFNO stack ────────────────────────────────────────────────────────────
class CycloneUFNOStack(nn.Module):
    """
    6 UFNO blocks operating on a (B, width, H, W) field.
    Blocks 3–5 add a UNet2d branch (as in the original paper).
    """

    def __init__(self, width: int, modes1: int, modes2: int,
                 tab_dim: int, unet_dropout: float = 0.0,
                 lf_embed_dim: int = 0):
        super().__init__()
        self.n_blocks    = 6
        self.lf_embed_dim = lf_embed_dim
        self._lf_blocks  = (3, 4, 5)   # blocks where landfall FiLM is applied

        self.spec  = nn.ModuleList(
            [SpectralConv2d(width, width, modes1, modes2)
             for _ in range(self.n_blocks)])
        self.wlin  = nn.ModuleList(
            [nn.Conv2d(width, width, kernel_size=1)
             for _ in range(self.n_blocks)])
        # UNet branches at blocks 3, 4, 5  (0-indexed)
        self.unets = nn.ModuleDict({
            str(i): UNet2d(width, dropout=unet_dropout)
            for i in (3, 4, 5)
        })
        self.film  = nn.ModuleList(
            [FiLM(tab_dim, width) for _ in range(self.n_blocks)])
        # Separate FiLM layers for landfall embedding — only at blocks 3–5
        if lf_embed_dim > 0:
            self.lf_film = nn.ModuleList(
                [FiLM(lf_embed_dim, width) for _ in self._lf_blocks])
        self.act   = nn.GELU()

    def forward(self, x: torch.Tensor,
                tab: Optional[torch.Tensor] = None,
                lf_embed: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x        : (B, width, H, W)
        # tab      : (B, tab_dim)       or None
        # lf_embed : (B, lf_embed_dim)  or None
        lf_idx = 0
        for i in range(self.n_blocks):
            h = self.spec[i](x) + self.wlin[i](x)
            if str(i) in self.unets:
                h = h + self.unets[str(i)](x)
            h = self.film[i](h, tab)
            if self.lf_embed_dim > 0 and lf_embed is not None \
                    and i in self._lf_blocks:
                h = self.lf_film[lf_idx](h, lf_embed)
                lf_idx += 1
            x = self.act(h)
        return x   # (B, width, H, W)


# ── Full CycloneUFNO model ─────────────────────────────────────────────────────
class CycloneUFNO(nn.Module):
    """
    U-FNO for cyclone post-landfall intensity decay prediction.

    Parameters
    ----------
    sp_channels   : C_sp  — number of Data_3d channels (default 5)
    T             : number of lookback timesteps         (default 8 = 48 h)
    tab_features  : F_tab — tabular feature dimension   (default 28)
    modes1/2      : Fourier modes along H / W           (default 12)
    width         : hidden channel width in UFNO stack  (default 32)
    grid_size     : G for tabular-only pseudo-grid       (default 16)
    pad_size      : replication padding added to edges   (default 4)
    unet_dropout  : dropout inside UNet2d branches       (default 0.0)
    n_outputs     : 1 for landfall classification, 2 for decay regression
    lf_embed_dim  : dim of landfall embedding for decay model (0 = disabled)
    spatial_sigma : std of Gaussian centre mask in pixels (0 = disabled).
                    Applied to the projected spatial field before the UFNO
                    stack.  sigma=10 focuses on the inner ~2.5 deg core.
                    No effect in tabular-only mode.

    Forward inputs
    --------------
    x_3d     : (B, T, C_sp, H, W)  or None — spatial patches
    x_tab    : (B, F_tab)           or None — tabular features
    lf_embed : (B, lf_embed_dim)    or None — landfall embedding from
               a frozen landfall model, injected via FiLM at blocks 3–5

    At least one of x_3d / x_tab must be provided.
    If only x_tab is given, the model runs in tabular-only mode.

    Forward output
    --------------
    landfall task : (B, 1) — logit (apply sigmoid for probability)
    decay task    : (B, 2) — predicted [wind_24h, wind_48h]
    """

    def __init__(self,
                 sp_channels:  int = 4,
                 T:            int = 8,
                 tab_features: int = 28,
                 modes1:       int = 12,
                 modes2:       int = 12,
                 width:        int = 32,
                 grid_size:    int = 16,
                 pad_size:     int = 4,
                 unet_dropout: float = 0.0,
                 n_outputs:    int = 2,
                 lf_embed_dim: int = 0,
                 spatial_sigma: float = 0.0):
        super().__init__()

        self.T           = T
        self.width       = width
        self.pad_size    = pad_size
        self.grid_size   = grid_size
        self.tab_dim     = tab_features
        self.n_outputs   = n_outputs
        self.lf_embed_dim = lf_embed_dim
        self.spatial_sigma = spatial_sigma

        # ── Gaussian centre mask ───────────────────────────────────
        # Precompute a (1, 1, H, W) Gaussian centred on pixel (40, 40),
        # the storm centre in the 81x81 TCND patches.  Registered as a
        # buffer so it moves to the correct device automatically and is
        # saved in checkpoints.  sigma=0 disables the mask entirely.
        if spatial_sigma > 0.0:
            cx, cy = 40, 40   # storm centre in 81x81 patch
            ys = torch.arange(81, dtype=torch.float32)
            xs = torch.arange(81, dtype=torch.float32)
            yy, xx = torch.meshgrid(ys, xs, indexing='ij')
            mask = torch.exp(
                -((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * spatial_sigma ** 2)
            )  # (81, 81), peak=1 at centre
            self.register_buffer('_spatial_mask',
                                 mask.unsqueeze(0).unsqueeze(0))  # (1,1,81,81)
        else:
            self._spatial_mask = None

        # ── Input projections ──────────────────────────────────────────
        sp_in = T * sp_channels
        self.input_proj_sp  = nn.Linear(sp_in, width)
        self.input_proj_tab = nn.Linear(tab_features, grid_size * grid_size * width)

        # ── UFNO stack ─────────────────────────────────────────────────
        self.stack = CycloneUFNOStack(width, modes1, modes2,
                                      tab_features, unet_dropout,
                                      lf_embed_dim=lf_embed_dim)

        # ── Prediction head ────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Linear(width, 128),
            nn.GELU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, n_outputs),
        )

    # -- internal helpers --------------------------------------------------
    def _pad(self, x: torch.Tensor) -> torch.Tensor:
        p = self.pad_size
        return F.pad(x, (p, p, p, p), mode="replicate")

    def _unpad(self, x: torch.Tensor) -> torch.Tensor:
        p = self.pad_size
        return x[:, :, p:-p, p:-p]

    # -- forward -----------------------------------------------------------
    def forward(self,
                x_3d:     Optional[torch.Tensor] = None,
                x_tab:    Optional[torch.Tensor] = None,
                lf_embed: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x_3d is None and x_tab is None:
            raise ValueError("At least one of x_3d or x_tab must be provided.")

        # ── Build spatial field ────────────────────────────────────────
        if x_3d is not None:
            B, T, C, H, W = x_3d.shape
            field = x_3d.permute(0, 3, 4, 1, 2).reshape(B, H, W, T * C)
            field = self.input_proj_sp(field)
            field = field.permute(0, 3, 1, 2)   # (B, width, H, W)
            # Apply Gaussian centre mask: upweight storm core, downweight
            # periphery.  Mask is (1,1,H,W) — broadcasts over B and width.
            # Only applied when x_3d is present; tabular-only is unaffected.
            if self._spatial_mask is not None:
                field = field * self._spatial_mask
        else:
            B = x_tab.shape[0]
            G = self.grid_size
            field = self.input_proj_tab(x_tab)
            field = field.view(B, self.width, G, G)

        # ── Pad → UFNO stack → unpad ───────────────────────────────────
        field = self._pad(field)
        field = self.stack(field, x_tab, lf_embed)
        field = self._unpad(field)

        # ── Global average pool → head ─────────────────────────────────
        pooled = field.mean(dim=(-2, -1))   # (B, width)
        return self.head(pooled).clamp(-10, 10)   # (B, n_outputs); clamp to ±10σ

    def extract_embedding(self,
                          x_tab: Optional[torch.Tensor] = None,
                          x_3d:  Optional[torch.Tensor] = None
                          ) -> torch.Tensor:
        """
        Return the pooled field representation (B, width) before the head.
        Used by the decay model to extract landfall conditioning signal.
        """
        if x_3d is None and x_tab is None:
            raise ValueError("At least one of x_3d or x_tab must be provided.")
        if x_3d is not None:
            B, T, C, H, W = x_3d.shape
            field = x_3d.permute(0, 3, 4, 1, 2).reshape(B, H, W, T * C)
            field = self.input_proj_sp(field).permute(0, 3, 1, 2)
            if self._spatial_mask is not None:
                field = field * self._spatial_mask
        else:
            B = x_tab.shape[0]
            G = self.grid_size
            field = self.input_proj_tab(x_tab).view(B, self.width, G, G)
        field = self._pad(field)
        field = self.stack(field, x_tab)
        field = self._unpad(field)
        return field.mean(dim=(-2, -1))     # (B, width)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Quick sanity check ────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, T, C, H, W = 4, 8, 5, 80, 80

    model = CycloneUFNO(sp_channels=C, T=T, tab_features=28).to(device)
    print(model)
    print(f"\nTotal trainable params: {model.count_params():,}")

    # Spatial + tabular
    x3d  = torch.randn(B, T, C, H, W, device=device)
    xtab = torch.randn(B, 28, device=device)
    out  = model(x3d, xtab)
    print(f"[spatial+tab]   output: {out.shape}")   # (4, 2)

    # Tabular only
    out2 = model(x_tab=xtab)
    print(f"[tabular-only]  output: {out2.shape}")  # (4, 2)