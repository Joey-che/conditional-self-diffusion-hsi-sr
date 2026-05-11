from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def apply_output_activation(x: torch.Tensor, activation: str) -> torch.Tensor:
    if activation == "sigmoid":
        return torch.sigmoid(x)
    if activation == "clamp":
        return torch.clamp(x, 0.0, 1.0)
    if activation == "identity":
        return x
    raise ValueError(f"Unsupported output activation: {activation}")


class ConvBlock(nn.Module):
    """Two 3x3 convolutions with batch normalization and ReLU."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, use_down: bool) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        if use_down:
            layers.append(nn.MaxPool2d(2))
        layers.append(ConvBlock(in_channels, out_channels))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        use_down: bool,
        align_corners: bool = True,
    ) -> None:
        super().__init__()
        self.use_down = use_down
        self.align_corners = align_corners
        self.conv = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if self.use_down:
            x = F.interpolate(
                x,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=self.align_corners,
            )
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class ConditionalUNet(nn.Module):
    """Resolution-preserving conditional U-Net used by the HSI self-diffusion demo.

    The default setting (`depth=2`, `use_down=False`) matches the paper
    description: encoder/decoder blocks keep the spatial grid unchanged while
    widening the feature dimension and using skip concatenations.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        guide_channels: int,
        width: int = 128,
        depth: int = 2,
        use_down: bool = False,
        output_activation: str = "sigmoid",
        align_corners: bool = True,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be at least 1.")

        self.guide_channels = guide_channels
        self.output_activation = output_activation

        self.input = ConvBlock(in_channels + guide_channels, width)

        self.downs = nn.ModuleList()
        skip_channels: list[int] = []
        channels = width
        for _ in range(depth):
            skip_channels.append(channels)
            next_channels = channels * 2
            self.downs.append(DownBlock(channels, next_channels, use_down=use_down))
            channels = next_channels

        self.bottleneck = ConvBlock(channels, channels)

        self.ups = nn.ModuleList()
        for skip_ch in reversed(skip_channels):
            self.ups.append(
                UpBlock(
                    in_channels=channels,
                    skip_channels=skip_ch,
                    out_channels=skip_ch,
                    use_down=use_down,
                    align_corners=align_corners,
                )
            )
            channels = skip_ch

        self.output = nn.Conv2d(channels, out_channels, kernel_size=1)

    def forward(
        self,
        x: torch.Tensor,
        guide: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.guide_channels > 0:
            if guide is None:
                raise ValueError("A guidance image is required when guide_channels > 0.")
            x = torch.cat([x, guide], dim=1)

        x = self.input(x)
        skips: list[torch.Tensor] = []
        for down in self.downs:
            skips.append(x)
            x = down(x)

        x = self.bottleneck(x)

        for up in self.ups:
            x = up(x, skips.pop())

        return apply_output_activation(self.output(x), self.output_activation)
