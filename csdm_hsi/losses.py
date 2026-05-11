from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def vgg_normalize(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - mean) / std


class VGGStructureLoss(nn.Module):
    """VGG feature loss for RGB-like guidance images.

    For 6-channel MSI guidance, the channels are split into two RGB-like
    triplets and averaged, matching the benchmark implementation.
    """

    def __init__(
        self,
        layers: tuple[str, ...] = ("conv1_2", "conv2_2"),
        weights: tuple[float, ...] = (1.0, 0.5),
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        try:
            from torchvision.models import VGG16_Weights, vgg16
        except Exception as exc:  # pragma: no cover - depends on optional install
            raise ImportError("VGG guidance requires torchvision.") from exc

        model_weights = VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        self.vgg = vgg16(weights=model_weights).features.eval()
        for param in self.vgg.parameters():
            param.requires_grad = False

        self.layers = layers
        self.weights = weights

    def forward(self, pred: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        if pred.shape != guide.shape:
            raise ValueError(f"Shape mismatch: pred={tuple(pred.shape)}, guide={tuple(guide.shape)}")

        channels = pred.shape[1]
        if channels == 3:
            return 0.25 * self._single_rgb_loss(pred, guide)
        if channels == 4:
            return 0.25 * self._single_rgb_loss(pred[:, 1:4], guide[:, 1:4])
        if channels == 6:
            return 0.5 * (
                self._single_rgb_loss(pred[:, 0:3], guide[:, 0:3])
                + self._single_rgb_loss(pred[:, 3:6], guide[:, 3:6])
            )
        raise ValueError("VGG guidance supports 3-, 4-, or 6-channel guidance only.")

    def _single_rgb_loss(self, pred: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        pred_feats = self._features(vgg_normalize(pred))
        guide_feats = self._features(vgg_normalize(guide))
        loss = pred.new_tensor(0.0)
        for idx, key in enumerate(pred_feats):
            pred_norm = _standardize_feature(pred_feats[key])
            guide_norm = _standardize_feature(guide_feats[key])
            loss = loss + self.weights[idx] * F.mse_loss(pred_norm, guide_norm)
        return loss

    def _features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features: dict[str, torch.Tensor] = {}
        h = x
        for idx, layer in enumerate(self.vgg):
            h = layer(h)
            if idx == 3 and "conv1_2" in self.layers:
                features["conv1_2"] = h
            if idx == 8 and "conv2_2" in self.layers:
                features["conv2_2"] = h
                break
        return features


def _standardize_feature(x: torch.Tensor) -> torch.Tensor:
    mean = x.mean(dim=(2, 3), keepdim=True)
    std = x.std(dim=(2, 3), keepdim=True)
    return (x - mean) / (std + 1e-8)
