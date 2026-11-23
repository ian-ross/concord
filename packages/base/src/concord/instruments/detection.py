"""Verified, deterministic full-scene detection-model inference."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.abc import Traversable
import json
from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from timelines.bands import BandArray


@dataclass(frozen=True, slots=True)
class Padding:
    """Reflection padding amounts in PyTorch's left, right, top, bottom order."""

    left: int
    right: int
    top: int
    bottom: int


@dataclass(slots=True)
class Detector:
    model: nn.Module
    inputs: list[str]
    means: np.ndarray
    stds: np.ndarray
    threshold: float
    device: torch.device

    @classmethod
    def load(
            cls, model_dir: Path | Traversable, *,
            device: str | torch.device = 'cpu'
    ) -> Detector:
        target = torch.device(device)

        def p(filename: str) -> Path | Traversable:
            return model_dir / filename

        config = json.loads(p('config.json').read_text())
        try:
            architecture = getattr(smp, config['architecture'])
            model = architecture(
                encoder_name=config['encoder'],
                encoder_weights=None,
                in_channels=config['n_channels'],
                classes=2,
                encoder_depth=config['encoder_depth'],
                decoder_channels=config['decoder_channels'],
            )
        except (AttributeError, KeyError, TypeError) as error:
            raise ValueError('invalid detection model configuration') from error

        with p('means.npy').open('rb') as fp:
            means = np.asarray(np.load(fp), dtype=np.float32)
        with p('stds.npy').open('rb') as fp:
            stds = np.asarray(np.load(fp), dtype=np.float32)
        threshold = float(p('threshold.dat').read_text().strip())

        with p('checkpoint.pt').open('rb') as checkpoint:
            state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        model.load_state_dict(state, strict=True)
        model.to(target).eval()

        with p('inputs.txt').open('r') as fp:
            inputs = [line.strip() for line in fp.readlines()]

        return cls(model, inputs, means, stds, threshold, target)

    def __call__(self, scene: BandArray) -> np.ndarray:
        """Classify one complete grid in FP32, clipping output to valid data."""
        inputs = self._normalized_inputs(scene)
        tensor = torch.from_numpy(inputs).to(self.device, dtype=torch.float32)
        padding = reflection_padding((tensor.shape[-2], tensor.shape[-1]))
        padded = _reflection_pad(tensor, padding)
        self.model.eval()
        with torch.inference_mode():
            logits = self.model(padded.float())
            probabilities = torch.softmax(logits.float(), dim=1)[:, 1]
        cropped = _crop(probabilities, padding)
        result = cropped[0].cpu().numpy() >= self.threshold
        return result & scene.valid_mask

    def _normalized_inputs(self, scene: BandArray) -> np.ndarray:
        """Build the FP32 tensor expected by the model."""
        channels = []
        for inexpr in self.inputs:
            hyphen_index = inexpr.find('-')
            if hyphen_index == -1:
                channels.append(scene[inexpr])
            else:
                left = inexpr[:hyphen_index]
                right = inexpr[hyphen_index + 1:]
                channels.append(scene[left] - scene[right])
        values = np.stack(channels, dtype=np.float32)
        values = ((values - self.means[:, None, None]) / self.stds[:, None, None])[None]
        values[np.isnan(values)] = 0.0
        return values


def reflection_padding(shape: tuple[int, int], stride: int = 32) -> Padding:
    """Return near-symmetric reflection padding that expands a field to *stride*."""
    if stride < 2:
        raise ValueError('stride must be at least 2')
    height, width = shape
    if height < 2 or width < 2:
        raise ValueError('reflection padding requires dimensions of at least 2')
    vertical, horizontal = (-height) % stride, (-width) % stride
    return Padding(
        horizontal // 2,
        horizontal - horizontal // 2,
        vertical // 2,
        vertical - vertical // 2
    )


def _reflection_pad(tensor: Tensor, padding: Padding) -> Tensor:
    """Apply reflection padding even to small test fields requiring >1 reflection."""
    left, right, top, bottom = padding.left, padding.right, padding.top, padding.bottom
    while left or right or top or bottom:
        height, width = tensor.shape[-2:]
        step_left, step_right = min(left, width - 1), min(right, width - 1)
        step_top, step_bottom = min(top, height - 1), min(bottom, height - 1)
        tensor = F.pad(tensor, (step_left, step_right, step_top, step_bottom), mode='reflect')
        left, right = left - step_left, right - step_right
        top, bottom = top - step_top, bottom - step_bottom
    return tensor


def _crop(output: Tensor, padding: Padding) -> Tensor:
    bottom = -padding.bottom if padding.bottom else None
    right = -padding.right if padding.right else None
    return output[..., padding.top:bottom, padding.left:right]
