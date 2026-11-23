import json
from pathlib import Path

import numpy as np
import pytest
import torch
from concord.bands import BandArray
from concord.instruments import detection
from concord.instruments.detection import (
    Detector,
    Padding,
    _crop,
    _reflection_pad,
    reflection_padding,
)
from torch import nn


class AlwaysContrail(nn.Module):
    def forward(self, inputs):
        shape = (inputs.shape[0], 2, inputs.shape[2], inputs.shape[3])
        logits = torch.zeros(shape, device=inputs.device)
        logits[:, 1] = 1.0
        return logits


class TinyNet(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.weight = nn.Parameter(torch.tensor([1.0]))

    def forward(self, inputs):
        return inputs


def test_reflection_padding_expands_to_stride_with_near_symmetric_sides():
    assert reflection_padding((33, 34)) == Padding(left=15, right=15, top=15, bottom=16)
    assert reflection_padding((32, 64)) == Padding(0, 0, 0, 0)


@pytest.mark.parametrize(
    ("shape", "stride", "message"),
    [
        ((3, 3), 1, "stride must be at least 2"),
        ((1, 3), 32, "dimensions of at least 2"),
    ],
)
def test_reflection_padding_rejects_invalid_arguments(shape, stride, message):
    with pytest.raises(ValueError, match=message):
        reflection_padding(shape, stride)


def test_reflection_padding_handles_padding_larger_than_the_input():
    tensor = torch.arange(6, dtype=torch.float32).reshape(1, 1, 2, 3)
    padding = Padding(left=5, right=4, top=3, bottom=6)

    padded = _reflection_pad(tensor, padding)

    assert padded.shape == (1, 1, 11, 12)
    torch.testing.assert_close(_crop(padded, padding), tensor)


def test_crop_handles_zero_padding():
    tensor = torch.arange(12).reshape(1, 1, 3, 4)

    torch.testing.assert_close(_crop(tensor, Padding(0, 0, 0, 0)), tensor)


def test_normalized_inputs_support_bands_and_band_differences():
    scene = BandArray()
    scene["C11"] = np.array([[1.0, np.nan], [5.0, 7.0]])
    scene["C13"] = np.array([[2.0, 4.0], [6.0, 8.0]])
    detector = Detector(
        model=AlwaysContrail(),
        inputs=["C11", "C13-C11"],
        means=np.array([1.0, 1.0], dtype=np.float32),
        stds=np.array([2.0, 1.0], dtype=np.float32),
        threshold=0.5,
        device=torch.device("cpu"),
    )

    result = detector._normalized_inputs(scene)

    assert result.shape == (1, 2, 2, 2)
    assert result.dtype == np.float32
    np.testing.assert_allclose(result[0, 0], [[0.0, 0.0], [2.0, 3.0]])
    np.testing.assert_allclose(result[0, 1], [[0.0, 0.0], [0.0, 0.0]])


def test_detector_thresholds_probabilities_and_clips_to_valid_data():
    scene = BandArray()
    values = np.ones((5, 5), dtype=np.float32)
    values[2, 2] = np.nan
    scene["C13"] = values
    detector = Detector(
        model=AlwaysContrail(),
        inputs=["C13"],
        means=np.array([0.0], dtype=np.float32),
        stds=np.array([1.0], dtype=np.float32),
        threshold=0.7,
        device=torch.device("cpu"),
    )

    result = detector(scene)

    np.testing.assert_array_equal(result, scene.valid_mask)
    assert detector.model.training is False


def write_model_directory(path: Path, model: TinyNet):
    (path / "config.json").write_text(
        json.dumps(
            {
                "architecture": "TinyNet",
                "encoder": "test-encoder",
                "n_channels": 2,
                "encoder_depth": 3,
                "decoder_channels": [8, 4, 2],
            }
        )
    )
    np.save(path / "means.npy", np.array([1.0, 2.0], dtype=np.float64))
    np.save(path / "stds.npy", np.array([3.0, 4.0], dtype=np.float64))
    (path / "threshold.dat").write_text("0.625\n")
    (path / "inputs.txt").write_text("C11\nC13-C15\n")
    torch.save(model.state_dict(), path / "checkpoint.pt")


def test_load_builds_model_and_reads_metadata(tmp_path: Path, monkeypatch):
    saved_model = TinyNet()
    write_model_directory(tmp_path, saved_model)
    monkeypatch.setattr(detection.smp, "TinyNet", TinyNet, raising=False)

    detector = Detector.load(tmp_path)

    assert isinstance(detector.model, TinyNet)
    assert detector.model.kwargs == {
        "encoder_name": "test-encoder",
        "encoder_weights": None,
        "in_channels": 2,
        "classes": 2,
        "encoder_depth": 3,
        "decoder_channels": [8, 4, 2],
    }
    assert detector.inputs == ["C11", "C13-C15"]
    assert detector.means.dtype == np.float32
    assert detector.stds.dtype == np.float32
    np.testing.assert_array_equal(detector.means, [1.0, 2.0])
    np.testing.assert_array_equal(detector.stds, [3.0, 4.0])
    assert detector.threshold == 0.625
    assert detector.device == torch.device("cpu")
    assert detector.model.training is False
    torch.testing.assert_close(detector.model.weight, saved_model.weight)


def test_load_reports_invalid_model_configuration(tmp_path: Path):
    (tmp_path / "config.json").write_text(json.dumps({"architecture": "Missing"}))

    with pytest.raises(ValueError, match="invalid detection model configuration"):
        Detector.load(tmp_path)
