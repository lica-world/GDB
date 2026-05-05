"""Tests for HuggingFace dataset loading helpers."""

from __future__ import annotations

import sys
import types

import pytest

from gdb.hf import load_from_hub


class _FakeImage:
    def save(self, dest, format="PNG"):  # noqa: A002 - matches PIL API name
        with open(dest, "wb") as f:
            f.write(b"fake-png")


def _install_fake_datasets(monkeypatch, load_dataset):
    module = types.ModuleType("datasets")
    module.load_dataset = load_dataset
    monkeypatch.setitem(sys.modules, "datasets", module)


def test_load_from_hub_materializes_image6_auxiliary_assets(monkeypatch, tmp_path):
    def fake_load_dataset(repo_id, benchmark_id, split, streaming):
        assert repo_id == "lica-world/GDB"
        assert benchmark_id == "image-6"
        assert split == "train"
        assert streaming is True
        return iter([
            {
                "sample_id": "sample-1",
                "ground_truth": '{"forbidden_texts": ["HELLO"]}',
                "prompt": "remove text",
                "metadata": '{"mask": "relative/mask.png"}',
                "image": _FakeImage(),
                "input_image_asset": _FakeImage(),
                "mask_asset": _FakeImage(),
                "ground_truth_image_asset": _FakeImage(),
            }
        ])

    _install_fake_datasets(monkeypatch, fake_load_dataset)

    sample = load_from_hub("image-6", n=1, cache_dir=tmp_path)[0]

    assert sample["input_image"].endswith("/image-6/input/sample-1.png")
    assert sample["mask"].endswith("/image-6/mask/sample-1.png")
    assert sample["ground_truth"]["image"].endswith("/image-6/ground_truth/sample-1.png")
    assert sample["ground_truth"]["mask"] == sample["mask"]
    assert sample["ground_truth"]["forbidden_texts"] == ["HELLO"]


def test_load_from_hub_image6_missing_config_error_is_actionable(monkeypatch):
    def fake_load_dataset(repo_id, benchmark_id, split, streaming):
        raise ValueError("BuilderConfig 'image-6' not found")

    _install_fake_datasets(monkeypatch, fake_load_dataset)

    with pytest.raises(ValueError, match="does not currently expose an image-6 configuration"):
        load_from_hub("image-6", n=1)
