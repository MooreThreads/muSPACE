"""Focused checks for model smoke-wrapper argument and scratch behavior."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("relative", "args", "message"),
    [
        (
            "model_examples/FlashOCC/scripts/train_step_musa.py",
            [
                "--checkout",
                "checkout",
                "--config",
                "config",
                "--data-root",
                "data",
                "--max-iters",
                "0",
            ],
            "--max-iters must be >= 1",
        ),
        (
            "model_examples/UniAD/scripts/train_step_musa.py",
            [
                "--checkout",
                "checkout",
                "--config",
                "config",
                "--data-root",
                "data",
                "--max-iters",
                "0",
            ],
            "--max-iters must be >= 1",
        ),
        (
            "model_examples/BEVFusion/scripts/train_fusion_musa.py",
            [
                "--checkout",
                "checkout",
                "--config",
                "config",
                "--data-root",
                "data",
                "--max-iters",
                "0",
            ],
            "--max-iters must be >= 1",
        ),
    ],
)
def test_wrappers_reject_zero_iterations(relative, args, message):
    result = subprocess.run(
        [sys.executable, str(ROOT / relative), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert message in result.stderr


def test_flashocc_rejects_non_musa_before_importing_patcher():
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "model_examples/FlashOCC/scripts/train_step_musa.py"),
            "--checkout",
            "checkout",
            "--config",
            "config",
            "--data-root",
            "data",
            "--device",
            "cpu",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "--device must be a MUSA device" in result.stderr


def test_bevfusion_rejects_non_accelerator_before_importing_model():
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "model_examples/BEVFusion/scripts/train_fusion_musa.py"),
            "--checkout",
            "checkout",
            "--config",
            "config",
            "--data-root",
            "data",
            "--device",
            "cpu",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "--device must be a MUSA or CUDA device" in result.stderr


def test_openvla_rejects_zero_steps_before_requiring_a_checkout():
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "model_examples/OpenVLA/scripts/train_step_musa.py"),
            "--steps",
            "0",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "--steps and --batch-size must be positive" in result.stderr


@pytest.mark.parametrize(
    ("name", "relative", "prefix"),
    [
        ("flashocc_smoke", "model_examples/FlashOCC/scripts/train_step_musa.py", "flashocc-"),
        ("uniad_smoke", "model_examples/UniAD/scripts/train_step_musa.py", "uniad-"),
    ],
)
def test_scratch_directory_uses_explicit_work_dir(tmp_path, name, relative, prefix):
    module = _load(name, relative)
    work_dir = tmp_path / "work"
    scratch = module._scratch_dir(work_dir, prefix)
    assert scratch.parent == work_dir
    assert scratch.name.startswith(prefix)


def test_wrappers_use_configured_optimizer_builder():
    for relative in (
        "model_examples/FlashOCC/scripts/train_step_musa.py",
        "model_examples/UniAD/scripts/train_step_musa.py",
    ):
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert "from mmcv.runner import build_optimizer" in source
        assert "build_optimizer(model, cfg.optimizer)" in source
        assert "torch.optim.AdamW(model.parameters(), lr=cfg.optimizer.lr)" not in source

def test_lingbotva_rejects_missing_checkout():
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "model_examples/LingBotVA/scripts/train_step_musa.py"),
            "--checkout",
            "does-not-exist",
            "--mode",
            "smoke",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "not a LingBot-VA checkout" in result.stderr


def test_lingbotva_requires_dataset_for_train_mode(tmp_path):
    checkout = tmp_path / "LingBotVA"
    (checkout / "wan_va").mkdir(parents=True)
    (checkout / "wan_va" / "train.py").write_text("", encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "model_examples/LingBotVA/scripts/train_step_musa.py"),
            "--checkout",
            str(checkout),
            "--mode",
            "train",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "--dataset-path is required" in result.stderr
