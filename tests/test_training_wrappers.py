"""Focused checks for model smoke-wrapper argument and scratch behavior."""

from __future__ import annotations

import importlib.util
import os
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
        (
            "model_examples/SparseDrive/scripts/train_step_musa.py",
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


def test_sparsedrive_rejects_non_musa_before_importing_patcher():
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "model_examples/SparseDrive/scripts/train_step_musa.py"),
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


def test_sparsedrive_rejects_missing_data_root(tmp_path):
    checkout = tmp_path / "SparseDrive"
    (checkout / "projects" / "mmdet3d_plugin").mkdir(parents=True)
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "model_examples/SparseDrive/scripts/train_step_musa.py"),
            "--checkout",
            str(checkout),
            "--data-root",
            str(tmp_path / "absent"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "data root does not exist" in result.stderr


def test_sparsedrive_rejects_missing_checkout():
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "model_examples/SparseDrive/scripts/train_step_musa.py"),
            "--checkout",
            "does-not-exist",
            "--data-root",
            "data",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "not a SparseDrive checkout" in result.stderr


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
        (
            "sparsedrive_smoke",
            "model_examples/SparseDrive/scripts/train_step_musa.py",
            "sparsedrive-",
        ),
        (
            "sparse4d_smoke",
            "model_examples/Sparse4D/scripts/train_step_musa.py",
            "sparse4d-",
        ),
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
        "model_examples/SparseDrive/scripts/train_step_musa.py",
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


def _run_qwendrive(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(ROOT / "model_examples/QwenDrive/scripts/train_step_musa.py"), *args],
        check=False,
        capture_output=True,
        text=True,
    )


def test_qwendrive_rejects_missing_checkout():
    result = _run_qwendrive("--checkout", "does-not-exist")
    assert result.returncode == 2
    assert "not a Qwen-Drive checkout" in result.stderr


def test_qwendrive_rejects_unknown_mode():
    result = _run_qwendrive("--checkout", "does-not-exist", "--mode", "bogus")
    assert result.returncode == 2
    assert "invalid choice" in result.stderr


def test_qwendrive_rejects_zero_device_count(tmp_path):
    checkout = tmp_path / "QwenDrive"
    (checkout / "src" / "qwen_drive_perception").mkdir(parents=True)
    result = _run_qwendrive("--checkout", str(checkout), "--device-count", "0")
    assert result.returncode == 2
    assert "--device-count must be at least 1" in result.stderr


def test_qwendrive_real_mode_requires_weight_dirs(tmp_path):
    checkout = tmp_path / "QwenDrive"
    (checkout / "src" / "qwen_drive_perception").mkdir(parents=True)
    result = _run_qwendrive("--checkout", str(checkout), "--mode", "real")
    assert result.returncode == 2
    assert "--vlm must be a directory" in result.stderr


def test_qwendrive_real_mode_requires_frame_files(tmp_path):
    checkout = tmp_path / "QwenDrive"
    (checkout / "src" / "qwen_drive_perception").mkdir(parents=True)
    vlm = tmp_path / "vlm"
    perception = tmp_path / "perception"
    frame = tmp_path / "frames" / "token"
    for path in (vlm, perception, frame):
        path.mkdir(parents=True)
    (frame / "frame.json").write_text("{}", encoding="utf-8")
    result = _run_qwendrive(
        "--checkout",
        str(checkout),
        "--mode",
        "real",
        "--vlm",
        str(vlm),
        "--perception",
        str(perception),
        "--frames",
        str(frame.parent),
    )
    assert result.returncode == 2
    assert "calib.npz and gt.npz" in result.stderr


def _anchors_checkout(tmp_path: Path, failing: str | None = None) -> Path:
    checkout = tmp_path / "SparseDrive"
    (checkout / "tools/kmeans").mkdir(parents=True)
    (checkout / "data/infos").mkdir(parents=True)
    (checkout / "data/infos/nuscenes_infos_train.pkl").write_bytes(b"infos")
    outputs = {
        "kmeans_det": "kmeans_det_900.npy",
        "kmeans_map": "kmeans_map_100.npy",
        "kmeans_motion": "kmeans_motion_6.npy",
        "kmeans_plan": "kmeans_plan_6.npy",
    }
    for name, produced in outputs.items():
        if name == failing:
            body = "raise SystemExit('clustering failed')\n"
        else:
            body = (
                "from pathlib import Path\n"
                f"target = Path('data/kmeans/{produced}')\n"
                "target.parent.mkdir(parents=True, exist_ok=True)\n"
                "target.write_bytes(b'anchors')\n"
            )
        (checkout / "tools/kmeans" / f"{name}.py").write_text(body, encoding="utf-8")
    return checkout


def _python_on_path(tmp_path: Path) -> dict[str, str]:
    bindir = tmp_path / "bin"
    bindir.mkdir()
    launcher = bindir / "python"
    launcher.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n', encoding="utf-8")
    launcher.chmod(0o755)
    return {**os.environ, "PATH": f"{bindir}{os.pathsep}{os.environ.get('PATH', '')}"}


def test_prepare_anchors_resolves_relative_infos(tmp_path):
    checkout = _anchors_checkout(tmp_path)
    result = subprocess.run(
        [
            "bash",
            str(ROOT / "model_examples/SparseDrive/scripts/prepare_anchors.sh"),
            "--checkout",
            str(checkout),
            "--infos",
            "data/infos/nuscenes_infos_train.pkl",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=_python_on_path(tmp_path),
    )
    assert result.returncode == 0, result.stderr
    assert (checkout / "data/kmeans/kmeans_plan_6.npy").is_file()


def test_prepare_anchors_explains_plan_command_gap(tmp_path):
    checkout = _anchors_checkout(tmp_path, failing="kmeans_plan")
    result = subprocess.run(
        [
            "bash",
            str(ROOT / "model_examples/SparseDrive/scripts/prepare_anchors.sh"),
            "--checkout",
            str(checkout),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=_python_on_path(tmp_path),
    )
    assert result.returncode == 1
    assert "kmeans_plan.py failed while clustering anchors" in result.stderr
    assert "navigation command" in result.stderr
    assert "v1.0-trainval" in result.stderr


def _sparse4d_checkout(tmp_path: Path) -> Path:
    checkout = tmp_path / "Sparse4D"
    (checkout / "tools").mkdir(parents=True)
    (checkout / "data/infos").mkdir(parents=True)
    (checkout / "data/infos/nuscenes_infos_train.pkl").write_bytes(b"infos")
    return checkout


def test_sparse4d_prepare_dataset_passes_converter_flags(tmp_path):
    checkout = _sparse4d_checkout(tmp_path)
    (checkout / "tools/nuscenes_converter.py").write_text(
        "import argparse\n"
        "from pathlib import Path\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.add_argument('--root_path')\n"
        "parser.add_argument('--info_prefix')\n"
        "parser.add_argument('--version')\n"
        "parser.add_argument('--max_sweeps', type=int)\n"
        "args = parser.parse_args()\n"
        "prefix = Path(args.info_prefix)\n"
        "prefix.parent.mkdir(parents=True, exist_ok=True)\n"
        "prefix.with_name(prefix.name + '_infos_train.pkl').write_bytes(b'train')\n"
        "prefix.with_name(prefix.name + '_infos_val.pkl').write_bytes(b'val')\n",
        encoding="utf-8",
    )
    data_root = tmp_path / "nuscenes"
    data_root.mkdir()
    out_dir = tmp_path / "infos"
    result = subprocess.run(
        [
            "bash",
            str(ROOT / "model_examples/Sparse4D/scripts/prepare_dataset.sh"),
            "--checkout",
            str(checkout),
            "--data-root",
            str(data_root),
            "--out-dir",
            str(out_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=_python_on_path(tmp_path),
    )
    assert result.returncode == 0, result.stderr
    assert (out_dir / "nuscenes_infos_train.pkl").is_file()
    assert (out_dir / "nuscenes_infos_val.pkl").is_file()


def test_sparse4d_prepare_anchors_uses_config_output_name(tmp_path):
    checkout = _sparse4d_checkout(tmp_path)
    (checkout / "tools/anchor_generator.py").write_text(
        "import argparse\n"
        "from pathlib import Path\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.add_argument('--ann_file')\n"
        "parser.add_argument('--num_anchor', type=int)\n"
        "parser.add_argument('--output_file_name')\n"
        "args = parser.parse_args()\n"
        "Path(args.output_file_name).write_bytes(b'anchors')\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            "bash",
            str(ROOT / "model_examples/Sparse4D/scripts/prepare_anchors.sh"),
            "--checkout",
            str(checkout),
            "--no-patch",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=_python_on_path(tmp_path),
    )
    assert result.returncode == 0, result.stderr
    assert (checkout / "data/anchors/nuscenes_kmeans900.npy").is_file()
