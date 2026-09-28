"""Run a bounded QCNet training launch on one MUSA card."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkout", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--processed-dir", type=Path, default=None)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=1)
    p.add_argument("--device", choices=("musa", "musa:0"), default="musa:0")
    p.add_argument("--dtype", choices=("float32", "float16"), default="float32")
    p.add_argument("--seed", type=int, default=20260909)
    p.add_argument("--output-dir", type=Path, default=Path(".tmp/qcnet_train"))
    return p.parse_args()


def _model_kwargs() -> dict[str, object]:
    return {
        "dataset": "argoverse_v2",
        "input_dim": 2,
        "hidden_dim": 128,
        "output_dim": 2,
        "output_head": False,
        "num_historical_steps": 50,
        "num_future_steps": 60,
        "num_modes": 6,
        "num_recurrent_steps": 3,
        "num_freq_bands": 64,
        "num_map_layers": 1,
        "num_agent_layers": 2,
        "num_dec_layers": 2,
        "num_heads": 8,
        "head_dim": 16,
        "dropout": 0.1,
        "pl2pl_radius": 150.0,
        "time_span": 10,
        "pl2a_radius": 50.0,
        "a2a_radius": 50.0,
        "num_t2m_steps": 30,
        "pl2m_radius": 150.0,
        "a2m_radius": 150.0,
        "lr": 5e-4,
        "weight_decay": 1e-4,
        "T_max": 64,
        "submission_dir": ".tmp/qcnet_train",
        "submission_file_name": "submission",
    }


def _add_paths(checkout: Path) -> None:
    script = Path(__file__).resolve()
    for parent in script.parents:
        if (parent / "mu_space").is_dir():
            sys.path.insert(0, str(parent))
            break
    sys.path.insert(0, str(checkout))


def _enable_subset_mode(checkout: Path) -> None:
    """Make the disposable upstream checkout honor the number of raw scenarios."""
    target = checkout / "datasets" / "argoverse_v2_dataset.py"
    marker = "self._num_samples = len(self._raw_file_names)"
    text = target.read_text(encoding="utf-8")
    if marker in text:
        return
    needle = "        }[split]\n"
    if needle not in text:
        raise RuntimeError("QCNet dataset source does not match the expected sample-count contract")
    replacement = (
        needle + "        if os.environ.get('MUSPACE_QCNET_SUBSET') == '1':\n"
        "            self._num_samples = len(self._raw_file_names)\n"
    )
    target.write_text(text.replace(needle, replacement, 1), encoding="utf-8")


def main() -> None:
    args = _args()
    if args.batch_size < 1 or args.max_steps < 1:
        raise SystemExit("--batch-size and --max-steps must be >= 1")
    if not hasattr(torch, "musa") or not torch.musa.is_available():
        raise RuntimeError("MUSA is unavailable; run inside a configured MUSA environment")
    checkout = args.checkout.resolve()
    _add_paths(checkout)
    os.environ["MUSPACE_QCNET_SUBSET"] = "1"
    _enable_subset_mode(checkout)
    from datasets import ArgoverseV2Dataset
    from transforms import TargetBuilder
    from torch_geometric.loader import DataLoader
    from predictors import QCNet

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    kwargs = _model_kwargs()
    model = QCNet(**kwargs)
    data_root = args.data_root.resolve()
    train_processed = (args.processed_dir or (data_root / "train" / "processed")).resolve()
    dataset = ArgoverseV2Dataset(
        root=str(data_root),
        split="train",
        processed_dir=str(train_processed),
        transform=TargetBuilder(50, 60),
        num_historical_steps=50,
        num_future_steps=60,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        persistent_workers=False,
    )
    try:
        first_batch = next(iter(loader))
    except StopIteration as exc:
        raise RuntimeError("the selected AV2 train subset is empty") from exc
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(first_batch.cpu(), args.output_dir / "batch.pt")
    torch.save(
        {"state_dict": model.state_dict(), "model_kwargs": kwargs},
        args.output_dir / "initial_state.pt",
    )
    model = model.to(device).train()
    before_update = next(model.parameters()).detach().clone()
    configured = model.configure_optimizers()
    optimizer = configured[0][0] if isinstance(configured, tuple) else configured[0]
    scale = 128.0 if args.dtype == "float16" else 1.0
    losses = []
    for step in range(args.max_steps):
        batch_device = next(iter(loader)).to(device)
        optimizer.zero_grad(set_to_none=True)
        context = (
            torch.autocast(device_type="musa", dtype=torch.float16)
            if args.dtype == "float16"
            else torch.autocast(device_type="musa", enabled=False)
        )
        with context:
            loss = model.training_step(batch_device, step)
        if not torch.isfinite(loss).item():
            raise RuntimeError(f"non-finite loss at step {step}: {loss}")
        (loss * scale).backward()
        for parameter in model.parameters():
            if parameter.grad is not None and scale != 1.0:
                parameter.grad.div_(scale)
        if any(
            parameter.grad is not None and not torch.isfinite(parameter.grad).all().item()
            for parameter in model.parameters()
        ):
            raise RuntimeError(f"non-finite gradient at step {step}")
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    torch.save(
        {"state_dict": model.state_dict(), "model_kwargs": kwargs},
        args.output_dir / "trained_state.pt",
    )
    parameter_update = not torch.equal(before_update.cpu(), next(model.parameters()).detach().cpu())
    report = {
        "mode": "real_subset",
        "device": str(device),
        "dtype": args.dtype,
        "seed": args.seed,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "finite_parameters": all(torch.isfinite(p).all().item() for p in model.parameters()),
        "finite_gradients": True,
        "losses": losses,
        "loss": losses[-1],
        "optimizer_step": parameter_update,
        "batch": str((args.output_dir / "batch.pt").resolve()),
        "initial_state": str((args.output_dir / "initial_state.pt").resolve()),
        "trained_state": str((args.output_dir / "trained_state.pt").resolve()),
    }
    (args.output_dir / "result.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
