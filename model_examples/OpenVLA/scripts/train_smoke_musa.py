"""Run a deterministic, tiny OpenVLA-style action-prediction training smoke."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch import nn


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkout", type=Path, default=Path(".tmp/OpenVLA"))
    p.add_argument("--device", default="musa:0", choices=("musa", "musa:0"))
    p.add_argument("--mode", default="fixture", choices=("fixture",))
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--seed", type=int, default=20260814)
    p.add_argument("--output-dir", type=Path, default=Path(".tmp/OpenVLA/runs/smoke"))
    return p.parse_args()


class TinyOpenVLA(nn.Module):
    """Small action-token predictor matching OpenVLA's 7-DoF/256-bin contract."""

    def __init__(self) -> None:
        super().__init__()
        self.image = nn.Sequential(nn.Conv2d(3, 8, 5, stride=4), nn.GELU(), nn.Flatten())
        self.token = nn.Embedding(128, 16)
        self.head = nn.Sequential(nn.Linear(8 * 7 * 7 + 16, 64), nn.GELU(), nn.Linear(64, 7 * 256))

    def forward(self, pixel_values: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        image = self.image(pixel_values)
        text = self.token(input_ids).mean(dim=1)
        return self.head(torch.cat((image, text), dim=-1)).view(-1, 7, 256)


def main() -> None:
    args = _args()
    if args.steps < 1 or args.batch_size < 1:
        raise ValueError("--steps and --batch-size must be positive")
    if not hasattr(torch, "musa") or not torch.musa.is_available():
        raise RuntimeError("MUSA is unavailable; run in a configured MUSA environment")
    checkout = args.checkout.resolve()
    if not (checkout / "vla-scripts/finetune.py").is_file():
        raise RuntimeError(f"OpenVLA checkout is missing: {checkout}/vla-scripts/finetune.py")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("musa:0" if args.device == "musa" else args.device)
    model = TinyOpenVLA()
    model = model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses: list[float] = []
    for step in range(args.steps):
        pixels = torch.randn(args.batch_size, 3, 32, 32, device=device)
        input_ids = torch.randint(0, 128, (args.batch_size, 12), device=device)
        target = torch.randint(0, 256, (args.batch_size, 7), device=device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(pixels, input_ids)
        loss = nn.functional.cross_entropy(logits.reshape(-1, 256), target.reshape(-1))
        loss.backward()
        if not torch.isfinite(loss) or any(
            p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()
        ):
            raise RuntimeError(f"non-finite training state at step {step}: {loss.item()}")
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output_dir / "tiny_openvla_step.pt")
    report = {
        "mode": args.mode,
        "device": str(device),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "losses": losses,
        "loss": losses[-1],
        "finite_gradients": True,
        "optimizer_step": True,
        "checkpoint": str((args.output_dir / "tiny_openvla_step.pt").resolve()),
    }
    (args.output_dir / "result.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
