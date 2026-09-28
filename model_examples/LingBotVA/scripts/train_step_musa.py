#!/usr/bin/env python3
# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""LingBot-VA MUSA validation driver.

Two modes share one entry point:

``--mode smoke``
    Dataset-free check of the MUSA runtime and of the FlexAttention BlockMask
    path the recipe patches.  It reports the device layer, compiles
    ``create_block_mask(dynamic=True)`` at realistic LIBERO sequence lengths
    (the compile path that asserts on MUSA without the recipe patch), runs a
    compiled FlexAttention forward/backward, and takes one optimizer step on a
    small ``WanAttention`` module.

``--mode train``
    Real-data training launch through the patched upstream trainer
    (``python -m torch.distributed.run -m train --config-name ...``).  It needs
    a prepared LeRobot-Latent dataset and the ``lingbot-va-base`` weights.  The
    driver parses the trainer log and emits the same JSON contract as the smoke
    mode.

Both modes print a JSON object on the last stdout line and write it to
``--output-dir``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_GATE_LENGTHS = (3968, 4096, 4224, 5504)
BLOCK_SIZE = 128
PERF_RE = re.compile(r"PERF step=(\d+) step_time=([0-9.]+)")
LOSS_RE = re.compile(r"latent_loss=([0-9.]+),\s*action_loss=([0-9.]+),\s*step=(\d+)")
BAD_RE = re.compile(r"sort\.stable|InductorError|AssertionError|FloatingPointError|Nan|NaN|nan|OutOfMemory|out of memory")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkout", required=True, help="Pinned LingBot-VA checkout (after `mu-space migrate`)")
    parser.add_argument("--mode", choices=("smoke", "train"), default="smoke")
    parser.add_argument("--device", default="musa:0")
    parser.add_argument("--attn-mode", default="flex", choices=("flex", "torch", "flashattn"))
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--output-dir", default=".tmp/lingbotva")
    parser.add_argument("--length-gate", default=",".join(str(v) for v in DEFAULT_GATE_LENGTHS))
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--pretrained-model-path", default=None)
    parser.add_argument("--ngpu", type=int, default=1)
    parser.add_argument("--num-steps", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=10)
    parser.add_argument("--dataset-init-worker", type=int, default=1)
    parser.add_argument("--load-worker", type=int, default=16)
    parser.add_argument("--save-root", default=None)
    parser.add_argument("--save-interval", type=int, default=200)
    parser.add_argument("--ifname", default=None, help="MCCL/NCCL socket interface for multi-card runs")
    parser.add_argument("--master-port", type=int, default=0)
    return parser


def device_type(device: str) -> str:
    return device.split(":", 1)[0]


def backend_env(attn_mode: str) -> dict:
    """Return the recipe's validated MUSA environment for an attention path.

    The fused RoPE path and the FSDP tuning are independent of each other, but
    only the FSDP one is memory-bound. FlexAttention keeps the BlockMask and the
    unfused attention intermediates resident, so the memory-for-collectives
    trades do not fit there: dropping `reshard_after_forward` OOMs at step 1 and
    block-level sharding OOMs at step 3, both at one sample per rank with
    gradient accumulation 10. The flex path therefore keeps the upstream FSDP
    layout and only takes the fused RoPE.
    """
    env = {
        "LINGBOT_ROPE_MODE": "kernel",
        "LINGBOT_FSDP_SHARD_GRANULARITY": "block",
        "LINGBOT_FSDP_RESHARD_AFTER_FORWARD": "0",
    }
    if attn_mode == "flex":
        env.update(
            {
                "LINGBOT_FLEX_SORT_CONTIGUOUS": "1",
                "LINGBOT_FLEX_DYNAMIC": "1",
                "LINGBOT_FSDP_SHARD_GRANULARITY": "unit",
                "LINGBOT_FSDP_RESHARD_AFTER_FORWARD": "1",
            }
        )
    return env


def apply_env(checkout: Path, args: argparse.Namespace) -> None:
    """Set the recipe's validated MUSA environment before importing the model."""
    os.environ.setdefault("LINGBOT_DEVICE", device_type(args.device))
    for key, value in backend_env(args.attn_mode).items():
        os.environ.setdefault(key, value)
    if args.ifname:
        os.environ.setdefault("MCCL_SOCKET_IFNAME", args.ifname)
        os.environ.setdefault("NCCL_SOCKET_IFNAME", args.ifname)
        os.environ.setdefault("MCCL_IB_DISABLE", "1")
        os.environ.setdefault("NCCL_IB_DISABLE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    sys.path.insert(0, str(checkout / "wan_va"))


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def run_smoke(checkout: Path, args: argparse.Namespace) -> dict:
    import torch

    recipe_patches_present = True
    try:
        from utils.musa_bootstrap import bootstrap_musa
    except ImportError:
        recipe_patches_present = False
    else:
        bootstrap_musa()

    try:
        from utils.device import (
            default_attn_mode,
            get_device_type,
            get_dist_backend,
            get_torch_device,
            is_torchada_active,
            supports_flex_attention,
            use_cuda_api,
        )
    except ImportError:
        # Unpatched checkout: the recipe device layer is absent. Fall back to a
        # minimal mapping so the BlockMask gate can still demonstrate the raw
        # (unpatched) MUSA behaviour this recipe exists for.
        recipe_patches_present = False
        if device_type(args.device) == "musa":
            try:
                import torch_musa  # noqa: F401
            except ImportError:
                pass

        def default_attn_mode(preferred: str = "torch") -> str:
            return preferred

        def get_device_type() -> str:
            musa = getattr(torch, "musa", None)
            return "musa" if musa is not None and musa.is_available() else "cpu"

        def get_torch_device(rank: int = 0) -> torch.device:
            return torch.device(f"musa:{rank}") if get_device_type() == "musa" else torch.device("cpu")

        def get_dist_backend() -> str:
            return "nccl" if get_device_type() == "musa" else "gloo"

        def is_torchada_active() -> bool:
            return "torchada" in sys.modules

        def supports_flex_attention() -> bool:
            return True

        def use_cuda_api() -> bool:
            return is_torchada_active()

    if not recipe_patches_present:
        # diffusers imports flash_attn eagerly when the Wan VAE module is loaded;
        # recipe patch 01 disables that probe on MUSA. Apply the same probe
        # shutdown on an unpatched checkout so the BlockMask gate below can still
        # run and show the raw (unpatched) behaviour.
        try:
            import diffusers.utils.import_utils as _import_utils

            _import_utils._flash_attn_available = False
            _import_utils._flash_attn_3_available = False
        except ImportError:
            pass

    from modules.model import WanAttention

    torch.manual_seed(args.seed)
    target = get_torch_device(0)
    versions = {"torch": torch.__version__}
    try:
        import torch_musa

        versions["torch_musa"] = torch_musa.__version__
    except ImportError:
        versions["torch_musa"] = None
    try:
        import torchada

        versions["torchada"] = getattr(torchada, "__version__", "unknown")
    except ImportError:
        versions["torchada"] = None

    lengths = [int(value) for value in args.length_gate.split(",") if value]
    gate: dict[str, bool] = {}
    gate_errors: dict[str, str] = {}
    gates_ok = True
    if supports_flex_attention():
        from torch.nn.attention.flex_attention import create_block_mask

        def causal_mask(batch, head, q_idx, kv_idx):
            return q_idx >= kv_idx

        compiled_mask = torch.compile(create_block_mask, dynamic=True)
        for length in lengths:
            try:
                compiled_mask(
                    causal_mask,
                    1,
                    1,
                    length,
                    length,
                    device=target,
                    BLOCK_SIZE=BLOCK_SIZE,
                    _compile=False,
                )
                gate[str(length)] = True
            except Exception as exc:  # noqa: BLE001 - every compile failure is reported
                gate[str(length)] = False
                gate_errors[str(length)] = f"{type(exc).__name__}: {str(exc)[:160]}"
                gates_ok = False
    else:
        gates_ok = False
        gate_errors["dataset_free_gate"] = "flex attention unavailable on this stack"

    flex_ok = False
    flex_error = None
    if supports_flex_attention():
        from torch.nn.attention.flex_attention import flex_attention

        try:
            compiled_flex = torch.compile(flex_attention, dynamic=True)
            query = torch.randn(1, 4, 16, 64, device=target, dtype=torch.bfloat16, requires_grad=True)
            out = compiled_flex(query, query, query)
            out.float().sum().backward()
            flex_ok = bool(torch.isfinite(out.float()).all() and torch.isfinite(query.grad).all())
        except Exception as exc:  # noqa: BLE001
            flex_error = f"{type(exc).__name__}: {str(exc)[:160]}"

    attn_ok = False
    attn_loss = None
    attn_error = None
    try:
        attn = WanAttention(dim=128, heads=8, dim_head=16, attn_mode=default_attn_mode("torch"))
        attn = attn.to(device=target, dtype=torch.bfloat16)
        hidden = torch.randn(1, 16, 128, device=target, dtype=torch.bfloat16)
        optimizer = torch.optim.AdamW(attn.parameters(), lr=1e-4)
        out = attn(hidden, hidden, hidden, rotary_emb=None)
        loss = out.float().pow(2).mean()
        loss.backward()
        grads_finite = all(
            parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in attn.parameters()
        )
        optimizer.step()
        attn_ok = bool(torch.isfinite(loss) and grads_finite)
        attn_loss = float(loss.detach().float().cpu())
    except Exception as exc:  # noqa: BLE001
        attn_error = f"{type(exc).__name__}: {str(exc)[:160]}"

    return {
        "mode": "smoke",
        "recipe_patches_present": recipe_patches_present,
        "device": args.device,
        "attn_mode": args.attn_mode,
        "seed": args.seed,
        "versions": versions,
        "device_type": get_device_type(),
        "device_layer": str(target),
        "torchada_active": is_torchada_active(),
        "use_cuda_api": use_cuda_api(),
        "dist_backend": get_dist_backend(),
        "supports_flex_attention": supports_flex_attention(),
        "block_mask_gate_lengths": lengths,
        "block_mask_gate": gate,
        "block_mask_gate_errors": gate_errors,
        "block_mask_gate_all_pass": gates_ok,
        "flex_forward_backward_finite": flex_ok,
        "flex_forward_backward_error": flex_error,
        "attention_module_forward_backward_finite": attn_ok,
        "attention_module_error": attn_error,
        "optimizer_step": attn_ok,
        "finite_gradients": attn_ok,
        "losses": [attn_loss] if attn_loss is not None else [],
    }


def parse_train_log(log_path: Path) -> dict:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    step_times: dict[int, float] = {}
    losses: dict[int, tuple[float, float]] = {}
    for line in text.splitlines():
        perf = PERF_RE.search(line)
        if perf:
            step_times[int(perf.group(1))] = float(perf.group(2))
        loss = LOSS_RE.search(line)
        if loss:
            losses[int(loss.group(3))] = (float(loss.group(1)), float(loss.group(2)))
    steps = [
        {
            "step": step,
            "latent_loss": losses.get(step, (None, None))[0],
            "action_loss": losses.get(step, (None, None))[1],
            "step_time": step_times.get(step),
        }
        for step in sorted(set(step_times) | set(losses))
    ]
    bad = sorted({match.group(0) for match in BAD_RE.finditer(text)})
    return {
        "steps": steps,
        "step_times": [entry["step_time"] for entry in steps if entry["step_time"] is not None],
        "losses": [
            entry["latent_loss"] + entry["action_loss"]
            for entry in steps
            if entry["latent_loss"] is not None and entry["action_loss"] is not None
        ],
        "training_completed": "Training completed!" in text,
        "failure_markers": bad,
    }


def run_train(checkout: Path, args: argparse.Namespace, output_dir: Path) -> dict:
    save_root = Path(args.save_root) if args.save_root else output_dir / "train_out"
    save_root.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train.log"
    port = args.master_port or free_port()
    device = device_type(args.device)
    command = [
        sys.executable, "-m", "torch.distributed.run",
        "--nnodes=1", f"--nproc_per_node={args.ngpu}", "--node_rank=0",
        "--master_addr=127.0.0.1", f"--master_port={port}",
        "--local-ranks-filter=0", "--tee", "3",
        "-m", "train", "--config-name", "libero_train",
        "--dataset-path", args.dataset_path,
        "--pretrained-model-path", args.pretrained_model_path,
        "--num-steps", str(args.num_steps),
        "--batch-size", str(args.batch_size),
        "--gradient-accumulation-steps", str(args.gradient_accumulation_steps),
        "--dataset-init-worker", str(args.dataset_init_worker),
        "--load-worker", str(args.load_worker),
        "--save-interval", str(args.save_interval),
        "--cfg-prob", "0.1",
        "--attn-mode", args.attn_mode,
        "--disable-wandb",
        "--save-root", str(save_root),
    ]
    env = os.environ.copy()
    env.setdefault("LINGBOT_DEVICE", device)
    env["LINGBOT_SEED"] = str(args.seed)
    for key, value in backend_env(args.attn_mode).items():
        env.setdefault(key, value)
    if args.ifname:
        env.setdefault("MCCL_SOCKET_IFNAME", args.ifname)
        env.setdefault("NCCL_SOCKET_IFNAME", args.ifname)
        env.setdefault("MCCL_IB_DISABLE", "1")
        env.setdefault("NCCL_IB_DISABLE", "1")
    env["PYTHONPATH"] = os.pathsep.join([str(checkout / "wan_va"), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)

    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(command, cwd=checkout / "wan_va", env=env, stdout=log, stderr=subprocess.STDOUT)
    elapsed = time.perf_counter() - started

    parsed = parse_train_log(log_path)
    finite_losses = bool(parsed["losses"]) and all(finite(value) for value in parsed["losses"])
    ok = process.returncode == 0 and parsed["training_completed"] and finite_losses and not parsed["failure_markers"]
    payload = {
        "mode": "train",
        "device": args.device,
        "attn_mode": args.attn_mode,
        "ngpu": args.ngpu,
        "num_steps": args.num_steps,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "seed": args.seed,
        "save_interval": args.save_interval,
        "exit_code": process.returncode,
        "wall_time": elapsed,
        "log": str(log_path),
        "steps": parsed["steps"],
        "step_times": parsed["step_times"],
        "losses": parsed["losses"],
        "training_completed": parsed["training_completed"],
        "failure_markers": parsed["failure_markers"],
        "optimizer_step": bool(parsed["steps"]) and process.returncode == 0,
        "finite_gradients": ok,
        "save_root": str(save_root),
    }
    return payload


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    checkout = Path(args.checkout).resolve()
    if not (checkout / "wan_va" / "train.py").is_file():
        parser.error(f"not a LingBot-VA checkout: {checkout}")
    if args.mode == "train":
        if not args.dataset_path:
            parser.error("--dataset-path is required for --mode train")
        if not args.pretrained_model_path:
            parser.error("--pretrained-model-path is required for --mode train")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    apply_env(checkout, args)

    if args.mode == "smoke":
        payload = run_smoke(checkout, args)
    else:
        payload = run_train(checkout, args, output_dir)

    result_path = output_dir / f"{args.mode}.json"
    result_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
