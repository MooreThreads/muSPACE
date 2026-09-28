"""Report CUDA-specific source usage left in a pinned BEVFormer checkout."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

PATTERNS = {
    "cuda_method": re.compile(r"\.cuda\s*\("),
    "torch_cuda": re.compile(r"\btorch\.cuda\b"),
    "cuda_amp": re.compile(r"\bcuda\.amp\b|from torch\.cuda import amp"),
    "cuda_device": re.compile(r"device\s*=\s*['\"]cuda|torch\.device\([^)]*cuda"),
    "cuda_backend": re.compile(r"torch\.backends\.(?:cuda|cudnn)"),
}


def audit(root: Path) -> list[dict[str, object]]:
    findings: list[dict[str, object]] = []
    for path in sorted(root.rglob("*.py")):
        if ".git" in path.parts:
            continue
        for line_number, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
            matches = sorted(name for name, pattern in PATTERNS.items() if pattern.search(line))
            if matches:
                findings.append(
                    {
                        "file": str(path.relative_to(root)),
                        "line": line_number,
                        "categories": matches,
                        "source": line.strip(),
                    }
                )
    return findings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkout", type=Path)
    args = parser.parse_args()
    findings = audit(args.checkout.resolve())
    print(json.dumps({"count": len(findings), "findings": findings}, indent=2))


if __name__ == "__main__":
    main()
