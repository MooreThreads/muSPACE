"""Idempotent migration of pinned official model checkouts."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import sysconfig
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from mu_space.compatibility import compatibility_report

RECIPE_DIRECTORIES = {
    "bevfusion": "BEVFusion",
    "bevformer": "BEVFormer",
    "bevdepth": "BEVDepth",
    "flashocc": "FlashOCC",
    "lingbotva": "LingBotVA",
    "maptrv2": "MapTRv2",
    "openvla": "OpenVLA",
    "qcnet": "QCNet",
    "qwendrive": "QwenDrive",
    "sparsedrive": "SparseDrive",
    "uniad": "UniAD",
}
RECIPE_SLUGS = tuple(RECIPE_DIRECTORIES)
MANIFEST_NAME = "manifest.yaml"


class MigrationError(RuntimeError):
    """Raised when a checkout cannot be migrated or restored safely."""


@dataclass(frozen=True)
class Recipe:
    """Checkout contract for one model, as declared by its recipe manifest."""

    slug: str
    name: str
    repository: str
    reference_commit: str
    marker_name: str
    patch_names: tuple[str, ...]
    patch_digests: dict[str, str]
    default_checkout: str
    root: Path


@dataclass(frozen=True)
class MigrationReport:
    checkout: str
    checkout_exists: bool
    git_commit: str | None
    reference_commit: str
    dependency_state: str
    patch_state: str
    action: str = "inspect"
    model: str = "BEVFormer"

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _run_git(checkout: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["git", "-c", f"safe.directory={checkout}", *args],
            cwd=checkout,
            check=check,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise MigrationError(f"failed to run git: {exc}") from exc


def _git_commit(checkout: Path) -> str | None:
    result = _run_git(checkout, "rev-parse", "HEAD", check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def _recipe_roots(directory: str) -> tuple[Path, ...]:
    source_root = Path(__file__).resolve().parents[2] / "model_examples" / directory
    prefixes = (
        sysconfig.get_path("data") or sys.prefix,
        sys.prefix,
        sys.base_prefix,
        "/usr/local",
        "/usr",
    )
    seen_prefixes = set()
    installed_roots = []
    for prefix in prefixes:
        if prefix in seen_prefixes:
            continue
        seen_prefixes.add(prefix)
        installed_roots.append(Path(prefix) / "share/mu-space/model_examples" / directory)
    return (source_root, *installed_roots)


def load_recipe(slug: str) -> Recipe:
    """Read one model's checkout contract from its recipe manifest."""

    slug = slug.lower()
    try:
        directory = RECIPE_DIRECTORIES[slug]
    except KeyError as exc:
        raise MigrationError(
            f"unknown model {slug!r}; expected one of {', '.join(RECIPE_SLUGS)}"
        ) from exc
    for root in _recipe_roots(directory):
        manifest = root / MANIFEST_NAME
        if not manifest.is_file():
            continue
        try:
            payload = yaml.safe_load(manifest.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError) as exc:
            raise MigrationError(f"unreadable recipe manifest: {manifest}") from exc
        entries = payload.get("patches") or []
        return Recipe(
            slug=slug,
            name=str(payload.get("name", directory)),
            repository=str(payload.get("repository", "")),
            reference_commit=str(payload.get("reference_commit", "")),
            marker_name=str(payload.get("patch_marker", f".mu_space_{slug}.json")),
            patch_names=tuple(Path(str(entry["path"])).name for entry in entries),
            patch_digests={
                Path(str(entry["path"])).name: str(entry["sha256"])
                for entry in entries
                if entry.get("sha256")
            },
            default_checkout=str(payload.get("default_checkout", directory)),
            root=root,
        )
    raise MigrationError(f"{directory} recipe manifest is not installed: {MANIFEST_NAME}")


def _verify_digests(recipe: Recipe, patches: Sequence[Path]) -> None:
    """Fail loudly when a shipped patch no longer matches its manifest digest."""

    for patch in patches:
        expected = recipe.patch_digests.get(patch.name)
        if expected is None:
            continue
        actual = hashlib.sha256(patch.read_bytes()).hexdigest()
        if actual != expected:
            raise MigrationError(
                f"{recipe.name} patch {patch.name} does not match its manifest digest: "
                f"expected {expected}, got {actual}"
            )


def _recipe_patches(recipe: Recipe) -> tuple[Path, ...]:
    if not recipe.patch_names:
        return ()
    for root in (recipe.root, *_recipe_roots(RECIPE_DIRECTORIES[recipe.slug])):
        patches = tuple(root / "patches" / name for name in recipe.patch_names)
        if all(path.is_file() for path in patches):
            _verify_digests(recipe, patches)
            return patches
    raise MigrationError(
        f"{recipe.name} recipe patches are not installed: " + ", ".join(recipe.patch_names)
    )


def _resolve_patches(
    recipe: Recipe,
    patch_path: Path | None,
    patch_paths: Sequence[Path] | None,
) -> tuple[Path, ...]:
    if patch_path is not None and patch_paths is not None:
        raise MigrationError("pass patch_path or patch_paths, not both")
    if patch_paths is not None:
        selected: Sequence[Path] = patch_paths
    elif patch_path is not None:
        selected = (patch_path,)
    else:
        selected = _recipe_patches(recipe)
    return tuple(Path(path).resolve() for path in selected)


def _patch_digest(patches: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for patch in patches:
        digest.update(patch.name.encode())
        digest.update(b"\0")
        digest.update(patch.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _patch_check(checkout: Path, patch: Path, *, reverse: bool = False) -> bool:
    args = ["apply", "--check", "--unidiff-zero", "--ignore-space-change"]
    if reverse:
        args.append("--reverse")
    args.append(str(patch))
    return _run_git(checkout, *args, check=False).returncode == 0


def _individual_states(checkout: Path, patches: Sequence[Path]) -> list[str]:
    states = []
    for patch in patches:
        if _patch_check(checkout, patch):
            states.append("not-applied")
        elif _patch_check(checkout, patch, reverse=True):
            states.append("applied")
        else:
            states.append("conflict")
    return states


def _tracked_diff_digest(checkout: Path) -> str:
    diff = _run_git(checkout, "diff", "--binary").stdout.encode()
    return hashlib.sha256(diff).hexdigest()


def _tracked_worktree_is_clean(checkout: Path) -> bool:
    return (
        _run_git(checkout, "diff", "--quiet", check=False).returncode == 0
        and _run_git(checkout, "diff", "--cached", "--quiet", check=False).returncode == 0
    )


def _patched_files(checkout: Path, patches: Sequence[Path]) -> list[str]:
    files = set()
    for patch in patches:
        result = _run_git(
            checkout, "apply", "--numstat", "--unidiff-zero", "--ignore-space-change", str(patch)
        )
        files.update(line.split("\t", 2)[-1] for line in result.stdout.splitlines() if line)
    return sorted(files)


def _read_marker(checkout: Path, recipe: Recipe) -> dict[str, object] | None:
    marker = checkout / recipe.marker_name
    if not marker.is_file():
        return None
    try:
        return json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"invalid": True}


def _patch_state(checkout: Path, recipe: Recipe, patches: Sequence[Path]) -> str:
    marker = _read_marker(checkout, recipe)
    if marker is not None:
        if marker.get("invalid") or marker.get("checkout_diff_sha256") != _tracked_diff_digest(
            checkout
        ):
            return "drifted"
        if marker.get("patches_sha256") == _patch_digest(patches):
            return "applied"
    states = _individual_states(checkout, patches)
    if "conflict" in states:
        return "conflict"
    if all(state == "not-applied" for state in states):
        return "not-applied"
    if all(state == "applied" for state in states):
        if marker and marker.get("patches_sha256") == _patch_digest(patches):
            return "applied"
        return "applied-unmarked"
    return "partial"


def _validate_checkout(checkout: Path, recipe: Recipe, expected_commit: str) -> str:
    if not checkout.is_dir():
        raise MigrationError(f"{recipe.name} checkout does not exist: {checkout}")
    commit = _git_commit(checkout)
    if commit is None:
        raise MigrationError(f"not a git checkout: {checkout}")
    if commit != expected_commit:
        raise MigrationError(
            f"{recipe.name} commit mismatch: expected {expected_commit}, got {commit}"
        )
    return commit


def _as_recipe(model: str | Recipe) -> Recipe:
    return model if isinstance(model, Recipe) else load_recipe(model)


def inspect_recipe(
    model: str | Recipe,
    checkout: Path,
    *,
    patch_path: Path | None = None,
    patch_paths: Sequence[Path] | None = None,
) -> MigrationReport:
    recipe = _as_recipe(model)
    checkout = checkout.resolve()
    exists = checkout.is_dir()
    commit = _git_commit(checkout) if exists else None
    state = (
        _patch_state(checkout, recipe, _resolve_patches(recipe, patch_path, patch_paths))
        if exists and commit is not None
        else "not-applied"
    )
    return MigrationReport(
        checkout=str(checkout),
        checkout_exists=exists,
        git_commit=commit,
        reference_commit=recipe.reference_commit,
        dependency_state=compatibility_report().state,
        patch_state=state,
        model=recipe.name,
    )


def migrate_recipe(
    model: str | Recipe,
    checkout: Path,
    *,
    patch_path: Path | None = None,
    patch_paths: Sequence[Path] | None = None,
    expected_commit: str | None = None,
) -> MigrationReport:
    recipe = _as_recipe(model)
    checkout = checkout.resolve()
    commit = _validate_checkout(checkout, recipe, expected_commit or recipe.reference_commit)
    patches = _resolve_patches(recipe, patch_path, patch_paths)
    state = _patch_state(checkout, recipe, patches)
    if state == "applied":
        action = "already-applied"
    elif state in {"not-applied", "partial", "applied-unmarked"}:
        if state == "not-applied" and not _tracked_worktree_is_clean(checkout):
            raise MigrationError("refusing to migrate a checkout with tracked changes")
        for patch, patch_state in zip(patches, _individual_states(checkout, patches), strict=True):
            if patch_state == "not-applied":
                _run_git(checkout, "apply", "--unidiff-zero", "--ignore-space-change", str(patch))
        metadata = {
            "model": recipe.name,
            "reference_commit": commit,
            "patches": [patch.name for patch in patches],
            "patches_sha256": _patch_digest(patches),
            "patched_files": _patched_files(checkout, patches),
            "checkout_diff_sha256": _tracked_diff_digest(checkout),
        }
        (checkout / recipe.marker_name).write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        action = {
            "not-applied": "applied",
            "partial": "upgraded",
            "applied-unmarked": "adopted-existing-patches",
        }[state]
    else:
        raise MigrationError(f"cannot migrate checkout with patch state: {state}")
    return MigrationReport(
        checkout=str(checkout),
        checkout_exists=True,
        git_commit=commit,
        reference_commit=recipe.reference_commit,
        dependency_state=compatibility_report().state,
        patch_state="applied",
        action=action,
        model=recipe.name,
    )


def restore_recipe(
    model: str | Recipe,
    checkout: Path,
    *,
    patch_path: Path | None = None,
    patch_paths: Sequence[Path] | None = None,
    expected_commit: str | None = None,
) -> MigrationReport:
    recipe = _as_recipe(model)
    checkout = checkout.resolve()
    commit = _validate_checkout(checkout, recipe, expected_commit or recipe.reference_commit)
    patches = _resolve_patches(recipe, patch_path, patch_paths)
    state = _patch_state(checkout, recipe, patches)
    if state == "not-applied":
        action = "already-restored"
    elif state in {"applied", "applied-unmarked", "partial"}:
        metadata = _read_marker(checkout, recipe) or {}
        for patch in reversed(patches):
            if state == "applied" or _patch_check(checkout, patch, reverse=True):
                _run_git(
                    checkout,
                    "apply",
                    "--reverse",
                    "--unidiff-zero",
                    "--ignore-space-change",
                    str(patch),
                )
        if not _tracked_worktree_is_clean(checkout):
            files = metadata.get("patched_files", []) or _patched_files(checkout, patches)
            files = [
                path
                for path in files
                if _run_git(
                    checkout, "ls-files", "--error-unmatch", "--", path, check=False
                ).returncode
                == 0
            ]
            if (
                files
                and _run_git(
                    checkout, "diff", "--quiet", "--ignore-space-at-eol", check=False
                ).returncode
                == 0
            ):
                _run_git(checkout, "restore", "--worktree", "--", *files)
        (checkout / recipe.marker_name).unlink(missing_ok=True)
        if not _tracked_worktree_is_clean(checkout):
            raise MigrationError("restore left unexpected tracked changes")
        action = "restored"
    else:
        raise MigrationError(f"cannot restore checkout with patch state: {state}")
    return MigrationReport(
        checkout=str(checkout),
        checkout_exists=True,
        git_commit=commit,
        reference_commit=recipe.reference_commit,
        dependency_state=compatibility_report().state,
        patch_state="not-applied",
        action=action,
        model=recipe.name,
    )


def inspect_bevformer(checkout: Path, **kwargs) -> MigrationReport:
    return inspect_recipe("bevformer", checkout, **kwargs)


def migrate_bevformer(checkout: Path, **kwargs) -> MigrationReport:
    return migrate_recipe("bevformer", checkout, **kwargs)


def restore_bevformer(checkout: Path, **kwargs) -> MigrationReport:
    return restore_recipe("bevformer", checkout, **kwargs)
