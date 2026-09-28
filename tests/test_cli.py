import json
import re
import subprocess
from pathlib import Path

import pytest
from click.testing import CliRunner

from mu_space.__main__ import cli
from mu_space.compatibility import CompatibilityReport
from mu_space.patcher import migration
from mu_space.patcher.migration import Recipe, migrate_recipe, restore_recipe

ALL_RECIPES = (
    "bevfusion",
    "bevformer",
    "flashocc",
    "lingbotva",
    "maptrv2",
    "openvla",
    "qcnet",
    "sparsedrive",
    "uniad",
)
# SparseDrive is integrated by the runtime patcher only and ships no checkout patches.
RECIPES = (
    "bevfusion",
    "bevformer",
    "flashocc",
    "lingbotva",
    "maptrv2",
    "openvla",
    "qcnet",
    "uniad",
)


def _fixture_recipe(slug: str, marker: str) -> Recipe:
    return Recipe(
        slug=slug,
        name=slug,
        repository="https://example.invalid/model.git",
        reference_commit="",
        marker_name=marker,
        patch_names=(),
        patch_digests={},
        default_checkout=slug,
        root=migration.Path("."),
    )


def _init_fixture_checkout(checkout, contents: str) -> str:
    checkout.mkdir()
    subprocess.run(["git", "init"], cwd=checkout, check=True, capture_output=True)
    (checkout / "target.py").write_text(contents, encoding="utf-8")
    subprocess.run(["git", "add", "target.py"], cwd=checkout, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=muSPACE tests",
            "-c",
            "user.email=tests@example.invalid",
            "commit",
            "-m",
            "fixture",
        ],
        cwd=checkout,
        check=True,
        capture_output=True,
    )
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip()


def test_module_default_output():
    result = CliRunner().invoke(cli, [])
    assert result.exit_code == 0
    assert result.output.startswith("muSPACE ")


def test_env_output_is_json():
    result = CliRunner().invoke(cli, ["env"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["compatibility"]["state"] in {
        "validated",
        "experimental",
        "unsupported",
        "unknown",
    }


def test_env_strict_returns_nonzero_for_unvalidated_stack(monkeypatch):
    report = CompatibilityReport(
        profile="core",
        state="unknown",
        reasons=("MUSA SDK version could not be detected",),
        versions={"torch": "2.9.1", "torch_musa": "2.9.1"},
    )
    monkeypatch.setattr("mu_space.__main__.compatibility_report", lambda profile: report)
    result = CliRunner().invoke(cli, ["env", "--strict"])
    assert result.exit_code != 0
    assert '"musa_sdk": null' in result.output


@pytest.mark.parametrize("slug", ALL_RECIPES)
def test_migrate_dry_run_reports_missing_checkout(tmp_path, slug):
    checkout = tmp_path / slug
    result = CliRunner().invoke(cli, ["migrate", slug, "--checkout", str(checkout), "--dry-run"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["checkout_exists"] is False
    assert payload["patch_state"] == "not-applied"
    assert payload["reference_commit"] == migration.load_recipe(slug).reference_commit


@pytest.mark.parametrize("slug", RECIPES)
def test_recipe_manifest_declares_every_shipped_patch(slug):
    recipe = migration.load_recipe(slug)
    assert recipe.patch_names
    assert set(recipe.patch_digests) == set(recipe.patch_names)
    shipped = sorted(path.name for path in (recipe.root / "patches").glob("*.patch"))
    declared = sorted(name for name in recipe.patch_names)
    assert declared == [name for name in shipped if not name.startswith("deps-")]
    # A stale digest must fail loudly instead of migrating a corrupted patch.
    migration._recipe_patches(recipe)


# `pip install` fills the wheel from the pyproject data-files table, so a recipe
# missing from that table still resolves from the source tree and only breaks for
# installed packages. These checks keep the table in step with the registry.
_PYPROJECT = Path(__file__).parents[1] / "pyproject.toml"
_DECLARED_RECIPE_FILES = (
    {
        name[len("model_examples/") :]
        for name in re.findall(r'"(model_examples/[^"]+)"', _PYPROJECT.read_text(encoding="utf-8"))
    }
    if _PYPROJECT.is_file()
    else set()
)
requires_source_tree = pytest.mark.skipif(
    not _PYPROJECT.is_file(), reason="pyproject.toml ships only in the source tree"
)


@requires_source_tree
def test_every_recipe_is_declared_as_package_data():
    declared = {name.split("/")[0] for name in _DECLARED_RECIPE_FILES}
    assert declared == set(migration.RECIPE_DIRECTORIES.values())


@requires_source_tree
@pytest.mark.parametrize("slug", RECIPES)
def test_recipe_declares_every_patch_as_package_data(slug):
    directory = migration.RECIPE_DIRECTORIES[slug]
    assert f"{directory}/manifest.yaml" in _DECLARED_RECIPE_FILES
    missing = [
        name
        for name in migration.load_recipe(slug).patch_names
        if f"{directory}/patches/{name}" not in _DECLARED_RECIPE_FILES
    ]
    assert missing == []


def test_sparsedrive_recipe_is_runtime_only():
    # SparseDrive is integrated by the runtime patcher only and ships no checkout patches.
    recipe = migration.load_recipe("sparsedrive")
    assert recipe.patch_names == ()
    assert migration._recipe_patches(recipe) == ()
    assert not (recipe.root / "patches").exists()


@pytest.mark.parametrize("slug", RECIPES)
def test_migrate_and_restore_are_idempotent(tmp_path, slug):
    checkout = tmp_path / slug
    commit = _init_fixture_checkout(checkout, "backend = 'cuda'\n")
    target = checkout / "target.py"
    recipe = _fixture_recipe(slug, f".mu_space_{slug}.json")
    patch = tmp_path / "fixture.patch"
    patch.write_text(
        "diff --git a/target.py b/target.py\n"
        "--- a/target.py\n"
        "+++ b/target.py\n"
        "@@ -1 +1 @@\n"
        "-backend = 'cuda'\n"
        "+backend = 'musa'\n",
        encoding="utf-8",
    )

    first = migrate_recipe(recipe, checkout, patch_path=patch, expected_commit=commit)
    second = migrate_recipe(recipe, checkout, patch_path=patch, expected_commit=commit)
    assert first.action == "applied"
    assert second.action == "already-applied"
    assert (checkout / recipe.marker_name).is_file()
    assert target.read_text(encoding="utf-8") == "backend = 'musa'\n"

    restored = restore_recipe(recipe, checkout, patch_path=patch, expected_commit=commit)
    restored_again = restore_recipe(recipe, checkout, patch_path=patch, expected_commit=commit)
    assert restored.action == "restored"
    assert restored_again.action == "already-restored"
    assert target.read_text(encoding="utf-8") == "backend = 'cuda'\n"
    assert subprocess.run(["git", "diff", "--quiet"], cwd=checkout, check=False).returncode == 0


def test_migrate_upgrades_a_partially_patched_checkout(tmp_path):
    checkout = tmp_path / "BEVFormer"
    commit = _init_fixture_checkout(checkout, "device = 'cuda'\nbackend = 'native'\n")
    target = checkout / "target.py"
    recipe = _fixture_recipe("bevformer", ".mu_space_bevformer.json")
    first = tmp_path / "first.patch"
    first.write_text(
        "diff --git a/target.py b/target.py\n--- a/target.py\n+++ b/target.py\n"
        "@@ -1,2 +1,2 @@\n-device = 'cuda'\n+device = 'musa'\n backend = 'native'\n",
        encoding="utf-8",
    )
    second = tmp_path / "second.patch"
    second.write_text(
        "diff --git a/target.py b/target.py\n--- a/target.py\n+++ b/target.py\n"
        "@@ -1,2 +1,2 @@\n device = 'musa'\n-backend = 'native'\n+backend = 'reference'\n",
        encoding="utf-8",
    )

    migrate_recipe(recipe, checkout, patch_path=first, expected_commit=commit)
    upgraded = migrate_recipe(recipe, checkout, patch_paths=[first, second], expected_commit=commit)
    assert upgraded.action == "upgraded"
    assert target.read_text(encoding="utf-8") == "device = 'musa'\nbackend = 'reference'\n"
    restore_recipe(recipe, checkout, patch_paths=[first, second], expected_commit=commit)
    assert target.read_text(encoding="utf-8") == "device = 'cuda'\nbackend = 'native'\n"


@pytest.mark.parametrize("slug", RECIPES)
def test_recipe_patches_resolve_from_installed_data_dir(tmp_path, monkeypatch, slug):
    source = migration.load_recipe(slug)
    directory = migration.RECIPE_DIRECTORIES[slug]
    installed = tmp_path / "share" / "mu-space" / "model_examples" / directory
    (installed / "patches").mkdir(parents=True)
    (installed / "manifest.yaml").write_bytes((source.root / "manifest.yaml").read_bytes())
    for name in source.patch_names:
        (installed / "patches" / name).write_bytes((source.root / "patches" / name).read_bytes())
    monkeypatch.setattr(
        migration,
        "__file__",
        str(tmp_path / "installed" / "mu_space" / "patcher" / "migration.py"),
    )
    monkeypatch.setattr(migration.sysconfig, "get_path", lambda *_args, **_kwargs: str(tmp_path))

    recipe = migration.load_recipe(slug)
    assert recipe.root == installed
    assert migration._recipe_patches(recipe) == tuple(
        installed / "patches" / name for name in source.patch_names
    )


def test_corrupted_patch_fails_the_manifest_digest_check(tmp_path, monkeypatch):
    source = migration.load_recipe("uniad")
    installed = tmp_path / "share" / "mu-space" / "model_examples" / "UniAD"
    (installed / "patches").mkdir(parents=True)
    (installed / "manifest.yaml").write_bytes((source.root / "manifest.yaml").read_bytes())
    for name in source.patch_names:
        (installed / "patches" / name).write_text("corrupted\n", encoding="utf-8")
    monkeypatch.setattr(
        migration,
        "__file__",
        str(tmp_path / "installed" / "mu_space" / "patcher" / "migration.py"),
    )
    monkeypatch.setattr(migration.sysconfig, "get_path", lambda *_args, **_kwargs: str(tmp_path))

    with pytest.raises(migration.MigrationError, match="manifest digest"):
        migration._recipe_patches(migration.load_recipe("uniad"))


def test_unknown_model_is_rejected():
    with pytest.raises(migration.MigrationError, match="unknown model"):
        migration.load_recipe("not-a-model")
