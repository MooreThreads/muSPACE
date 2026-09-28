"""Command line interface."""

from __future__ import annotations

import json
import platform
from pathlib import Path

import click

from . import __git_version__, __version__
from .compatibility import compatibility_report
from .patcher import MigrationError, inspect_recipe, load_recipe, migrate_recipe, restore_recipe
from .patcher.migration import RECIPE_SLUGS


@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx: click.Context) -> None:
    """Inspect muSPACE and prepare supported model checkouts."""
    if ctx.invoked_subcommand is None:
        click.echo(f"muSPACE {__version__}")


@cli.command("env")
@click.option(
    "--profile", type=click.Choice(["core", "bevformer"]), default="core", show_default=True
)
@click.option(
    "--strict",
    is_flag=True,
    help="Exit non-zero unless the stack and MUSA device are fully validated.",
)
def environment(profile: str, strict: bool) -> None:
    """Print machine-readable environment and compatibility information."""
    report = compatibility_report(profile=profile)  # type: ignore[arg-type]
    payload = {
        "mu_space": __version__,
        "git_revision": __git_version__,
        "python": platform.python_version(),
        "compatibility": report.as_dict(),
    }
    click.echo(json.dumps(payload, indent=2, sort_keys=True))
    if strict and report.state != "validated":
        raise click.exceptions.Exit(1)


def _recipe_and_checkout(model: str, checkout: Path | None):
    recipe = load_recipe(model)
    return recipe, checkout if checkout is not None else Path(recipe.default_checkout)


@cli.command()
@click.argument("model", type=click.Choice(RECIPE_SLUGS, case_sensitive=False))
@click.option(
    "--source",
    "--checkout",
    "checkout",
    type=click.Path(path_type=Path),
    default=None,
    help="Official model checkout; defaults to the recipe's directory name.",
)
@click.option("--dry-run", is_flag=True, help="Inspect only; do not modify checkout.")
def migrate(model: str, checkout: Path | None, dry_run: bool) -> None:
    """Inspect or migrate an official model checkout."""
    try:
        recipe, target = _recipe_and_checkout(model, checkout)
        report = inspect_recipe(recipe, target) if dry_run else migrate_recipe(recipe, target)
    except MigrationError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(json.dumps(report.as_dict(), indent=2, sort_keys=True))


@cli.command()
@click.argument("model", type=click.Choice(RECIPE_SLUGS, case_sensitive=False))
@click.option(
    "--source",
    "--checkout",
    "checkout",
    type=click.Path(path_type=Path),
    default=None,
)
def restore(model: str, checkout: Path | None) -> None:
    """Restore a checkout changed by the muSPACE recipe."""
    try:
        recipe, target = _recipe_and_checkout(model, checkout)
        report = restore_recipe(recipe, target)
    except MigrationError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(json.dumps(report.as_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    cli()
