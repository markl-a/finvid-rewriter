"""CLI entry: `finvid run --url ...`, `finvid serve`, `finvid costs`, `finvid clean`."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .config import DATA_DIR, get_settings
from .context import BudgetExceeded, RunContext, extract_video_id, run_stage
from .manifest import STAGE_ORDER, Manifest
from .pricing import ai_video_reference_cost

app = typer.Typer(add_completion=False, help="Cost-aware YouTube finance video -> short clips pipeline")
console = Console()

DEFAULT_URL = "https://www.youtube.com/watch?v=KjAI9r8tnOs"


def _stages():
    from .stages import s1_download, s2_transcribe, s3_script, s4_render

    return [s1_download, s2_transcribe, s3_script, s4_render]


def run_pipeline(url: str, *, dry_run: bool = False, force: bool = False,
                 max_clips: int | None = None, until: str | None = None,
                 log=console.print, data_dir: Path | None = None) -> RunContext:
    """Programmatic entry used by both the CLI and the web UI."""
    settings = get_settings()
    ctx = RunContext.create(settings, url, dry_run=dry_run, force=force, max_clips=max_clips,
                            log=log, data_dir=data_dir)
    log(f"video_id={ctx.video_id} workdir={ctx.workdir} budget=${settings.max_budget_usd} "
        f"max_clips={ctx.max_clips} dry_run={dry_run} force={force}")
    try:
        for mod in _stages():
            run_stage(ctx, mod.STAGE, mod.stage_config(ctx), mod.execute)
            if until and mod.STAGE.startswith(until):
                break
    except BudgetExceeded as e:
        log(f"[bold red]BUDGET GUARD:[/] {e}")
        raise
    finally:
        if not dry_run:
            ctx.manifest.add_run({
                "started": ctx.started_at, "dry_run": dry_run, "force": force,
                "stages_run": ctx.stages_run, "stages_cached": ctx.stages_cached,
                "usd": round(ctx.spent_usd, 6),
            })
    return ctx


@app.command()
def run(
    url: str = typer.Option(DEFAULT_URL, "--url", "-u", help="YouTube URL or 11-char video id"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Estimate cost only; no API calls, no downloads"),
    force: bool = typer.Option(False, "--force", help="Ignore cache and redo every stage"),
    max_clips: int | None = typer.Option(None, "--max-clips", help="Upper bound on clips generated"),
    until: str | None = typer.Option(None, "--until", help="Stop after this stage (s1|s2|s3|s4)"),
):
    """Run the full pipeline (idempotent: re-running the same video costs $0)."""
    try:
        ctx = run_pipeline(url, dry_run=dry_run, force=force, max_clips=max_clips, until=until)
    except BudgetExceeded:
        raise typer.Exit(code=2)
    _print_costs(ctx.manifest, this_run_usd=ctx.spent_usd, dry_run=dry_run, ctx=ctx)


@app.command()
def costs(url: str = typer.Option(DEFAULT_URL, "--url", "-u")):
    """Print the cost ledger recorded in data/<video_id>/manifest.json."""
    vid = extract_video_id(url)
    workdir = DATA_DIR / vid
    if not (workdir / "manifest.json").exists():
        console.print(f"no manifest for {vid} yet")
        raise typer.Exit(1)
    _print_costs(Manifest(workdir, vid, url))


@app.command()
def clean(url: str = typer.Option(DEFAULT_URL, "--url", "-u"),
          stage: str = typer.Option("s1", help="Invalidate from this stage onward (s1|s2|s3|s4)")):
    """Invalidate cache from a stage onward (files are kept, manifest entries dropped)."""
    vid = extract_video_id(url)
    m = Manifest(DATA_DIR / vid, vid, url)
    full = next(s for s in STAGE_ORDER if s.startswith(stage))
    m.invalidate_from(full)
    console.print(f"invalidated {full} and later for {vid}")


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000):
    """Start the local web UI (FastAPI + one HTML page)."""
    import uvicorn

    from .ui.server import app as web_app

    console.print(f"open http://{host}:{port}")
    uvicorn.run(web_app, host=host, port=port, log_level="warning")


def _print_costs(m: Manifest, this_run_usd: float | None = None, dry_run: bool = False,
                 ctx: RunContext | None = None) -> None:
    t = Table(title="cost ledger (USD)")
    for col in ("stage", "provider/model", "unit", "qty", "unit price", "usd", "est?", "note"):
        t.add_column(col)
    entries = m.all_costs()
    if dry_run and ctx is not None:
        entries = [c for r in ctx.stage_results.values() for c in r.costs]
    for c in entries:
        t.add_row(c.stage, f"{c.provider}/{c.model}", c.unit, f"{c.quantity:g}",
                  f"{c.unit_price_usd:.6g}", f"{c.usd:.4f}", "est" if c.estimated else "", c.note)
    console.print(t)
    real = sum(c.usd for c in entries if not c.estimated)
    est = sum(c.usd for c in entries if c.estimated and c.provider != "reference")
    ref = sum(c.usd for c in entries if c.provider == "reference")
    console.print(f"actual spent (all runs, cached included): ${real:.4f}")
    if dry_run:
        console.print(f"estimated for this dry run: ${est:.4f}")
    if this_run_usd is not None and not dry_run:
        console.print(f"[bold]this run: ${this_run_usd:.4f}[/] "
                      f"(cached stages: {', '.join(ctx.stages_cached) if ctx and ctx.stages_cached else 'none'})")
    if ref:
        console.print(f"for comparison, AI-video-API rendering of the same clips: ~${ref:.2f}")


if __name__ == "__main__":
    app()
