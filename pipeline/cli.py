"""CLI entry: `finvid run --url ...`, `finvid serve`, `finvid costs`, `finvid clean`."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .config import DATA_DIR, ConfigError, check_openai_key, get_settings
from .context import AlreadyRunning, BudgetExceeded, RunContext, WorkdirLock, extract_video_id, run_stage
from .manifest import STAGE_ORDER, Manifest
from .pricing import ai_video_reference_cost
from .render.aivideo.base import AIVideoError

app = typer.Typer(add_completion=False, help="Cost-aware YouTube finance video -> short clips pipeline")
console = Console()


def _cli_log(msg: str) -> None:
    # stage logs start with "[s1_download]" etc.; markup=False stops Rich treating that as a style tag
    console.print(msg, markup=False, highlight=False)


DEFAULT_URL = "https://www.youtube.com/watch?v=KjAI9r8tnOs"


def _stages():
    from .stages import s1_download, s2_transcribe, s3_script, s4_render

    return [s1_download, s2_transcribe, s3_script, s4_render]


def _needs_openai(settings, until: str | None) -> bool:
    """Which stages this run reaches, and whether any of them calls OpenAI."""
    last = next((s for s in STAGE_ORDER if until and s.startswith(until)), STAGE_ORDER[-1])
    reached = STAGE_ORDER[: STAGE_ORDER.index(last) + 1]
    return (("s2_transcribe" in reached and settings.stt_provider == "openai")
            or "s3_script" in reached  # the LLM passes are OpenAI-only
            or ("s4_render" in reached and settings.tts_provider == "openai"))


def run_pipeline(url: str, *, dry_run: bool = False, force: bool = False,
                 max_clips: int | None = None, until: str | None = None,
                 log=_cli_log, data_dir: Path | None = None) -> RunContext:
    """Programmatic entry used by both the CLI and the web UI."""
    settings = get_settings()
    if not dry_run and _needs_openai(settings, until):
        check_openai_key(settings)  # before s1 downloads anything
    ctx = RunContext.create(settings, url, dry_run=dry_run, force=force, max_clips=max_clips,
                            log=log, data_dir=data_dir)
    log(f"video_id={ctx.video_id} workdir={ctx.workdir} budget=${settings.max_budget_usd} "
        f"max_clips={ctx.max_clips} dry_run={dry_run} force={force}")
    lock = WorkdirLock(ctx.workdir)
    if not dry_run:
        lock.acquire()  # cross-process duplicate guard: two `finvid run`s on one video = one bill
    try:
        for mod in _stages():
            run_stage(ctx, mod.STAGE, mod.stage_config(ctx), mod.execute)
            if until and mod.STAGE.startswith(until):
                break
    except BudgetExceeded as e:
        log(f"BUDGET GUARD: {e}")
        raise
    finally:
        if not dry_run:
            ctx.manifest.add_run({
                "started": ctx.started_at, "dry_run": dry_run, "force": force,
                "stages_run": ctx.stages_run, "stages_cached": ctx.stages_cached,
                "usd": round(ctx.spent_usd, 6),
            })
            lock.release()
    return ctx


@app.command()
def run(
    url: str = typer.Option(DEFAULT_URL, "--url", "-u", help="YouTube URL or 11-char video id"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Estimate only: no paid API calls, no download, nothing written (only fetches YouTube metadata)"),
    force: bool = typer.Option(False, "--force", help="Ignore cache and redo every stage"),
    max_clips: int | None = typer.Option(None, "--max-clips", help="Upper bound on clips generated"),
    until: str | None = typer.Option(None, "--until", help="Stop after this stage (s1|s2|s3|s4)"),
):
    """Run the full pipeline (idempotent: re-running the same video costs $0)."""
    try:
        ctx = run_pipeline(url, dry_run=dry_run, force=force, max_clips=max_clips, until=until)
    except BudgetExceeded:
        raise typer.Exit(code=2)
    except ConfigError as e:
        console.print(f"[bold red]CONFIG:[/] {e}")
        raise typer.Exit(code=3)
    except AlreadyRunning as e:
        console.print(f"[bold yellow]DUPLICATE GUARD:[/] {e}")
        raise typer.Exit(code=4)
    except AIVideoError as e:
        console.print("[bold red]AI VIDEO:[/] ", end="")
        console.print(str(e), markup=False, highlight=False)
        raise typer.Exit(code=5)
    _print_costs(ctx.manifest, this_run_usd=ctx.spent_usd, dry_run=dry_run, ctx=ctx)


def _resolve_workdir(url_or_name: str) -> tuple[Path, str]:
    """`--url` accepts a YouTube URL, an 11-char id, or the name of an existing
    data/<name>/ folder (e.g. `demo`, the run shipped with the repo)."""
    folder = DATA_DIR / url_or_name
    if "/" not in url_or_name and (folder / "manifest.json").exists():
        return folder, url_or_name
    try:
        vid = extract_video_id(url_or_name)
    except ValueError:
        have = sorted(p.parent.name for p in DATA_DIR.glob("*/manifest.json"))
        console.print(f"[red]{url_or_name!r} is not a YouTube URL/id and data/{url_or_name}/ has no manifest.[/] "
                      f"Processed folders: {', '.join(have) or 'none'}")
        raise typer.Exit(1)
    return DATA_DIR / vid, vid


@app.command()
def costs(url: str = typer.Option(DEFAULT_URL, "--url", "-u",
                                  help="YouTube URL/id, or a data/ folder name such as `demo`")):
    """Print the cost ledger recorded in data/<video_id>/manifest.json."""
    workdir, vid = _resolve_workdir(url)
    if not (workdir / "manifest.json").exists():
        console.print(f"no manifest for {vid} yet (try `finvid costs --url demo` for the bundled run)")
        raise typer.Exit(1)
    _print_costs(Manifest(workdir, vid, url))


@app.command()
def clean(url: str = typer.Option(DEFAULT_URL, "--url", "-u",
                                  help="YouTube URL/id, or a data/ folder name such as `demo`"),
          stage: str = typer.Option("s1", help="Invalidate from this stage onward (s1|s2|s3|s4)")):
    """Invalidate cache from a stage onward (files are kept, manifest entries dropped)."""
    workdir, vid = _resolve_workdir(url)
    m = Manifest(workdir, vid, url)
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
