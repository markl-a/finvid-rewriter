"""Core guarantees: manifest idempotency, budget guard, dry-run never records."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline.config import Settings
from pipeline.context import BudgetExceeded, RunContext, run_stage
from pipeline.manifest import Manifest, StageResult, config_hash
from pipeline.models import CostEntry
from pipeline.pricing import llm_cost, stt_cost

URL = "https://www.youtube.com/watch?v=KjAI9r8tnOs"


def _ctx(tmp_path: Path, **kw) -> RunContext:
    s = Settings(openai_api_key="x", max_budget_usd=kw.pop("budget", 1.0))
    return RunContext.create(s, URL, data_dir=tmp_path, **kw)


def _fake_stage(calls: list, usd: float = 0.01):
    def fn(ctx: RunContext) -> StageResult:
        calls.append(1)
        out = ctx.path("out.txt")
        out.write_text("x", encoding="utf-8")
        cost = CostEntry(stage="t", provider="p", model="m", unit="call", quantity=1,
                         unit_price_usd=usd, usd=usd, estimated=ctx.dry_run)
        return StageResult(outputs={} if ctx.dry_run else {"out": "out.txt"}, costs=[cost])
    return fn


def test_second_run_same_config_is_cached_and_free(tmp_path):
    calls: list = []
    cfg = {"a": 1}
    ctx = _ctx(tmp_path)
    r1 = run_stage(ctx, "s1_download", cfg, _fake_stage(calls))
    assert not r1.cached and ctx.spent_usd == pytest.approx(0.01)

    ctx2 = _ctx(tmp_path)  # fresh context, same workdir -> reads manifest from disk
    r2 = run_stage(ctx2, "s1_download", cfg, _fake_stage(calls))
    assert r2.cached and ctx2.spent_usd == 0 and len(calls) == 1
    assert ctx2.stages_cached == ["s1_download"]


def test_changed_config_reruns(tmp_path):
    calls: list = []
    ctx = _ctx(tmp_path)
    run_stage(ctx, "s1_download", {"a": 1}, _fake_stage(calls))
    run_stage(_ctx(tmp_path), "s1_download", {"a": 2}, _fake_stage(calls))
    assert len(calls) == 2


def test_missing_output_file_invalidates_cache(tmp_path):
    calls: list = []
    ctx = _ctx(tmp_path)
    run_stage(ctx, "s1_download", {"a": 1}, _fake_stage(calls))
    ctx.path("out.txt").unlink()
    run_stage(_ctx(tmp_path), "s1_download", {"a": 1}, _fake_stage(calls))
    assert len(calls) == 2


def test_force_bypasses_cache(tmp_path):
    calls: list = []
    run_stage(_ctx(tmp_path), "s1_download", {"a": 1}, _fake_stage(calls))
    run_stage(_ctx(tmp_path, force=True), "s1_download", {"a": 1}, _fake_stage(calls))
    assert len(calls) == 2


def test_rerunning_early_stage_invalidates_later_stages(tmp_path):
    calls: list = []
    ctx = _ctx(tmp_path)
    run_stage(ctx, "s1_download", {"a": 1}, _fake_stage(calls))
    run_stage(ctx, "s2_transcribe", {"b": 1}, _fake_stage(calls))
    assert set(ctx.manifest.data["stages"]) == {"s1_download", "s2_transcribe"}
    ctx3 = _ctx(tmp_path, force=True)
    run_stage(ctx3, "s1_download", {"a": 1}, _fake_stage(calls))
    assert "s2_transcribe" not in ctx3.manifest.data["stages"]


def test_dry_run_records_nothing_and_spends_nothing(tmp_path):
    calls: list = []
    ctx = _ctx(tmp_path, dry_run=True)
    r = run_stage(ctx, "s1_download", {"a": 1}, _fake_stage(calls))
    assert r.estimated_usd == pytest.approx(0.01) and r.usd == 0
    assert ctx.spent_usd == 0
    assert ctx.manifest.data["stages"] == {}
    assert not ctx.path("manifest.json").read_text(encoding="utf-8").count("s1_download")


def test_budget_guard_aborts_before_call(tmp_path):
    ctx = _ctx(tmp_path, budget=0.05)
    ctx.charge(0.04, "stt")  # fine
    ctx.spent_usd = 0.04
    with pytest.raises(BudgetExceeded):
        ctx.charge(0.02, "pass B")


def test_manifest_ledger_totals(tmp_path):
    m = Manifest(tmp_path / "v", "v", URL)
    real = stt_cost("s2_transcribe", "openai", "gpt-4o-mini-transcribe", 600)
    est = stt_cost("s2_transcribe", "openai", "gpt-4o-mini-transcribe", 600, estimated=True)
    (tmp_path / "v" / "t.json").write_text("{}", encoding="utf-8")
    m.record("s2_transcribe", {"x": 1}, StageResult(outputs={"t": "t.json"}, costs=[real, est]))
    assert real.usd == pytest.approx(0.03)
    assert m.total_usd() == pytest.approx(0.03)
    assert m.total_usd(include_estimated=True) == pytest.approx(0.06)
    reloaded = json.loads(m.path.read_text(encoding="utf-8"))
    assert reloaded["stages"]["s2_transcribe"]["config_hash"] == config_hash({"x": 1})


def test_llm_cost_split():
    entries = llm_cost("s3_script", "openai", "gpt-5-mini", 1_000_000, 100_000)
    assert entries[0].usd == pytest.approx(0.25)
    assert entries[1].usd == pytest.approx(0.20)
