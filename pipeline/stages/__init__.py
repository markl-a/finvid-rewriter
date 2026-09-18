"""Each stage module exposes:

STAGE: str                                  # "s1_download" ...
def stage_config(ctx: RunContext) -> dict   # everything that affects the output -> cache key
def execute(ctx: RunContext) -> StageResult # honours ctx.dry_run (no API/network, estimates only)
"""
