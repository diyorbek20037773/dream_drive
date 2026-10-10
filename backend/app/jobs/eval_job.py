"""RQ entry point for the synthetic LiteRoad scenario suite: ``run_scenario_suite_job(eval_id)``.

Runs ``app.evaluation.scenario_suite.run_scenario_suite`` (real closed-loop runs with the requested seeds) and stores
its Evaluation dict. Results are labelled synthetic scenario validation, separate from learned-video ML accuracy.
"""

from __future__ import annotations

import logging
from typing import Any

from rq.timeouts import JobTimeoutException

from app import metrics
from app.db import utc_now
from app.jobs.context import SHUTDOWN, WorkerContext, get_context
from app.logging_setup import log_context
from app.storage import repo

logger = logging.getLogger("dreamdrive.jobs.eval")

KIND = "evaluation"


class SuiteInterrupted(Exception):  # noqa: N818
    pass


def run_suite(**kwargs: Any) -> dict[str, Any]:
    from app.evaluation.scenario_suite import run_scenario_suite

    return run_scenario_suite(**kwargs)


def run_scenario_suite_job(eval_id: str) -> str:
    ctx = get_context()
    with log_context(run_id=eval_id):
        return _run(ctx, eval_id)


def _finish(ctx: WorkerContext, row: repo.EvaluationRow, state: str, **fields: Any) -> None:
    if repo.transition_evaluation(ctx.db, row.id, ("queued", "running"), state, finished_utc=utc_now(), **fields):
        metrics.JOBS_TOTAL.labels(KIND, state).inc()
        logger.info("evaluation finished", extra={"state": state, "code": fields.get("error_code")})


def _run(ctx: WorkerContext, eval_id: str) -> str:
    row = repo.get_evaluation_any(ctx.db, eval_id)
    if row is None:
        return "missing"
    if not repo.transition_evaluation(ctx.db, eval_id, ("queued",), "running"):
        return row.state
    seeds = list(row.seeds or [])
    out_dir = ctx.settings.evaluation_root / eval_id

    def progress(*_args: Any, **_kwargs: Any) -> None:
        if SHUTDOWN.is_set():
            raise SuiteInterrupted()

    try:
        result = run_suite(seeds=seeds, out_dir=out_dir, code_commit=ctx.settings.git_commit, progress_cb=progress)
    except SuiteInterrupted:
        _finish(ctx, row, "interrupted", error_code="worker_shutdown")
        return "interrupted"
    except JobTimeoutException:
        _finish(ctx, row, "failed", error_code="job_timeout")
        return "failed"
    except Exception:
        logger.exception("scenario suite crashed", extra={"code": "internal_error"})
        _finish(ctx, row, "failed", error_code="internal_error")
        return "failed"
    artifact_id = str(result.get("id")) if isinstance(result, dict) and result.get("id") else None
    _finish(ctx, row, "succeeded", summary=result, artifact_id=artifact_id)
    return "succeeded"
