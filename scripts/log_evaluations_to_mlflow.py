#!/usr/bin/env python3
"""Log DreamDrive evaluation JSON files (DATA_ROOT/evaluations) and the resolved model manifest to MLflow.

Runs inside the official MLflow image (compose.ops.yaml service ``mlflow-log``; ``make mlflow-log``). Local JSON
written by the application is the source of truth; MLflow only indexes it, so nothing here is invented:

* only metrics with ``status == "measured"`` and a numeric value are logged as MLflow metrics; every other metric
  (``not_evaluated``, ``unsupported``...) is recorded as a ``metric_status.<name>`` tag with its note;
* runs are idempotent: an evaluation already logged (tag ``dreamdrive.eval_id``) is skipped;
* only terminal evaluations (succeeded / failed / interrupted) are logged.

Exit codes: 0 ok (or tracking outage while ENABLE_MLFLOW is false: warning only), 1 invalid input files,
2 tracking server unreachable while ENABLE_MLFLOW is true, 64 usage error.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

TERMINAL_STATES = {"succeeded", "failed", "interrupted"}
REQUIRED_KEYS = {"id", "kind", "state", "metrics"}
EXIT_OK, EXIT_INVALID, EXIT_TRACKING_DOWN, EXIT_USAGE = 0, 1, 2, 64
_KEY_SAFE = re.compile(r"[^A-Za-z0-9_\-. /]")
MAX_TAG_LEN = 5000
MAX_PARAM_LEN = 6000


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def _key(name: str) -> str:
    return _KEY_SAFE.sub("_", name)[:250]


def _ms(iso: str | None) -> int | None:
    if not iso:
        return None
    try:
        return int(datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp() * 1000)
    except ValueError:
        return None


def find_evaluation_files(root: Path) -> list[Path]:
    """Flat ``<id>.json`` files plus ``<id>/evaluation.json`` (both layouts are accepted)."""
    if not root.is_dir():
        return []
    files = {p for p in root.glob("*.json") if p.is_file()}
    files |= {p for p in root.glob("*/evaluation.json") if p.is_file()}
    return sorted(files)


def load_evaluation(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("top-level JSON value is not an object")
    missing = REQUIRED_KEYS - data.keys()
    if missing:
        raise ValueError(f"missing keys: {sorted(missing)}")
    if not isinstance(data["metrics"], dict):
        raise ValueError("metrics is not an object")
    return data


def load_manifest(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    import yaml  # PyYAML is a dependency of mlflow

    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else None


def manifest_params(manifest: dict[str, Any] | None) -> dict[str, str]:
    if not manifest:
        return {"model.manifest": "missing"}
    params: dict[str, str] = {"model.manifest_state": str(manifest.get("state"))}
    for model in manifest.get("models") or []:
        if not isinstance(model, dict) or not model.get("required_for_video"):
            continue
        mid = str(model.get("id"))
        params.update({
            "model.id": mid,
            "model.repository_revision": str(model.get("repository_revision")),
            "model.checkpoint_sha256": str(model.get("checkpoint_sha256")),
            "model.checkpoint_size_bytes": str(model.get("checkpoint_size_bytes")),
            "model.format": str(model.get("format")),
            "model.class_map": json.dumps(model.get("supported_class_map"), sort_keys=True),
            "model.input_shape_policy": str(model.get("input_shape_policy")),
            "model.torch_version": str(model.get("torch_version")),
        })
    return {k: v[:MAX_PARAM_LEN] for k, v in params.items()}


def build_run_payload(evaluation: dict[str, Any], manifest: dict[str, Any] | None) -> dict[str, Any]:
    """Pure mapping from one evaluation to MLflow params/metrics/tags (unit-tested without a server)."""
    metrics: dict[str, float] = {}
    tags: dict[str, str] = {
        "dreamdrive.eval_id": str(evaluation["id"]),
        "dreamdrive.kind": str(evaluation["kind"]),
        "dreamdrive.state": str(evaluation["state"]),
        "dreamdrive.source": str(evaluation.get("source")),
        "dreamdrive.code_commit": str(evaluation.get("code_commit")),
        "dreamdrive.model_sha256": str(evaluation.get("model_sha256")),
    }
    for name, metric in sorted(evaluation["metrics"].items()):
        if not isinstance(metric, dict):
            tags[_key(f"metric_status.{name}")] = "invalid: not an object"
            continue
        value, status = metric.get("value"), metric.get("status")
        if status == "measured" and isinstance(value, int | float) and not isinstance(value, bool):
            metrics[_key(name)] = float(value)
            tags[_key(f"metric_unit.{name}")] = str(metric.get("unit"))
            tags[_key(f"metric_samples.{name}")] = str(metric.get("sample_count"))
        else:
            note = str(metric.get("note") or "")
            tags[_key(f"metric_status.{name}")] = f"{status}: {note}"[:MAX_TAG_LEN]
    limitations = evaluation.get("limitations") or []
    if limitations:
        tags["dreamdrive.limitations"] = json.dumps(limitations, ensure_ascii=False)[:MAX_TAG_LEN]
    params = {
        "sample_count": str(evaluation.get("sample_count")),
        "seeds": json.dumps(evaluation.get("seeds")),
        "source": str(evaluation.get("source")),
        "code_commit": str(evaluation.get("code_commit")),
        "eval_model_sha256": str(evaluation.get("model_sha256")),
    }
    params.update(manifest_params(manifest))
    return {
        "experiment": f"dreamdrive/{evaluation['kind']}",
        "run_name": f"{evaluation['kind']}-{evaluation['id'][:12]}",
        "start_ms": _ms(evaluation.get("created_utc")),
        "end_ms": _ms(evaluation.get("finished_utc")),
        "status": "FINISHED" if evaluation["state"] == "succeeded" else "FAILED",
        "params": params,
        "metrics": metrics,
        "tags": tags,
    }


def _chunks(items: list[Any], size: int) -> Iterable[list[Any]]:
    for i in range(0, len(items), size):
        yield items[i:i + size]


def log_one(client: Any, payload: dict[str, Any], files: list[Path]) -> str:
    from mlflow.entities import Metric, Param, RunTag

    exp = client.get_experiment_by_name(payload["experiment"])
    exp_id = exp.experiment_id if exp else client.create_experiment(payload["experiment"])
    run = client.create_run(exp_id, start_time=payload["start_ms"], run_name=payload["run_name"],
                            tags={"dreamdrive.eval_id": payload["tags"]["dreamdrive.eval_id"]})
    run_id = run.info.run_id
    step_ts = payload["end_ms"] or payload["start_ms"] or 0
    params = [Param(k, v) for k, v in payload["params"].items()]
    tags = [RunTag(k, v) for k, v in payload["tags"].items()]
    metrics = [Metric(k, v, step_ts, 0) for k, v in payload["metrics"].items()]
    for batch in _chunks(params, 100):
        client.log_batch(run_id, params=batch)
    for batch in _chunks(tags, 100):
        client.log_batch(run_id, tags=batch)
    for batch in _chunks(metrics, 1000):
        client.log_batch(run_id, metrics=batch)
    for f in files:
        client.log_artifact(run_id, str(f))
    client.set_terminated(run_id, status=payload["status"], end_time=payload["end_ms"])
    return run_id


def already_logged(client: Any, experiment: str, eval_id: str) -> bool:
    exp = client.get_experiment_by_name(experiment)
    if exp is None:
        return False
    found = client.search_runs([exp.experiment_id], filter_string=f"tags.`dreamdrive.eval_id` = '{eval_id}'",
                               max_results=1)
    return len(found) > 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Log DreamDrive evaluation JSON to MLflow.")
    parser.add_argument("--data-root", type=Path, default=Path(os.environ.get("DATA_ROOT", "/data")))
    parser.add_argument("--manifest", type=Path,
                        default=Path(os.environ.get("MODEL_MANIFEST_PATH", "/app/config/models.resolved.yaml")))
    parser.add_argument("--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    parser.add_argument("--dry-run", action="store_true", help="print the payloads; do not contact MLflow")
    args = parser.parse_args(argv)
    enabled = _truthy(os.environ.get("ENABLE_MLFLOW"))

    eval_dir = args.data_root / "evaluations"
    invalid = 0
    loaded: list[tuple[Path, dict[str, Any]]] = []
    for path in find_evaluation_files(eval_dir):
        try:
            evaluation = load_evaluation(path)
        except (OSError, ValueError) as exc:  # json.JSONDecodeError is a ValueError
            invalid += 1
            print(f"ERROR invalid evaluation file {path.name}: {exc}", file=sys.stderr)
            continue
        if evaluation["state"] not in TERMINAL_STATES:
            print(f"skip {evaluation['id']}: state {evaluation['state']} is not terminal")
            continue
        loaded.append((path, evaluation))
    manifest = load_manifest(args.manifest)
    if manifest is None:
        print(f"WARNING model manifest not readable at {args.manifest}; logging without model identity",
              file=sys.stderr)
    print(f"found {len(loaded)} terminal evaluation(s) in {eval_dir} ({invalid} invalid)")

    if args.dry_run:
        for path, evaluation in loaded:
            print(json.dumps({"file": path.name, **build_run_payload(evaluation, manifest)}, sort_keys=True))
        return EXIT_INVALID if invalid else EXIT_OK

    # Bound MLflow's HTTP retry/backoff so an outage is reported in seconds, not minutes.
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "10")
    from mlflow import MlflowClient
    from mlflow.exceptions import MlflowException

    client = MlflowClient(tracking_uri=args.tracking_uri)
    try:
        client.search_experiments(max_results=1)
    except Exception as exc:  # noqa: BLE001 - any transport/server error means "tracking unavailable"
        level = "ERROR" if enabled else "WARNING"
        print(f"{level} MLflow tracking at {args.tracking_uri} unavailable: {type(exc).__name__}", file=sys.stderr)
        if enabled:
            return EXIT_TRACKING_DOWN
        print("ENABLE_MLFLOW is false: local evaluation JSON remains the record; nothing logged.", file=sys.stderr)
        return EXIT_INVALID if invalid else EXIT_OK

    logged = skipped = 0
    manifest_file = args.manifest if args.manifest.is_file() else None
    for path, evaluation in loaded:
        payload = build_run_payload(evaluation, manifest)
        try:
            if already_logged(client, payload["experiment"], str(evaluation["id"])):
                skipped += 1
                print(f"skip {evaluation['id']}: already logged")
                continue
            run_id = log_one(client, payload, [p for p in (path, manifest_file) if p is not None])
        except MlflowException as exc:
            print(f"ERROR logging {evaluation['id']}: {exc}", file=sys.stderr)
            return EXIT_TRACKING_DOWN if enabled else (EXIT_INVALID if invalid else EXIT_OK)
        logged += 1
        print(f"logged {evaluation['id']} -> run {run_id} ({len(payload['metrics'])} measured metric(s))")
    print(f"done: logged={logged} skipped={skipped} invalid={invalid}")
    return EXIT_INVALID if invalid else EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
