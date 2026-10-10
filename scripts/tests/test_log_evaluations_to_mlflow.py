"""Self-tests for scripts/log_evaluations_to_mlflow.py. Run inside the official MLflow image (stdlib unittest):

    make mlflow-selftest
    # = docker run --rm -v "$PWD/scripts:/opt/dd/scripts:ro" -v "$PWD/config:/opt/dd/config:ro" -w /tmp \
    #     ghcr.io/mlflow/mlflow:v3.17.0 python -m unittest discover -s /opt/dd/scripts/tests -v

The evaluation documents below are test fixtures shaped like the API's Evaluation model, not real results.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))

import log_evaluations_to_mlflow as lem  # noqa: E402

MANIFEST = SCRIPTS.parent / "config" / "models.resolved.yaml"


def fixture_evaluation(eval_id: str = "a" * 32, state: str = "succeeded") -> dict:
    return {
        "id": eval_id,
        "kind": "synthetic_scenario_suite",
        "state": state,
        "created_utc": "2026-10-10T10:00:00Z",
        "finished_utc": "2026-10-10T10:05:00Z",
        "source": "literoad_synthetic_rgb_cv",
        "sample_count": 3,
        "seeds": [1, 2, 3],
        "metrics": {
            "arrival_rate": {"value": 1.0, "unit": "ratio", "sample_count": 3, "status": "measured",
                             "note": "fixture"},
            "lane_f1": {"value": None, "unit": "ratio", "sample_count": 0, "status": "not_evaluated",
                        "note": "no lane labels"},
            "flag_like": {"value": True, "unit": "bool", "sample_count": 1, "status": "measured", "note": "bool"},
        },
        "limitations": ["synthetic LiteRoad domain only"],
        "code_commit": "0" * 40,
        "model_sha256": None,
    }


def run_main(args: list[str], env: dict[str, str]) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with mock.patch.dict(os.environ, env, clear=False), contextlib.redirect_stdout(out), \
            contextlib.redirect_stderr(err):
        code = lem.main(args)
    return code, out.getvalue(), err.getvalue()


class PayloadTests(unittest.TestCase):
    def test_only_measured_numeric_metrics_are_logged(self) -> None:
        payload = lem.build_run_payload(fixture_evaluation(), lem.load_manifest(MANIFEST))
        self.assertEqual(payload["metrics"], {"arrival_rate": 1.0})
        self.assertEqual(payload["tags"]["metric_status.lane_f1"], "not_evaluated: no lane labels")
        self.assertIn("metric_status.flag_like", payload["tags"])  # bool is not a number
        self.assertEqual(payload["status"], "FINISHED")
        self.assertEqual(payload["experiment"], "dreamdrive/synthetic_scenario_suite")

    def test_manifest_identity_comes_from_the_resolved_file(self) -> None:
        manifest = lem.load_manifest(MANIFEST)
        self.assertIsNotNone(manifest)
        params = lem.manifest_params(manifest)
        model = next(m for m in manifest["models"] if m.get("required_for_video"))
        self.assertEqual(params["model.checkpoint_sha256"], str(model["checkpoint_sha256"]))
        self.assertEqual(json.loads(params["model.class_map"]), model["supported_class_map"])

    def test_missing_manifest_is_explicit(self) -> None:
        self.assertEqual(lem.manifest_params(None), {"model.manifest": "missing"})


class MainTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "data" / "evaluations" / ("b" * 32)).mkdir(parents=True)
        ev_dir = self.root / "data" / "evaluations"
        (ev_dir / f"{'a' * 32}.json").write_text(json.dumps(fixture_evaluation()), encoding="utf-8")
        (ev_dir / ("b" * 32) / "evaluation.json").write_text(
            json.dumps(fixture_evaluation("b" * 32, state="failed")), encoding="utf-8")
        (ev_dir / f"{'c' * 32}.json").write_text(json.dumps(fixture_evaluation("c" * 32, state="running")),
                                                 encoding="utf-8")
        self.base = ["--data-root", str(self.root / "data"), "--manifest", str(MANIFEST)]
        self.cwd = os.getcwd()
        os.chdir(self.root)  # file-store artifacts land in ./mlruns under the temp dir

    def tearDown(self) -> None:
        os.chdir(self.cwd)
        self.tmp.cleanup()

    def test_dry_run_lists_terminal_evaluations_only(self) -> None:
        code, out, _ = run_main([*self.base, "--dry-run"], {"ENABLE_MLFLOW": "false"})
        self.assertEqual(code, lem.EXIT_OK)
        self.assertIn("found 2 terminal evaluation(s)", out)
        self.assertIn("skip " + "c" * 32, out)

    def test_invalid_file_fails(self) -> None:
        (self.root / "data" / "evaluations" / "broken.json").write_text("{not json", encoding="utf-8")
        code, _, err = run_main([*self.base, "--dry-run"], {"ENABLE_MLFLOW": "false"})
        self.assertEqual(code, lem.EXIT_INVALID)
        self.assertIn("broken.json", err)

    def test_logs_runs_and_is_idempotent(self) -> None:
        uri = f"sqlite:///{self.root.as_posix()}/mlflow.db"
        code, out, err = run_main([*self.base, "--tracking-uri", uri], {"ENABLE_MLFLOW": "true"})
        self.assertEqual(code, lem.EXIT_OK, err)
        self.assertIn("logged=2 skipped=0", out)

        from mlflow import MlflowClient

        client = MlflowClient(tracking_uri=uri)
        exp = client.get_experiment_by_name("dreamdrive/synthetic_scenario_suite")
        runs = client.search_runs([exp.experiment_id])
        self.assertEqual(len(runs), 2)
        ok = next(r for r in runs if r.data.tags["dreamdrive.eval_id"] == "a" * 32)
        self.assertEqual(ok.data.metrics, {"arrival_rate": 1.0})
        self.assertEqual(ok.info.status, "FINISHED")
        failed = next(r for r in runs if r.data.tags["dreamdrive.eval_id"] == "b" * 32)
        self.assertEqual(failed.info.status, "FAILED")
        artifacts = {a.path for a in client.list_artifacts(ok.info.run_id)}
        self.assertEqual(artifacts, {f"{'a' * 32}.json", "models.resolved.yaml"})

        code, out, _ = run_main([*self.base, "--tracking-uri", uri], {"ENABLE_MLFLOW": "true"})
        self.assertEqual(code, lem.EXIT_OK)
        self.assertIn("logged=0 skipped=2", out)

    def test_outage_is_a_warning_only_when_mlflow_disabled(self) -> None:
        dead = "http://127.0.0.1:9"
        code, _, err = run_main([*self.base, "--tracking-uri", dead], {"ENABLE_MLFLOW": "false"})
        self.assertEqual(code, lem.EXIT_OK)
        self.assertIn("WARNING", err)
        code, _, err = run_main([*self.base, "--tracking-uri", dead], {"ENABLE_MLFLOW": "true"})
        self.assertEqual(code, lem.EXIT_TRACKING_DOWN)
        self.assertIn("ERROR", err)


if __name__ == "__main__":
    unittest.main()
