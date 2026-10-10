# Model and data operations

MLOps starts with reproducible pretrained evaluation. Default inference is not a training pipeline. Record and compare model/data/config changes before promoting them.

## Required identity
Each analysis/evaluation report includes code commit, model ID/revision/checkpoint SHA256, class map, preprocessing/postprocessing identity, dependency locks/image digest, device/precision, input/label manifest hashes, seed, timestamps and measured timing/sample counts. Uploads have their own SHA256 and private ownership; public evaluation data includes documented rights. Preserve code license notices separately from weight and dataset permissions.

Use MLflow for evaluation experiments and artifacts/parameters/metrics. Its optional service uses a database-backed store (single-node SQLite is acceptable) and persistent artifacts. Do not send MLflow traffic in the real-time control path. Export local evaluation JSON first; tracking outage is a warning, not a driving or upload failure. Keep service private and document authentication for shared access. Register/promote only tested compatible versions; simpler resolved manifest selection is sufficient for the MVP.

## Evaluation execution
Implement a command that runs the frozen labeled evaluation subset, logs raw predictions/derived metrics and reports sample coverage. An MLflow run stores the complete resolved model/data config and benchmark inputs. Ablations: detector only, +tracking, +decision, +supervisor with identical dataset/seeds and explicit unsupported comparisons. Don't compute synthetic-CV vs learned-model accuracy on incompatible label definitions as if it were one leaderboard.

Regression gating: compare to a recorded, genuinely run baseline on the same manifest/hardware assumptions. Define meaningful tolerances after collecting the first baseline; do not invent a 95% pass threshold from upstream results. Release fails on schema/mask integrity errors and required synthetic safety behavior; domain accuracy degradation is reported with sample counts and uncertainty.

## DVC and training are conditional
Use DVC only if a licensed evaluation/training corpus and usable remote storage are supplied. Commit small manifests/labels/`.dvc` references; put credentials in local config or environment, never git. No fake remote bucket and no untracked dataset dependence. A tiny authorized fixture plus hash manifest is adequate initially. Do not force users to download BDD100K to run the app.

If fine-tuning is needed later: frozen splits by clip/session, seed/config, training command, VRAM checks, checkpoint lineage, validation separate from final test, MLflow runs, data version, rollback and model card. No scheduled retraining merely because incoming user-uploaded videos exist; those inputs are unlabeled and may lack training consent. No unsupported TorchScript training claim.

## Runtime monitoring limits
Track brightness/blur/input dimensions, no-detection rate, lane validity, confidence distribution, fallback frequency and latency. They are data-quality/domain-shift indicators, not online model accuracy. True drift/accuracy needs labeled observations and a defined baseline. Show “not evaluated” without labels. Record failures and unknown classes rather than hiding them.
