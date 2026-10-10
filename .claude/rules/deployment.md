---
paths:
  - "deploy/**"
  - "ops/**"
  - ".github/workflows/**"
  - "compose*.yaml"
  - "**/Dockerfile*"
  - "scripts/**"
---
# Delivery rules
Read docs/DEVOPS.md, docs/QA.md and docs/OBSERVABILITY.md.
CPU default must run without host Python/Node/GPU/source mounts. GPU and ARM require their own support evidence.
Secrets never enter git/images/browser/logs. Pin verified actions/digests, no fabricated SHA values.
Publish only after required QA for the exact commit. Verify registry pulls and record digests.
Prod uses pull-only images and protected persistent volumes. No volume destruction or blind prune.
No server exists at handoff; prepare files, then deploy when target access is supplied.
