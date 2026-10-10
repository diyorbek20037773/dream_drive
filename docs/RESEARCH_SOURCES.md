# Official technical sources and research provenance

Checked 2026-10-10. Links below were opened through research tools except the target repository's web view, which failed. These pages support tool behavior, not DreamDrive performance or completed implementation. Resolve exact versions again when coding; prefer versioned docs to moving latest pages.

| Source | Finding used in this kit |
|---|---|
| https://code.claude.com/docs/en/memory | Short root CLAUDE.md and path-scoped `.claude/rules` support modular instructions; prose is guidance, not a hard security boundary |
| https://code.claude.com/docs/en/settings | Project settings/permission controls; no unrestricted permissions required |
| https://github.com/CAIC-AD/YOLOPv2 | Panoptic road tasks, BDD100K provenance, MIT repository license; upstream performance is not laptop performance |
| https://github.com/CAIC-AD/YOLOPv2/blob/main/demo.py | TorchScript loading, custom head decoding and lane/drivable helpers; class taxonomy not assumed |
| https://github.com/FoundationVision/ByteTrack | Tracker associates high/low-confidence detections; repository MIT |
| https://github.com/roboflow/rf-detr | Nano listed among Apache-designated models; other Plus components have separate licensing |
| https://carla.readthedocs.io/en/0.9.16/start_quickstart/ | Packaged version recommends 2070-equivalent and at least 8 GB VRAM; separate from UE5 |
| https://carla-ue5.readthedocs.io/en/latest/start_quickstart/ | UE5 heavier hardware requirements; not the default for 6 GB laptop |
| https://carla.readthedocs.io/en/0.9.16/adv_synchrony_timestep/ | Fixed timestep/synchronous mode and one ticking client; simulated and wall time differ |
| https://carla.readthedocs.io/en/latest/build_docker/ | Version-specific CARLA Docker/NVIDIA/offscreen setup; dev docs must be checked against pinned version |
| https://docs.docker.com/desktop/features/wsl/ | WSL2 integration for Linux workflows on Windows |
| https://docs.docker.com/compose/how-tos/gpu-support/ | Explicit device reservation and NVIDIA requirements for GPU Compose |
| https://docs.docker.com/build/ci/github-actions/multi-platform/ | Buildx CI options; architecture support still needs actual tests |
| https://docs.github.com/en/actions/reference/security/secure-use | Least privileges and pinned actions; credentials excluded from untrusted PR execution |
| https://docs.github.com/en/actions/tutorials/publish-packages/publish-docker-images | Registry login/build/publish CI structure |
| https://mlflow.org/docs/latest/ml/tracking/ | Track experiment parameters/metrics/artifacts |
| https://mlflow.org/docs/latest/self-hosting/architecture/backend-store/ | Database-backed tracking persistence; deployed exact version needs compatibility check |
| https://prometheus.io/docs/practices/naming/ | Metrics with units and bounded labels |
| https://grafana.com/docs/grafana/latest/alerting/ | Provision/test useful alerts; no destination assumed |
| https://caddyserver.com/docs/automatic-https | Automatic HTTPS depends on actual domain/connectivity/server conditions |
| https://doc.dvc.org/user-guide/data-management/remote-storage | DVC remotes for real data-versioning needs; no fake storage target |

Target: https://github.com/diyorbek20037773/dream_drive . Git remote probes returned no branch refs at research time; re-audit locally before writing/pushing. The upstream YOLOPv2 repository was subsequently cloned read-only to verify its README checkpoint URL and source commit `9388c6aa55f3bf5a878679a1338bb090cff6a256`. No DreamDrive app checkout, model loading or app tests were conducted by the handoff author.

Source Word file: New Microsoft Word Document.docx, supplied by owner; extracted full text is `docs/source/master-brief.md`. Hardware/Docker Hub/server answers were supplied in this conversation. All application architecture, default quotas, sizing and phase choices are engineering recommendations, not empirical DreamDrive results. No weight checksum, specific release image digest, benchmark or class map was fabricated.

Additional official implementation references checked:
- https://fastapi.tiangolo.com/deployment/docker/ — building explicit container runtimes.
- https://playwright.dev/docs/docker — browser image/version dependencies for E2E tests.
- https://python-rq.org/docs/ — actual RQ queue/job/worker behavior.
- https://trivy.dev/docs/latest/target/container_image/ — container scanning support.
- https://docs.astral.sh/uv/guides/integration/docker/ — optional frozen Python dependency workflow.
