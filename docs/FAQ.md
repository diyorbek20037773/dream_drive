# Resolved implementation questions

| Question | Answer |
|---|---|
| What are the owner targets? | GitHub diyorbek20037773/dream_drive; Docker Hub namespace diyorbek7377. They differ intentionally. |
| Is the laptop enough? | RTX 4050 6 GB can be benchmarked for batch-one perception. No measured FPS/VRAM claim yet. CPU mode is required. |
| Must CARLA run? | No default dependency. LiteRoad is mandatory, CARLA optional because the 0.9.16 VRAM recommendation exceeds this GPU. |
| Where are YOLOPv2 weights? | Observed upstream README link is recorded in config/models.yaml. Hash/classes/runtime compatibility still require the actual bytes. |
| Can all listed object categories be claimed? | Only classes verified in the exact checkpoint. An additional licensed detector is conditional, not blindly enabled. |
| Which LLM drives the car? | None. FSM/controller/supervisor use structured observations. Explanations are reason-code templates. |
| Can uploaded video react to steering? | No; only the live simulator changes its next rendered image. |
| Is synthetic CV fake inference? | It is real pixel processing in a bounded synthetic domain and labeled explicitly. It is not proof of learned real-road autonomy. |
| Does Docker need a GPU? | CPU image does not. CUDA profile requires compatible NVIDIA driver/toolkit and actual support testing. |
| Must ARM computers work? | Do not claim ARM until built and true inference tested. Initial contract is linux/amd64, including typical Windows WSL2 x86 laptops. |
| Must models be uploaded to git? | No. Verified bootstrap/cache or licensed image layer; model hash/revision in source. |
| Is training mandatory? | No. Pretrained baseline/evaluation/manifests/MLflow first; training only if data and measured need exist. |
| Are Kubernetes, DVC, n8n required? | Kubernetes/n8n not in MVP. DVC conditional on real licensed data and remote. |
| Are all package versions already known? | No. Pin working compatible versions through real dependency resolution; never invent locks from guesses. |
| How to publish before CI has a commit? | Local QA → push commit → CI for same SHA → release/publish → registry pull smoke. |
| What if authentication is missing? | Finish independent implementation, then one specific CLI login/secret action. No secrets in prompts. |
| Where is the server? | Not purchased/provisioned. Prepare deploy now; actual host/domain/access supplied by owner later. |
| Can an agent buy a server? | No budget or provider choice was authorized. Present measured needs and owner-selected target first. |
| What is a successful deploy? | Same tested digests on real host, healthy public HTTPS plus upload/browser smoke; file generation alone is not deployment. |
| What does the source deadline mean? | 2026-10-11 13:00 Asia/Tashkent was in the supplied brief; verify relevance before submission. |
| What can remain null? | Checkpoint hash/classes until resolved, unsupported accuracy metrics, unavailable TTC, unprovided server parameters. None are successful results. |

Read the corresponding long spec before implementing. Empirical observations (throughput, compatibility, model taxonomy, digests) cannot be obtained by guessing from documentation.
