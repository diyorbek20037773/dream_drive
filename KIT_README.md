# DreamDrive implementation handoff kit

Owner-facing entry: BOSHLASH_UZ.md. Agent entry: CLAUDE.md and PROMPT_FOR_CLAUDE_CODE.md.
This is a specification package, not an implemented application or a successful release.
Prepared on 2026-10-10 from the attached Word brief and the owner's three answers.

## Contents
- `docs/`: product, architecture, interfaces, pixel/control separation, design, QA, delivery and research.
- `.claude/rules/`: scoped backend, ML, UI and deployment instructions.
- `.claude/settings.json`: narrow secret-read restrictions, no automatic permission bypass.
- `config/`: project choices, candidate model manifest with verified upstream source link and simulation defaults.
- `contracts/`: versioned result/event contract schemas to implement and validate.
- `.env.example`: nonsecret configuration contract; secrets remain blank.
- `docs/source/master-brief.md`: text extracted from the supplied Word file, preserved as context.
- `HANDOFF_VALIDATION.md`: checks of this package, separate from application QA.

Architecture changes from the brief: LiteRoad default for RTX 4050 6 GB; CARLA optional. Synthetic RGB CV is explicit and distinct from learned real-video inference. Real checkpoint classes must be verified. This avoids pretending a road model already generalizes to procedural images.

The handoff author did not train models, run app QA, push code, publish images or deploy a server. Those are implementation tasks assigned to Claude Code. No credentials or paid infrastructure are included.
