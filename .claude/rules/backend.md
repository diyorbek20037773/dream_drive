---
paths:
  - "backend/**/*.py"
  - "tests/integration/**"
  - "contracts/**"
---
# Backend implementation
Read docs/ARCHITECTURE.md and docs/API.md. Validate all IO with versioned schemas.
No blocking inference in API requests. Persist ownership/state and serialize errors without internals.
Protected artifacts/SSE require ownership. Decode with argument arrays, limits and timeouts.
Use DB migrations and atomic artifact writes. Cancellation and crash cleanup are actual behaviors.
Generate frontend types from final OpenAPI; don't maintain incompatible handwritten copies.
