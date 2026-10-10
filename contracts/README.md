# Contract implementation notes

Schemas are handoff contracts, not generated application OpenAPI. Validate them with Draft 2020-12 plus format checking. Generate matching Pydantic models and contract tests, and derive frontend types from actual OpenAPI.

AdditionalProperties rejects dynamic simulator truth in world state. Add semantic validation for xyxy ordering/image bounds, mask dimensions, class-map consistency, mutually exclusive throttle/brake, monotonic timestamps and mode/action kind. Schema bounds alone do not prove these conditions.

Model identity is required when source is learned_model. Analytical video has no ego localization or actuator command. Non-measured metrics are null with a reason; measured metrics need positive sample counts. A null model identity is valid for synthetic-only CV. Evaluation truth may be used for evaluation; runtime dynamic-truth field must remain false.

The SSE envelope deliberately permits event-specific payloads until Claude implements the discriminated variants; it does not excuse schema-free application payloads. Source/evaluation labels are required in UI/reports.

Schema $id values are local identifiers, not claims that a website exists. No public dreamdrive.local deployment is intended.
