# LLM-output fixture corpus

Representative LLM-output fixtures for each agent schema. Loaded by
`tests/test_fixture_replay.py` and validated against the corresponding
Pydantic schema in `graph/state_schemas.py` to catch schema-vs-LLM drift
before deploys.

## How this catches drift

If a schema change tightens a constraint that real Anthropic outputs
already violate (or relaxes one in a backward-incompatible direction),
the fixture replay test fails in CI before the PR merges. Surface area:

- Required-field additions
- Numeric-bound tightening (`Field(ge=, le=)`)
- Literal value removals
- Type changes (e.g., `str` → `int`)

Fixtures are intentionally **minimal-conformant** representatives — just
enough fields to validate the schema's required surface. Schema
relaxations (adding optional fields, widening literals) won't break
fixtures, only tightenings will. That's the desired drift-detection
posture.

## Adding a new fixture

1. Pick a real captured artifact from
   `s3://alpha-engine-research/decision_artifacts/{Y}/{M}/{D}/{agent_id}/{run_id}.json`
   that you trust as canonical.
2. Reduce it to a minimal-conformant shape (drop optional extras, anonymize
   ticker/dates if needed).
3. Save as `tests/fixtures/llm_outputs/<schema_name>.json` using
   snake_case of the schema class name.
4. Add a row to `_FIXTURES` in `tests/test_fixture_replay.py` mapping
   the file to its schema.

## Refreshing from real captured artifacts

Run from repo root:

```bash
python scripts/refresh_llm_fixtures.py --since 2026-04-30
```

Downloads the latest captured artifacts and updates the fixtures.
Review diffs before committing — the refresh shouldn't change shapes,
only field values; structural changes are a signal that the schema or
prompt is drifting.

## Why these fixtures aren't from `local/offline_stubs.py`

The synthetic stubs produce schema-conformant outputs by definition
(we wrote them after the schemas were fixed) — using them as fixtures
would tautologically pass the schema check. The fixture corpus must
represent **real Anthropic output shapes** to catch drift between
schema expectations and actual LLM behavior. Refresh from S3
`decision_artifacts/` after each Saturday SF to keep them current.
