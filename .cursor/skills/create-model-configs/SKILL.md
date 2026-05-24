---
name: create-model-configs
description: >-
  Scaffold model.yaml and bench/ under models/ for a new GGUF model.
  Shared probe YAML lives under tester-v1/calibration-tests/ only. Model slugs:
  lowercase kebab-case with dots in version segments (e.g. qwen3.5-9b-q8). Use
  when adding a model, scaffolding configs, or setting up calibration for a new slug.
---

# Create model configs

Write under `models/{model_slug}/`:

- `model.yaml` with the user's GGUF path
- `bench/server.yaml` and `bench/client.yaml` copied verbatim from `tester-v1/calibration-tests/hello-world-baseline/`

Do **not** copy calibration probe dirs into the model slug. Calibration loads probes from `tester-v1/calibration-tests/` in-process. See `calibration-tests.mdc` and [tester-v1/calibration-tests/README.md](../../../tester-v1/calibration-tests/README.md).

## Required inputs

Ask before writing if anything is missing:

| Input | Example |
|-------|---------|
| `model_slug` | `qwen3.5-9b-q8` |
| `gguf_path` | `~/.cache/.../model-Q8_0.gguf` (must exist; expand `~` to verify) |

## Slug normalization

Allowed: `a-z`, `0-9`, `-`, `.` (lowercase only). Apply in order:

1. Lowercase
2. Replace `_`, `/`, whitespace with `-` (keep `.`)
3. Drop other characters
4. Collapse `-`; strip leading/trailing `-`

If the user's name is non-conforming, derive a slug, state it in the summary, and ask once if ambiguous.

## Workflow

1. Collect and normalize `model_slug`; verify `gguf_path` exists.
2. If `model.yaml` or `bench/` exists, show contents and do not overwrite without explicit OK.
3. Write `model.yaml` (`model:` = user path, including `~` if given).
4. Copy current `tester-v1/calibration-tests/hello-world-baseline/{server,client}.yaml` into `bench/`.
5. Summarize paths and commands below.

## Safety

- No overwrite without confirmation; no edits under `tester-v1/src/`.
- Do not run GPU calibration or `--save-result` unless the user asks (`tester-harness-cli.mdc`).
- New global probes: edit `tester-v1/calibration-tests/` and register in `tester-v1/src/calibration.py`; do not auto-sync live `bench/` (`calibration-tests-sync.mdc`).

## Commands (from repo root)

Calibration (shared probes, six-line stdout summary; `--margin-mib` default 100):

```bash
./tester_v1_calibrate.py models/{model_slug}
```

Bench (editable model-local YAML):

```bash
./tester_v1_run_test.py models/{model_slug}/bench
```

Add `--save-result` when JSON on disk is needed (gitignored under `tester-v1/results/`). MoE: pass `--n-cpu-moe N` on the CLI; do not edit calibration probe YAML.

Probe details: [Calibration](../../../tester-v1/README.md#calibration).
