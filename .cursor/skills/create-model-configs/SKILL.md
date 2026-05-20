---
name: create-model-configs
description: >-
  Scaffold model.yaml under tester-v1/configs for a new GGUF model. Shared probe
  YAML lives under configs/reference/ only. Model directories use lowercase
  kebab-case slugs with dots allowed for point versions (e.g. qwen3.5-9b-q8). Use
  when the user adds a new model, asks to scaffold model configs, create configs,
  or set up calibration for a new slug.
---

# Create model configs

Create `model.yaml` at `tester-v1/configs/{model_slug}/` with the user's GGUF path. Shared calibration and hello-world probe YAML live under `configs/reference/{variant}/` only; do not copy variant dirs into live model folders.

## Core rule

Every variant subdirectory under `tester-v1/configs/reference/` defines shared probe templates used by `model_calibration.py` and by manual runs via `--server` / `--client`. When scaffolding a new model, write **only** `{model_slug}/model.yaml`.

Model-only variants (for example `hello-world-bench` on one model) are added manually under that model's slug when needed; they do not live in `reference/` unless promoted there.

## Required inputs

Ask for any missing value before writing files:

| Input | Example | Notes |
|-------|---------|-------|
| `model_slug` | `qwen3.5-9b-q8` | Directory name for this model (see naming below) |
| `gguf_path` | `~/.cache/.../model-Q8_0.gguf` | Absolute or `~`; must exist on disk |

Expand `~` when checking existence. Do not invent paths.

## Model directory naming (`model_slug`)

All model directories under `configs/` use **lowercase kebab-case** with optional **dots in version segments**:

- Characters: `a-z`, `0-9`, `-`, and `.` only
- Lowercase only; no underscores, spaces, or uppercase
- Words/segments separated by a single hyphen; collapse repeated hyphens
- Preserve intentional `.` in version numbers (e.g. `qwen3.5-9b-q8`, not `qwen3-5-9b-q8`)
- Keep it short but distinct: family, size, and quant when helpful

**Normalize** before creating directories (apply in order):

1. Lowercase the string
2. Replace `_`, `/`, and whitespace runs with `-` (do **not** replace `.`)
3. Remove characters that are not `a-z`, `0-9`, `-`, or `.`
4. Collapse consecutive `-`; strip leading/trailing `-`

If the user gives a non-conforming name (e.g. `Qwen3.5-9B-Q8`, `gemma_4_e4b`), derive the slug with the rules above (often from their label or the GGUF basename) and **state the chosen `model_slug` in the summary** before writing files. If ambiguous, ask once; do not invent unrelated names.

**Always create new models** with a normalized slug.

## Reference templates

Canonical shared YAML is under `tester-v1/configs/reference/`. List `reference/*/` with both `server.yaml` and `client.yaml` when explaining what calibration will run. Do not copy those files into `{model_slug}/`.

## Variant summary

Illustrative; enumeration under `configs/reference/` is authoritative.

| Variant | Purpose (brief) |
|---------|-----------------|
| `calibration-footprint/` | VRAM cal, `-c 4096`, `--fit off` |
| `calibration-ctx-probe/` | VRAM cal, `-c 16384`, `--fit off` |
| `hello-world-baseline/` | Smoke probe, default server flags |

## Directory layout

```text
tester-v1/configs/
  reference/
    calibration-footprint/
      server.yaml
      client.yaml
    ...
  {model_slug}/
    model.yaml
```

Target root: `tester-v1/configs/{model_slug}/`

## Workflow

1. **Collect inputs** — `model_slug` (or a label to derive it), `gguf_path`. Stop and ask if any are missing. Normalize `model_slug` before any path checks or writes.
2. **Verify GGUF** — Resolve `~` if needed; confirm the file exists. Do not proceed if missing.
3. **Collision handling**
   - If `{model_slug}/model.yaml` is absent, create it.
   - If `model.yaml` already exists, show contents and **do not overwrite** without explicit user confirmation.
4. **Write file** — Write `{model_slug}/model.yaml` with `model:` set to `gguf_path` (use the path the user gave, including `~` if they provided it that way).
5. **Summarize** — List the created path and remind the user of the calibration command (below).

## Safety rules

- Never overwrite existing configs without explicit user confirmation.
- Do not modify harness Python under `tester-v1/src/`.
- Do not run live GPU calibration (`model_calibration.py` or `single_test_runner.py` against real hardware) unless the user explicitly asks to run tests.
- Do not copy or edit reference variant YAML when scaffolding; only set `model:` in live `model.yaml`.
- When adding a new standard variant, update `tester-v1/configs/reference/` and the variant table in this skill in the same change.

## After scaffolding

Suggest running calibration from `tester-v1/`:

```bash
python3 model_calibration.py configs/{model_slug}
```

Default: probe stdout for footprint, ctx-probe, and hello-world-baseline (YAML from `configs/reference/`), then a six-line summary (throughput plus VRAM/context; no JSON). A 100 MiB VRAM safety buffer is subtracted from the KV budget (`--margin-mib` to change). Add `--save-result` when probe JSON under `results/{model_slug}/{variant}/` is needed on disk (gitignored); calibration also writes `results/{model_slug}/calibration-sessions/{session_id}.json`. Copy summary lines into the repo README when documenting the model.

For what the probes mean and how to interpret stdout, see [Model calibration](../../../tester-v1/README.md#model-calibration) in `tester-v1/README.md`.

## Example

User: scaffold configs for GGUF `~/models/My_Model-Q4_0.gguf` (they said "my model q4")

1. Normalize slug → `my-model-q4` (from user words or basename if clearer).
2. Verify the GGUF exists.
3. Confirm `tester-v1/configs/my-model-q4/model.yaml` is absent (or get overwrite OK).
4. Create `model.yaml` with `model: ~/models/My_Model-Q4_0.gguf`.
5. Tell user: `python3 model_calibration.py configs/my-model-q4` from `tester-v1/`.
