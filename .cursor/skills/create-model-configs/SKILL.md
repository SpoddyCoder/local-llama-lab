---
name: create-model-configs
description: >-
  Scaffold all reference variant directories under tester-v1/configs for a new
  GGUF model (calibration-footprint, calibration-ctx-probe, hello-world-baseline
  as of this change; dynamic thereafter). Model directories use lowercase
  kebab-case slugs with dots allowed for point versions (e.g. qwen3.5-9b-q8). Use
  when the user adds a new model, asks to scaffold model configs, create configs,
  set up calibration variants, or scaffold from the reference templates at
  configs/reference/.
---

# Create model configs

Create `server.yaml` and `client.yaml` for every reference variant under `tester-v1/configs/{model_slug}/{variant}/`. Copy structure and comments from the reference templates; change only the `model:` path to the user's GGUF.

## Core rule

Every variant subdirectory under `tester-v1/configs/reference/` is a global template. When scaffolding a new model, copy **all** of them (not a fixed list). If the user adds a new dir under `reference/` with both YAML files, it is meant for every model.

Model-only variants (for example `hello-world-bench` on one model) are added manually under that model's slug; they do not live in `reference/` unless promoted there.

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

The canonical template source is `tester-v1/configs/reference/`. List `reference/*/` and include only subdirs that contain both `server.yaml` and `client.yaml`. Do not copy from live model dirs (`qwen3.5-9b-q8/`, etc.).

Reference `server.yaml` files use block-scalar `args: |` (one `llama-server` flag per line). Copy structure verbatim; replace only the `model:` value with `gguf_path` (use the path the user gave, including `~` if they provided it that way). Do not use YAML list format for `args`.

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
  {model_slug}/
    calibration-footprint/
      server.yaml
      client.yaml
    calibration-ctx-probe/
      server.yaml
      client.yaml
    hello-world-baseline/
      server.yaml
      client.yaml
```

Target root: `tester-v1/configs/{model_slug}/`

## Workflow

1. **Collect inputs** — `model_slug` (or a label to derive it), `gguf_path`. Stop and ask if any are missing. Normalize `model_slug` before any path checks or writes.
2. **Verify GGUF** — Resolve `~` if needed; confirm the file exists. Do not proceed if missing.
3. **List reference variants** — Subdirs under `configs/reference/` with both `server.yaml` and `client.yaml`; list the same variant paths that would be created under `{model_slug}/` (two YAML files per variant).
4. **Collision handling**
   - If nothing exists under `{model_slug}/`, create one dir per reference variant.
   - If any variant dir or YAML already exists, show what is present and **do not overwrite** without explicit user confirmation.
   - On confirmed overwrite, only replace files the user named; prefer creating missing variants only.
5. **Read templates** — Load reference YAML from each qualifying subdir of `tester-v1/configs/reference/`.
6. **Write files** — For each variant, write `server.yaml` and `client.yaml` with the same content as the template except `model:` set to `gguf_path`.
7. **Summarize** — List created paths and remind the user of the calibration command (below).

## Safety rules

- Never overwrite existing configs without explicit user confirmation.
- Do not modify harness Python under `tester-v1/src/`.
- Do not run live GPU calibration (`model_calibration.py` or `single_test_runner.py` against real hardware) unless the user explicitly asks to run tests.
- Do not change template comments, args, or client prompts except the `model:` line.
- When adding a new standard variant, update `tester-v1/configs/reference/` and the variant table in this skill in the same change.

## After scaffolding

Suggest running calibration from `tester-v1/`:

```bash
python3 model_calibration.py configs/{model_slug}
```

Default: probe stdout for each calibration variant, then the four-line VRAM summary (no JSON). Add `--save-result` when probe JSON under `results/{model_slug}/{variant}/` is needed; calibration also writes `results/{model_slug}/calibration-sessions/{session_id}.json`.

For what the probes mean and how to interpret stdout, see [Model calibration](../../../tester-v1/README.md#model-calibration) in `tester-v1/README.md`.

## Example

User: scaffold configs for GGUF `~/models/My_Model-Q4_0.gguf` (they said "my model q4")

1. Normalize slug → `my-model-q4` (from user words or basename if clearer).
2. Verify the GGUF exists.
3. Confirm `tester-v1/configs/my-model-q4/` is absent (or get overwrite OK).
4. Create YAML files from all `configs/reference/` variants with `model: ~/models/My_Model-Q4_0.gguf` (preserve the user's path in YAML; only the directory name is normalized).
5. Tell user: `python3 model_calibration.py configs/my-model-q4` from `tester-v1/`.
