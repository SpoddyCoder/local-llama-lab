---
name: create-new-model-test
description: >-
  Scaffold four calibration and benchmark variant directories (calibration-footprint,
  calibration-ctx-probe, baseline, no-mmap) under tester-v1/test-configs for a new GGUF
  model. Model directories use lowercase kebab-case slugs (e.g. qwen3-5-9b-q8). Use when
  the user adds a new model, asks to scaffold model tests, create test-configs, set up
  calibration variants, or scaffold from the reference templates at
  reference/reference-model.
---

# Create new model test configs

Create `server.yaml` and `client.yaml` for four variants under `tester-v1/test-configs/{test_name}/{model_slug}/`. Copy structure and comments from the reference templates; change only the `model:` path to the user's GGUF.

## Required inputs

Ask for any missing value before writing files:

| Input | Example | Notes |
|-------|---------|-------|
| `test_name` | `test1` | First segment under `test-configs/` |
| `model_slug` | `qwen3-5-9b-q8` | Directory name for this model (see naming below) |
| `gguf_path` | `~/.cache/.../model-Q8_0.gguf` | Absolute or `~`; must exist on disk |

Expand `~` when checking existence. Do not invent paths.

## Model directory naming (`model_slug`)

All model directories under `test-configs/{test_name}/` use **lowercase kebab-case**:

- Characters: `a-z`, `0-9`, and `-` only
- Lowercase only; no underscores, dots, spaces, or uppercase
- Words/segments separated by a single hyphen; collapse repeated hyphens
- Keep it short but distinct: family, size, and quant when helpful (e.g. `qwen3-5-9b-q8`, `gemma-4-4b-it-q8`)

**Normalize** before creating directories (apply in order):

1. Lowercase the string
2. Replace `_`, `.`, `/`, and whitespace runs with `-`
3. Remove characters that are not `a-z`, `0-9`, or `-`
4. Collapse consecutive `-`; strip leading/trailing `-`

If the user gives a non-conforming name (e.g. `Qwen3.5-9B-Q8`, `gemma_4_e4b`), derive the slug with the rules above (often from their label or the GGUF basename) and **state the chosen `model_slug` in the summary** before writing files. If ambiguous, ask once; do not invent unrelated names.

**Always create new models** with a normalized kebab-case `model_slug`.

## Reference templates

The canonical template source is `tester-v1/test-configs/reference/reference-model/`. Enumerate its subdirectories; each variant dir must contain `server.yaml` and `client.yaml`:

- `calibration-footprint/`
- `calibration-ctx-probe/`
- `baseline/`
- `no-mmap/`

Keep all comments and YAML structure. Replace only the `model:` value with `gguf_path` (use the path the user gave, including `~` if they provided it that way).

## Variant summary

| Variant | `server.yaml` | `client.yaml` |
|---------|---------------|---------------|
| `calibration-footprint/` | `model` + args: `--host 127.0.0.1`, `--port 8080`, `--fit off`, `-c 4096`, `--parallel 1` | `user: "ok"`, `max_tokens: 1`, `temperature: 0` |
| `calibration-ctx-probe/` | Same as calibration-footprint but `-c 16384` | Same as calibration-footprint |
| `baseline/` | `model` + host/port only (no `--fit off`) | Test 1 prompt from qwen baseline |
| `no-mmap/` | baseline server + `--no-mmap` | Same as baseline client |

## Directory layout

```text
tester-v1/test-configs/
  {test_name}/
    {model_slug}/
      calibration-footprint/
        server.yaml
        client.yaml
      calibration-ctx-probe/
        server.yaml
        client.yaml
      baseline/
        server.yaml
        client.yaml
      no-mmap/
        server.yaml
        client.yaml
```

Target root: `tester-v1/test-configs/{test_name}/{model_slug}/`

## Workflow

1. **Collect inputs** — `test_name`, `model_slug` (or a label to derive it), `gguf_path`. Stop and ask if any are missing. Normalize `model_slug` to lowercase kebab-case before any path checks or writes.
2. **Verify GGUF** — Resolve `~` if needed; confirm the file exists. Do not proceed if missing.
3. **Check target tree** — List subdirs under `reference/reference-model/` (each must have `server.yaml` + `client.yaml`); list the same variant paths that would be created under `{test_name}/{model_slug}/` (two YAML files per variant).
4. **Collision handling**
   - If nothing exists under `{test_name}/{model_slug}/`, create all four variants.
   - If any variant dir or YAML already exists, show what is present and **do not overwrite** without explicit user confirmation.
   - On confirmed overwrite, only replace files the user named; prefer creating missing variants only.
5. **Read templates** — Load reference YAML from each subdir of `tester-v1/test-configs/reference/reference-model/` (`server.yaml` and `client.yaml` per variant).
6. **Write files** — For each variant, write `server.yaml` and `client.yaml` with the same content as the template except `model:` set to `gguf_path`.
7. **Summarize** — List created paths and remind the user of the calibration command (below).

## Safety rules

- Never overwrite existing configs without explicit user confirmation.
- Do not modify `tester-v1/src/calibration.py` or `tester-v1/src/model_calibration.py`.
- Do not run live GPU calibration (`model_calibration.py` or `python_runner.py` against real hardware) unless the user explicitly asks to run tests.
- Do not change template comments, args, or client prompts except the `model:` line.
- When adding a new standard variant, update `tester-v1/test-configs/reference/reference-model/` and the variant table in this skill in the same change.

## After scaffolding

Suggest running calibration from `tester-v1/`:

```bash
python3 src/model_calibration.py test-configs/{test_name}/{model_slug}
```

For what the probes mean and how to interpret stdout, see [Model calibration](../../../tester-v1/README.md#model-calibration) in `tester-v1/README.md`.

## Example

User: scaffold tests for `test1` with GGUF `~/models/My_Model-Q4_0.gguf` (they said "my model q4")

1. Normalize slug → `my-model-q4` (from user words or basename `my-model-q4-0` if basename is clearer).
2. Verify the GGUF exists.
3. Confirm `tester-v1/test-configs/test1/my-model-q4/` is absent (or get overwrite OK).
4. Create YAML files from `reference/reference-model/` templates with `model: ~/models/My_Model-Q4_0.gguf` (preserve the user's path in YAML; only the directory name is kebab-case).
5. Tell user: `python3 src/model_calibration.py test-configs/test1/my-model-q4` from `tester-v1/`.
