---
name: create-new-model-test
description: >-
  Scaffold four calibration and benchmark variant directories (calibration-footprint,
  calibration-ctx-probe, baseline, no-mmap) under tester-v1/test-configs for a new GGUF
  model. Use when the user adds a new model, asks to scaffold model tests,
  create test-configs, set up calibration variants, or copy the qwen test1
  layout for another model slug.
---

# Create new model test configs

Create `server.yaml` and `client.yaml` for four variants under `tester-v1/test-configs/{test_name}/{model_slug}/`. Copy structure and comments from the reference templates; change only the `model:` path to the user's GGUF.

## Required inputs

Ask for any missing value before writing files:

| Input | Example | Notes |
|-------|---------|-------|
| `test_name` | `test1` | First segment under `test-configs/` |
| `model_slug` | `qwen3.5-9b-q8` | Directory name for this model |
| `gguf_path` | `~/.cache/.../model-Q8_0.gguf` | Absolute or `~`; must exist on disk |

Expand `~` when checking existence. Do not invent paths.

## Reference templates

Read and copy from (repo root relative):

```
tester-v1/test-configs/test1/qwen3.5-9b-q8/
  calibration-footprint/server.yaml
  calibration-footprint/client.yaml
  calibration-ctx-probe/server.yaml
  calibration-ctx-probe/client.yaml
  baseline/server.yaml
  baseline/client.yaml
  no-mmap/server.yaml
  no-mmap/client.yaml
```

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

1. **Collect inputs** — `test_name`, `model_slug`, `gguf_path`. Stop and ask if any are missing.
2. **Verify GGUF** — Resolve `~` if needed; confirm the file exists. Do not proceed if missing.
3. **Check target tree** — List paths that would be created (all eight YAML files under the four variant dirs).
4. **Collision handling**
   - If nothing exists under `{test_name}/{model_slug}/`, create all four variants.
   - If any variant dir or YAML already exists, show what is present and **do not overwrite** without explicit user confirmation.
   - On confirmed overwrite, only replace files the user named; prefer creating missing variants only.
5. **Read templates** — Load the eight reference files from `test1/qwen3.5-9b-q8/`.
6. **Write files** — For each variant, write `server.yaml` and `client.yaml` with the same content as the template except `model:` set to `gguf_path`.
7. **Summarize** — List created paths and remind the user of the calibration command (below).

## Safety rules

- Never overwrite existing configs without explicit user confirmation.
- Do not modify `tester-v1/src/calibration.py` or `tester-v1/src/model_calibration.py`.
- Do not run live GPU calibration (`model_calibration.py` or `python_runner.py` against real hardware) unless the user explicitly asks to run tests.
- Do not change template comments, args, or client prompts except the `model:` line.

## After scaffolding

Suggest running calibration from `tester-v1/`:

```bash
python3 src/model_calibration.py test-configs/{test_name}/{model_slug}
```

For what the probes mean and how to interpret stdout, see [Model calibration](../../../tester-v1/README.md#model-calibration) in `tester-v1/README.md`.

## Example

User: scaffold tests for `test1` / `my-model-q4` with GGUF at `~/models/my-model-q4.gguf`

1. Verify `~/models/my-model-q4.gguf` exists.
2. Confirm `tester-v1/test-configs/test1/my-model-q4/` is absent (or get overwrite OK).
3. Create eight YAML files from `test1/qwen3.5-9b-q8/` templates with `model: ~/models/my-model-q4.gguf`.
4. Tell user: `python3 src/model_calibration.py test-configs/test1/my-model-q4` from `tester-v1/`.
