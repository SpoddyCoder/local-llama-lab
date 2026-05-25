---
name: create-model-configs
description: >-
  Scaffold model.yaml, server.yaml, and llama_bench.yaml under models/ for a
  new GGUF model from tester-v1/templates/. Shared calibration probes live
  under tester-v1/calibration-tests/ only. Model directory names: lowercase
  kebab-case with dots in version segments (e.g. qwen3.5-9b-q8). Use when
  adding a model, scaffolding configs, or setting up calibration.
---

# Create model configs

Write under `models/{model}/`:

- `model.yaml` with the user's GGUF path (from `tester-v1/templates/model.yaml`)
- `server.yaml` copied from `tester-v1/templates/server.yaml`
- `llama_bench.yaml` copied from `tester-v1/templates/llama_bench.yaml`

Do **not** copy calibration probe YAML into the model directory. Calibration loads probes from `tester-v1/calibration-tests/` in-process. See `calibration-tests.mdc` and [tester-v1/calibration-tests/README.md](../../../tester-v1/calibration-tests/README.md).

## Required inputs

Ask before writing if anything is missing:

| Input | Example |
|-------|---------|
| `model` | `qwen3.5-9b-q8` (directory name under `models/`) |
| `gguf_path` | `~/.cache/.../model-Q8_0.gguf` (must exist; expand `~` to verify) |

## Optional inputs

| Input | Example |
|-------|---------|
| `hf_repo` | `bartowski/Qwen_Qwen3.5-9B-GGUF` |
| `hf_file` | `Qwen_Qwen3.5-9B-Q8_0.gguf` |

When both are known, set the `hf-download` block in `model.yaml` so `./download_model.py` can fetch the GGUF without manual `hf download` commands.

## Model directory name

Allowed: `a-z`, `0-9`, `-`, `.` (lowercase only). Apply in order:

1. Lowercase
2. Replace `_`, `/`, whitespace with `-` (keep `.`)
3. Drop other characters
4. Collapse `-`; strip leading/trailing `-`

If the user's name is non-conforming, derive a directory name, state it in the summary, and ask once if ambiguous.

## Workflow

1. Collect and normalize `model`; verify `gguf_path` exists.
2. If `model.yaml`, `server.yaml`, or `llama_bench.yaml` exists, show contents and do not overwrite without explicit OK.
3. Copy `tester-v1/templates/model.yaml` to `models/{model}/model.yaml`; set `model:` to the user's path (including `~` if given). When `hf_repo` and `hf_file` are known, uncomment or set the `hf-download` block (`repo`, `file`).
4. Copy `tester-v1/templates/server.yaml` to `models/{model}/server.yaml`.
5. Copy `tester-v1/templates/llama_bench.yaml` to `models/{model}/llama_bench.yaml`.
6. Summarize paths and commands below.
7. If the user wants the model in the README table, add or update its row in [README.md](../../README.md) and sort rows by LiveCodeBench v6 score descending (`models-readme-table.mdc`).

## Sandbox (optional)

For ad-hoc prompts, copy or adapt `tester-v1/templates/sandbox/` to `models/{model}/sandbox/` (`client.yaml` required; `server.yaml` optional thin override). Run with `--variant sandbox`.

## MoE models

Pass `--n-cpu-moe N` on the CLI for calibration and run-test. Optionally bake MoE flags into that model's `server.yaml` when they are stable.

For `llama_bench.yaml`, uncomment or add `-ngl 999` and `-ncmoe N` (or tune per model). Pass `--n-cpu-moe N` to `./llama_bench.py` to override `-ncmoe` at run time.

## Safety

- No overwrite without confirmation; no edits under `tester-v1/src/`.
- Templates live under `tester-v1/templates/`; do not run GPU calibration or `--save-result` unless the user asks (`tester-harness-cli.mdc`).
- New global probes: edit `tester-v1/calibration-tests/` and register in `tester-v1/src/calibration.py` (`calibration-tests-sync.mdc`).
- `llama_bench.yaml` is required for `./llama_bench.py`; the CLI exits with an error if it is missing.

## Commands (from repo root)

Calibration (shared probes, six-line stdout summary; `--margin-mib` default 100):

```bash
./tester_v1_calibrate.py models/{model}
```

Run test (default variant `hello-world-baseline` from calibration-tests):

```bash
./tester_v1_run_test.py models/{model}/
./tester_v1_run_test.py models/{model}/ --variant hello-world-baseline
```

Download GGUF (optional `hf-download` in `model.yaml`):

```bash
./download_model.py models/{model}/
./download_model.py models/{model}/ --dry-run
```

Launch server (model root `server.yaml`):

```bash
./launch_server.py models/{model}/
```

llama-bench (model root `llama_bench.yaml`; uses `model.yaml` for the GGUF path):

```bash
./llama_bench.py models/{model}/
./llama_bench.py models/{model}/ --n-cpu-moe N
./llama_bench.py models/{model}/ --dry-run
```

Add `--save-result` when JSON on disk is needed (gitignored under `tester-v1/results/`).

Probe details: [Calibration](../../../tester-v1/README.md#calibration).
