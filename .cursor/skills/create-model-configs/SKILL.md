---
name: create-model-configs
description: Scaffold models/{model}/ from tester-v1/templates/. Look up HF repo/file for model.yaml (do not download). Tailor server.yaml for 16GB VRAM. Copy sandbox/. Kebab-case dir names; dots OK in version segments.
---

# Create model configs

Scaffold under `models/{model}/`:

| File | Source |
|------|--------|
| `model.yaml` | `tester-v1/templates/model.yaml` |
| `server.yaml` | template base, tuned for 16GB (see below) |
| `llama_bench.yaml` | `tester-v1/templates/llama_bench.yaml` |
| `sandbox/client.yaml`, `sandbox/server.yaml` | `tester-v1/templates/sandbox/` |

Do **not** copy calibration probes into the model dir. Probes live in `tester-v1/calibration-tests/` only (`calibration-tests.mdc`).

## Required inputs

Ask before writing if missing:

| Input | Example |
|-------|---------|
| `model` | `qwen3.5-9b-q8` (directory under `models/`) |
| Model identity | name, quant, architecture (dense / MoE / effective) |

Optional: existing `gguf_path`, or user-supplied `hf_repo` / `hf_file`.

## Model directory name

Lowercase; allowed `a-z`, `0-9`, `-`, `.`. Normalize: lowercase; `_`, `/`, space to `-`; drop other chars; collapse/strip `-`. If ambiguous, derive a name, state it, ask once.

## Workflow

1. Normalize `model`.
2. If any target file exists, show contents; do not overwrite without explicit OK.
3. **HF metadata (lookup only):** find the canonical GGUF on Hugging Face (repo + exact filename). Prefer quantizers already used in this repo (e.g. bartowski, unsloth, ggml-org). Set `hf-download.repo` and `hf-download.file` in `model.yaml`. **Do not** run `./download_model.py`, `hf download`, or other download commands.
4. **`model.yaml`:** set `model:` to the user's existing path if the GGUF is already on disk; otherwise leave the template placeholder `/path/to/model.gguf` (or `/path/to/{file}`). Always include the `hf-download` block when repo/file are known.
5. **`server.yaml`:** start from `tester-v1/templates/server.yaml` (host, port, parallel, `--no-mmap`, `--fit off`). Add flags and `--ctx-size` for a **16GB VRAM** card. Use a similar model under `models/` as the primary reference when one exists:

   | Architecture | Typical extras | `--ctx-size` hint |
   |--------------|------------------|-------------------|
   | Dense, headroom (e.g. 9B Q8) | none | high (see `qwen3.5-9b-q8`: 76800) |
   | Dense, tight (e.g. 27B Q4) | none | low (see `qwen3.6-27b-q4`: 4096) |
   | MoE on 16GB | `--n-gpu-layers 999`, `--n-cpu-moe N` | mid-high (see `qwen3.6-35b-a3b-ud-q4-k-xl`: 66560, `-ncmoe 24`) |
   | Effective / small multimodal | none | high (see `gemma-4-e4b-it-q8`: 128000) |
   | Unknown | none | template default 4096; note calibration will refine |

   Add comment `# safe ctx for 16GB VRAM` above `--ctx-size`. Tune `-ncmoe` / ctx from model size and quant when no close match exists.

6. **`llama_bench.yaml`:** copy template; for MoE, uncomment/set `-ngl 999` and `-ncmoe N` to match `server.yaml`.
7. **`sandbox/`:** always copy `client.yaml` and `server.yaml` from `tester-v1/templates/sandbox/`.
8. Summarize paths and next steps (download command for the user).
9. README row only if the user asks (`models-readme-table.mdc`).

## Safety

- No overwrite without confirmation; no edits under `tester-v1/src/`.
- No GPU runs or `--save-result` unless the user asks (`tester-harness-cli.mdc`).
- New global probes: `tester-v1/calibration-tests/` + `calibration.py` (`calibration-tests-sync.mdc`).

## Commands (repo root; user runs download)

```bash
./download_model.py models/{model}/
./download_model.py models/{model}/ --dry-run
./tester_v1_calibrate.py models/{model}/
./tester_v1_run_test.py models/{model}/ --variant sandbox
./launch_server.py models/{model}/
./llama_bench.py models/{model}/
```

MoE: pass `--n-cpu-moe N` on calibrate, run-test, launch, and llama_bench when not fully baked into YAML. Add `--save-result` only when the user wants JSON under `tester-v1/results/`.

Probe details: [tester-v1/README.md](../../../tester-v1/README.md#calibration).
