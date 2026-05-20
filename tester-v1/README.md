# Tester v1

Single-run harness for `llama-server`: start the server, run one streaming chat completion, print all thirteen metrics plus a headline footer on stdout, then tear down. Pass `--save-result` with a config directory to write JSON under `results/{model}/{variant}/` and print the full run summary. Add `--include-output` to print completion text on stdout; with `--save-result`, also write `{timestamp}-output.txt` beside the JSON.

## What it does

One invocation of `python single_test_runner.py` (default probe):

1. Load parent `model.yaml`, variant `server.yaml`, and `client.yaml`
2. Start `llama-server` with the configured model and flags
3. Poll until the API is healthy
4. POST one streaming `/v1/chat/completions` request
5. Print a rounded metrics block, a blank line, and a three-line headline footer on stdout (end-to-end time, peak VRAM in GiB, generation throughput).
6. Stop the server process

With `--save-result`, step 5 also writes `results/{model}/{variant}/{timestamp}.json` and prints `format_run_summary` on stdout (run metadata plus the same metrics block and headlines; or, with `--save-result --quiet`, a single `Wrote ...` line on stderr). With `--include-output`, completion text is printed on stdout after the metrics block; with `--save-result --include-output`, it is also written to `{timestamp}-output.txt` next to the JSON.

## Quick start

From `tester-v1/`, check Python deps (`httpx`, PyYAML). Skip the venv block if this prints `deps ok`:

```bash
python3 -c "import httpx, yaml" 2>/dev/null && echo "deps ok" || echo "need install"
```

If you see `need install` (common on stock Debian `python3`; see repo root Dependencies for `wsl-builder dev-python`):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run a probe (reference hello-world YAML + model `model.yaml`):

```bash
./single_test_runner.py \
  --server configs/reference/hello-world-baseline/server.yaml \
  --client configs/reference/hello-world-baseline/client.yaml \
  --model-yaml configs/qwen3.5-9b-q8/model.yaml
```

Both CLIs are executable (`#!/usr/bin/env python3`). Use `./single_test_runner.py` and `./model_calibration.py` from this directory, or `python3` with the same arguments if you prefer.

Unit tests:

```bash
python3 -m unittest discover -s tests -v
```

## Workflow

1. Pick or create a model directory under `configs/` with `model.yaml` only (see layout below). Shared probe YAML lives under `configs/reference/{variant}/`.
2. Set the GGUF path in `model.yaml`; tune shared server flags and prompts in `configs/reference/` (or add model-specific variant dirs for experiments).
3. Run a probe with reference YAML and `--model-yaml` (metrics block and headline footer on stdout, no JSON):

   ```bash
   python3 single_test_runner.py \
     --server configs/reference/hello-world-baseline/server.yaml \
     --client configs/reference/hello-world-baseline/client.yaml \
     --model-yaml configs/qwen3.5-9b-q8/model.yaml
   ```

4. When you want a recorded run under `results/{model}/{variant}/`, pass a `config_dir` at `configs/{model}/{variant}/` (may be an empty dir) with the same `--server` / `--client` overrides, plus `--save-result`.

Use separate `config_dir` names (for example `hello-world-baseline` vs `hello-world-bench`) instead of editing root-level yamls when comparing configurations.

## Model calibration

Derive GGUF size on disk, model VRAM, KV VRAM budget, estimated max context, and generation throughput from three probe runs. By default, `model_calibration.py` runs footprint, ctx-probe, and `hello-world-baseline` as probes (same as `single_test_runner.py` with no flags): probe metrics on stdout, no JSON under `results/`. Pass `--save-result` to record probe JSON via `single_test_runner.py --save-result --quiet` (shared `session_id` on all three probes), read metrics from the latest JSON under each variant directory, and write `results/{model}/calibration-sessions/{session_id}.json`.

Each model directory (for example `configs/qwen3.5-9b-q8/`) must include `model.yaml` only. `model_calibration.py` runs three probes using `server.yaml` and `client.yaml` from `configs/reference/`:

- **`calibration-footprint/`** — `server.yaml` with `--fit off`, `-c 4096` (or your chosen footprint context), `--parallel 1`; `client.yaml` with a minimal prompt (`ok`) and `max_tokens: 1`. Idle VRAM after ready is the model footprint at that context.
- **`calibration-ctx-probe/`** — same server flags except a higher `-c` (typically `16384`). The footprint and ctx-probe `-c` values must differ so KV VRAM per token can be estimated from the idle delta.
- **`hello-world-baseline/`** — standard smoke prompt; `decode_tok_s` from the completion becomes generation throughput in the summary.

Live models may add extra variant dirs (for example `hello-world-bench/`) for experiments; those are not copied from `reference/` automatically.

From `tester-v1/`:

```bash
./model_calibration.py configs/qwen3.5-9b-q8
```

Progress and errors go to stderr. On success, stdout is each probe's metrics block and headline footer (default mode), then a blank line and six calibration summary lines (unless `--save-result --quiet`):

```text

* Generation throughput: ~91 tok/s
* Estimated Max Context: 241987 tokens
* Model Max Context: 128000 tokens
* Model VRAM: 10353 MiB
* KV VRAM: 4414 MiB
* GGUF on disk: 9.55 GB
```

`Generation throughput` uses `decode_tok_s` from the hello-world probe (rounded, README-style `~N tok/s`). `estimated_context_max` is VRAM-derived and can exceed the model cap. `model_max_context` is the native limit from llama-server `GET /v1/models` (`data[0].meta.n_ctx_train`), recorded on the footprint probe after the server is ready; it is `n/a` when the API omits that field.

A small VRAM safety buffer (default **100 MiB**, `--margin-mib`) is subtracted from the KV budget. Use `--margin-mib 0` for no buffer, or a higher value for extra headroom.

Copy the summary lines into your model notes (see the repo root README Qwen section). Add `--save-result` when you want calibration probe JSON on disk under `results/{model}/{variant}/` and a session summary for later local reuse (gitignored).

| Flag | Behavior |
| ---- | -------- |
| (default) | Run footprint, ctx-probe, and hello-world as probes; print probe stdout, then calibration summary; no JSON |
| `--save-result` | Run probes with `--save-result --quiet` and shared `--session-id`; read latest JSON per variant dir; write `calibration-sessions/{session_id}.json` |
| `--quiet` | Only with `--save-result`: suppress the six-line calibration summary on stdout |
| `--margin-mib` | VRAM safety buffer subtracted from KV budget (default `100`) |
| `--tester-root` | Harness root (default: `tester-v1/`) |

```bash
# Probe calibration (stdout only)
./model_calibration.py configs/qwen3.5-9b-q8

# Record probe JSON + calibration-sessions summary
./model_calibration.py configs/qwen3.5-9b-q8 --save-result
```

To scaffold all reference variants for a new model, use the [create-model-configs](../.cursor/skills/create-model-configs/SKILL.md) project skill (copies every variant under `configs/reference/`).

## configs layout

Variant configs live under `configs/reference/` for shared probes; live model dirs hold `model.yaml` only unless you add experiment variants. Saved results mirror `configs/{model}/{variant}/` under `results/` (calibration creates ephemeral `{model}/{variant}/` dirs for layout when saving). [configs/reference/README.md](configs/reference/README.md) documents the reference tree.

Example:

```text
configs/
  reference/           # global templates (model.yaml + three variants)
    model.yaml
    calibration-footprint/
    calibration-ctx-probe/
    hello-world-baseline/
  qwen3.5-9b-q8/       # live model
    model.yaml
    hello-world-bench/     # optional, model-specific experiment
```

Each live model folder (for example `qwen3.5-9b-q8`) must contain `model.yaml` with the GGUF path. Reference variant subdirs each have `server.yaml` and `client.yaml`. `model_calibration.py` reads reference YAML and uses `{model}/{variant}/` only for result path layout (and removes empty dirs after each probe).

## Config essentials

Example files: [configs/reference/model.yaml](configs/reference/model.yaml), [configs/reference/hello-world-baseline/server.yaml](configs/reference/hello-world-baseline/server.yaml), [configs/reference/hello-world-baseline/client.yaml](configs/reference/hello-world-baseline/client.yaml). Root [server.yaml](server.yaml) and [client.yaml](client.yaml) remain the default when no config directory is passed (override-only mode requires `--model-yaml`; see CLI).

**model.yaml** (one per model directory, parent of variant dirs)


| Field   | Role                                              |
| ------- | ------------------------------------------------- |
| `model` | Path to the GGUF file (required; file must exist) |

Example: [configs/reference/model.yaml](configs/reference/model.yaml).

**server.yaml**


| Field                   | Role                                                                                     |
| ----------------------- | ---------------------------------------------------------------------------------------- |
| `args`                  | Extra `llama-server` flags (block scalar only; see below). Do not pass `-m` or `--model` (the runner injects `-m` from `model.yaml`) |
| `binary`                | Server executable (default `llama-server`)                                               |
| `ready_timeout_s`       | Max seconds to wait for health (default 120)                                             |
| `ready_poll_interval_s` | Poll interval (default 0.5)                                                              |

`args` must be a multiline block scalar (`args: |`). Put one flag per line. Space seperated `key value` forms (for example `--host 127.0.0.1`, `-c 4096`); the loader splits each into separate argv tokens for `llama-server`. Flags without a value (for example `--no-mmap`) stay on one line. Omit `args` or use an empty block for no extra flags. Example: [configs/reference/hello-world-baseline/server.yaml](configs/reference/hello-world-baseline/server.yaml).

**client.yaml**


| Field       | Role                                                                   |
| ----------- | ---------------------------------------------------------------------- |
| `base_url`  | API root; must match the server port (default `http://127.0.0.1:8080`) |
| `messages`  | Chat messages for the completion                                       |
| `params`    | OpenAI-style fields (`max_tokens`, `temperature`, and so on)           |
| `timeout_s` | Max seconds for the HTTP stream (default 600)                          |


The runner always sets `stream: true` on the request body regardless of `params`.

If `server.yaml` sets `--port` / `-p`, keep `base_url` in sync (or omit port in `base_url` and let the runner align it when possible).

## CLI

Run from `tester-v1/` as `./single_test_runner.py` or `./model_calibration.py` (see Quick start), or equivalently with `python3`.

Probe with reference YAML and explicit `model.yaml` (no `config_dir`; results go under `results/_other/` if you use `--save-result`):

```bash
python3 single_test_runner.py \
  --server configs/reference/hello-world-baseline/server.yaml \
  --client configs/reference/hello-world-baseline/client.yaml \
  --model-yaml configs/qwen3.5-9b-q8/model.yaml
```

Recorded run under `results/{model}/{variant}/` (`config_dir` may be empty; overrides supply YAML; model path from parent `model.yaml`):

```bash
mkdir -p configs/qwen3.5-9b-q8/hello-world-baseline
python3 single_test_runner.py configs/qwen3.5-9b-q8/hello-world-baseline \
  --server configs/reference/hello-world-baseline/server.yaml \
  --client configs/reference/hello-world-baseline/client.yaml \
  --save-result
rmdir configs/qwen3.5-9b-q8/hello-world-baseline 2>/dev/null || true
```

Variant dir with local `server.yaml` and `client.yaml` (optional model-specific experiments):

```bash
python3 single_test_runner.py configs/qwen3.5-9b-q8/hello-world-bench
```

Custom paths without a config directory (probe only; `--save-result` writes under `results/_other/`). You must pass all three paths:

```bash
python3 single_test_runner.py --server /path/to/server.yaml --client /path/to/client.yaml \
  --model-yaml /path/to/model.yaml
```

Flags (same config resolution as above; pass `config_dir` when testing a variant):


| Flag             | Behavior                                                                                                      |
| ---------------- | ------------------------------------------------------------------------------------------------------------- |
| (default)        | Full run; all thirteen metrics plus headline footer on stdout; no JSON                                         |
| `--save-result`  | Requires `config_dir`; write `results/{model}/{variant}/{timestamp}.json`; print full run summary on stdout |
| `--quiet`        | Only with `--save-result`: suppress stdout summary; on success print `Wrote results/...` to stderr              |
| `--include-output` | Print completion text on stdout; with `--save-result`, also write `{timestamp}-output.txt` beside the JSON |
| `--model-yaml`   | `model.yaml` path (default: parent of `config_dir`; required with `--server` and `--client` when `config_dir` is omitted) |
| `--session-id`   | Optional; set on saved JSON (used by `model_calibration` subprocesses)                                      |
| `--test-server`  | Start server, wait for health, hold until Ctrl+C (no completion, no JSON)                                     |


Examples:

```bash
# Probe (stdout only)
python3 single_test_runner.py \
  --server configs/reference/hello-world-baseline/server.yaml \
  --client configs/reference/hello-world-baseline/client.yaml \
  --model-yaml configs/qwen3.5-9b-q8/model.yaml

# Probe with completion text on stdout
python3 single_test_runner.py \
  --server configs/reference/hello-world-baseline/server.yaml \
  --client configs/reference/hello-world-baseline/client.yaml \
  --model-yaml configs/qwen3.5-9b-q8/model.yaml \
  --include-output

# Server-only debug
python3 single_test_runner.py \
  --server configs/reference/hello-world-baseline/server.yaml \
  --client configs/reference/hello-world-baseline/client.yaml \
  --model-yaml configs/qwen3.5-9b-q8/model.yaml \
  --test-server
```

## Results

`results/` is gitignored local output. Nothing under it is tracked in git; copy metrics into the repo README (or other docs) when you want them versioned.

With `--save-result`, the results tree mirrors `configs/{model}/{variant}/`:

```text
results/
  {model}/
    {variant}/
      {YYYYMMDDTHHMMSSZ}.json
      {YYYYMMDDTHHMMSSZ}-output.txt   # only with --save-result --include-output
    calibration-sessions/
      {session_id}.json
  _other/
    {slug}/
      {YYYYMMDDTHHMMSSZ}.json
```

When `config_dir` is not under the standard two-level `configs/{model}/{variant}/` path, JSON is written under `results/_other/{slug}/` instead (`slug` from `slug_from_config_dir`).

**run_id** in JSON is still globally unique: `{YYYYMMDDTHHMMSSZ}_{model}-{variant}` (e.g. `20260517T170240Z_qwen3.5-9b-q8-hello-world-baseline`). On-disk filenames use the timestamp stem only.

Each JSON document includes:

- `schema_version`: `"1"`
- `suite`: `"probe"` or `"calibration"` (from variant name)
- `session_id`: `null` or a string (`model_calibration --save-result` sets the same id on all three probe runs)
- `run_id`, `started_at`, `finished_at`, `status`, `error`, `notes`
- `server_config` and `client_config` (embedded copies of what ran)
- `metadata` (`server_version` from `llama-server --version`, `gpu_name` and `driver_version` from `nvidia-smi`; when `config_dir` is under `configs/`, also `config_path` e.g. `qwen3.5-9b-q8/hello-world-baseline/` plus `model` and `variant` from the path; each field is `null` when unavailable)
- `metrics` (see table below)

**Calibration sessions:** `model_calibration --save-result` generates one `session_id`, passes it to all three probe subprocesses, reads the latest JSON from each variant directory, and writes `results/{model}/calibration-sessions/{session_id}.json` (summary: `gguf_gb`, `model_vram_mb`, `kv_vram_mb`, `estimated_context_max`, `model_max_context`, `generation_throughput_tok_s`; plus footprint, ctx-probe, and hello-world probe `run_id` and path refs).

Default probe stdout prints every metric key below (rounded, `n/a` when missing), then a blank line and three headline lines derived from `wall_time_s` (end-to-end time), `peak_vram_mb` (peak VRAM as GiB), and `decode_tok_s` (generation throughput). Completion text is omitted unless you pass `--include-output` (printed after the headlines). With `--save-result --include-output`, the same text is also saved beside the JSON. With `--save-result`, stdout is the run summary (run id, model, result path, then the same metrics block and headlines) unless `--quiet`. The JSON `metrics` object keeps full floating-point values for every field.

Example probe tail (after `Model: ...`):

```text
server_ready_s      1.50
wall_time_s         9.14
ttft_s              0.42
completion_time_s   8.72
prompt_tokens       128
completion_tokens   256
total_tokens        384
prefill_tok_s       304.76
decode_tok_s        116.32
tokens_per_second   28.01
idle_vram_mb        8000
peak_vram_mb        8961
model_max_context   128000

End-to-end time         9.14 s
Peak VRAM               8.75 GiB
Generation throughput   116.32 tok/s
```


| Metric              | What it measures                                                                                                 |
| ------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `server_ready_s`    | Seconds from `llama-server` start until the first successful `/health` or `/v1/models` check (proxy for load).   |
| `wall_time_s`       | Total time from sending the request until the stream ends.                                                       |
| `ttft_s`            | Time until the first streamed token (first non-empty `content`, `reasoning_content`, or `text` delta).           |
| `completion_time_s` | `wall_time_s` minus `ttft_s`; time spent after the first token (decode phase).                                   |
| `prompt_tokens`     | Input token count from the server (`usage` or `timings.prompt_n`).                                               |
| `completion_tokens` | Generated token count (`usage` or `timings.predicted_n`).                                                        |
| `total_tokens`      | Prompt plus completion when both counts are known; otherwise `null`.                                             |
| `prefill_tok_s`     | `prompt_tokens` / `ttft_s`; approximate prompt-processing rate.                                                  |
| `decode_tok_s`      | `completion_tokens` / `completion_time_s`; approximate generation rate after the first token.                    |
| `tokens_per_second` | `completion_tokens` / `wall_time_s`; end-to-end completion throughput including TTFT.                            |
| `idle_vram_mb`      | GPU memory in MB after the server is ready and before the measured prompt (`nvidia-smi`; `null` if unavailable). |
| `peak_vram_mb`      | Peak GPU memory in MB during the measured prompt (background `nvidia-smi` poll; `null` if unavailable).          |
| `model_max_context` | Native context limit from `GET /v1/models` (`data[0].meta.n_ctx_train`; `null` if missing or unreadable).       |


## Critical gotchas

These are easy to miss from a quick read of the code:

- **Model load time:** Recorded as `server_ready_s` (subprocess start to first health 200). The first start can take several minutes. If load exceeds `ready_timeout_s`, the run fails even though the server might still be loading. Raise `ready_timeout_s` for large models or slow disks.
- **Token counts on this stack:** Many `llama-server` builds omit `usage` on stream chunks. The client reads `timings` (`prompt_n`, `predicted_n`) from the final chunk instead. TTFT is time to the first non-empty delta on `content`, `reasoning_content`, or `text`; models that stream reasoning before visible content can show a lower TTFT than "first answer token."
- **Clean stop:** Use Ctrl+C in the terminal for a controlled interrupt. With `--save-result`, the runner records `status: error` and still writes JSON when possible. Killing the process from outside (or a hard external timeout) may leave `llama-server` running in the background.
- **No model in server.yaml:** The GGUF path lives only in `model.yaml` at the model root. Putting `model:` in `server.yaml` is rejected.
- **No model flag in args:** Putting `-m` or `--model` in `server.yaml` `args` is rejected; the runner appends `-m` with the path from `model.yaml`.
- **Removed `label` field:** `server.yaml` must not contain `label`; use a config directory so results land under `results/{model}/{variant}/`.
