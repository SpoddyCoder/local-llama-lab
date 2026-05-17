# Tester v1

Single-run harness for `llama-server`: start the server, run one streaming chat completion, print all twelve metrics plus a headline footer on stdout, then tear down. Pass `--save-result` to also write `results/{timestamp}_{slug}.json` and print the full run summary. On a probe run, `--full-output FILE` writes the full completion text to a file (stdout stays metrics only).

## What it does

One invocation of `python single_test_runner.py` (default probe):

1. Load `server.yaml` and `client.yaml`
2. Start `llama-server` with the configured model and flags
3. Poll until the API is healthy
4. POST one streaming `/v1/chat/completions` request
5. Print a rounded metrics block, a blank line, and a three-line headline footer on stdout (end-to-end time, peak VRAM in GiB, generation throughput). Optionally write full completion text to a file with `--full-output`.
6. Stop the server process

With `--save-result`, step 5 is replaced by writing `results/{timestamp}_{slug}.json` and printing `format_run_summary` on stdout (run metadata plus the same metrics block and headlines; or, with `--save-result --quiet`, a single `Wrote ...` line on stderr). `--full-output` is not used with `--save-result` (warning on stderr, no file write).

## Quick start

From `tester-v1/`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
./single_test_runner.py configs/qwen3.5-9b-q8/hello-world-baseline
```

Both CLIs are executable (`#!/usr/bin/env python3`). Use `./single_test_runner.py` and `./model_calibration.py` from this directory, or `python3` with the same arguments if you prefer.

Unit tests:

```bash
python3 -m unittest discover -s tests -v
```

## Workflow

1. Pick or create a variant directory under `configs/` (see layout below). Each leaf holds `server.yaml` and `client.yaml` for one run configuration.
2. Tune model path, server flags, prompt, and API params in that pair of files.
3. Run the variant (probe: metrics block and headline footer on stdout, no JSON; optional `--full-output` for completion text):

   ```bash
   python3 single_test_runner.py configs/qwen3.5-9b-q8/hello-world-baseline
   ```

4. When you want a recorded run, add `--save-result` and compare files under `results/` (timestamps and path-derived slugs distinguish runs).

Use separate variant directories (for example `hello-world-baseline` vs `hello-world-no-mmap`) instead of editing root-level yamls when comparing configurations.

## Model calibration

Derive GGUF size on disk, model VRAM, KV VRAM budget, and estimated max context from two probe runs. By default, `model_calibration.py` runs each calibration variant as a probe (same as `single_test_runner.py` with no flags): probe metrics on stdout, no JSON under `results/`. Pass `--save-result` to record probe JSON via `single_test_runner.py --save-result --quiet` and read `idle_vram_mb` from the latest footprint and ctx-probe files.

Each model directory (for example `configs/qwen3.5-9b-q8/`) must include two calibration variants:

- **`calibration-footprint/`** — `server.yaml` with `--fit off`, `-c 4096` (or your chosen footprint context), `--parallel 1`; `client.yaml` with a minimal prompt (`ok`) and `max_tokens: 1`. Idle VRAM after ready is the model footprint at that context.
- **`calibration-ctx-probe/`** — same server flags except a higher `-c` (typically `16384`). The footprint and ctx-probe `-c` values must differ so KV VRAM per token can be estimated from the idle delta.

`hello-world-baseline` is the standard smoke probe variant; calibration does not run it. Live models may also have extra probe dirs (for example `hello-world-no-mmap`) that are not in `reference/`.

From `tester-v1/`:

```bash
./model_calibration.py configs/qwen3.5-9b-q8
```

Progress and errors go to stderr. On success, stdout is each probe's metrics block and headline footer (default mode), then a blank line and four calibration summary lines (unless `--save-result --quiet`):

```text

* GGUF on disk: 9.55 GB
* Model VRAM: 10353 MiB
* KV VRAM: 4414 MiB
* Estimated Max Context: 241987 tokens
```

Optional `--margin-mib` reserves headroom for non-KV GPU use (default `1536`). Copy the four summary lines into your model notes (see the repo root README Qwen section). Add `--save-result` when you want calibration probe JSON under `results/` for later reuse.

| Flag | Behavior |
| ---- | -------- |
| (default) | Run footprint and ctx-probe as probes; print probe stdout, then calibration summary; no JSON |
| `--save-result` | Run probes with `--save-result --quiet`; read `idle_vram_mb` from latest JSON in `results/` |
| `--quiet` | Only with `--save-result`: suppress the four-line calibration summary on stdout |
| `--margin-mib` | VRAM headroom subtracted from KV budget (default `1536`) |
| `--tester-root` | Harness root (default: `tester-v1/`) |

```bash
# Probe calibration (stdout only)
./model_calibration.py configs/qwen3.5-9b-q8

# Record probe JSON under results/
./model_calibration.py configs/qwen3.5-9b-q8 --save-result
```

To scaffold all reference variants for a new model, use the [create-model-configs](../.cursor/skills/create-model-configs/SKILL.md) project skill (copies every variant under `configs/reference/`).

## configs layout

Variant configs live under `configs/`. The directory tree is organizational only; the runner does not interpret segment names beyond building the result slug. [configs/reference/README.md](configs/reference/README.md) documents the reference template tree.

Example:

```text
configs/
  reference/           # global templates (three variants)
    calibration-footprint/
    calibration-ctx-probe/
    hello-world-baseline/
  qwen3.5-9b-q8/       # live model; may include extra dirs
    calibration-footprint/
    calibration-ctx-probe/
    hello-world-baseline/
    hello-world-no-mmap/   # optional, model-specific
```

Each variant directory must contain both `server.yaml` and `client.yaml`. Model folders (for example `qwen3.5-9b-q8`) group related variants; they are not special-cased in code beyond metadata and the result slug.

## Config essentials

Example files (variant shape): [configs/reference/hello-world-baseline/server.yaml](configs/reference/hello-world-baseline/server.yaml), [configs/reference/hello-world-baseline/client.yaml](configs/reference/hello-world-baseline/client.yaml). Root [server.yaml](server.yaml) and [client.yaml](client.yaml) remain the default when no config directory is passed.

**server.yaml**


| Field                   | Role                                                                                     |
| ----------------------- | ---------------------------------------------------------------------------------------- |
| `model`                 | Path to the GGUF file (required; file must exist)                                        |
| `args`                  | Extra `llama-server` flags only; do not pass `-m` or `--model` (the runner injects `-m`) |
| `binary`                | Server executable (default `llama-server`)                                               |
| `ready_timeout_s`       | Max seconds to wait for health (default 120)                                             |
| `ready_poll_interval_s` | Poll interval (default 0.5)                                                              |


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

Config directory (primary workflow; loads `config_dir/server.yaml` and `config_dir/client.yaml`, path-derived slug):

```bash
python3 single_test_runner.py configs/qwen3.5-9b-q8/hello-world-baseline
```

Override one or both config files while still using the config directory for the slug:

```bash
python3 single_test_runner.py configs/qwen3.5-9b-q8/hello-world-baseline \
  --server /path/to/server.yaml \
  --client /path/to/client.yaml
```

Custom paths without a config directory (model file stem as slug):

```bash
python3 single_test_runner.py --server /path/to/server.yaml --client /path/to/client.yaml
```

Flags (same config resolution as above; pass `config_dir` when testing a variant):


| Flag             | Behavior                                                                                                      |
| ---------------- | ------------------------------------------------------------------------------------------------------------- |
| (default)        | Full run; all twelve metrics plus headline footer on stdout; no JSON                                          |
| `--save-result`  | Write `results/{run_id}.json`; print full run summary on stdout (includes metrics block and headlines)        |
| `--quiet`        | Only with `--save-result`: suppress stdout summary; on success print `Wrote results/...` to stderr              |
| `--full-output`  | Probe only: write full completion text to `FILE` (creates parent dirs). Ignored with `--save-result` (stderr warning, no write) |
| `--test-server`  | Start server, wait for health, hold until Ctrl+C (no completion, no JSON)                                     |


Examples:

```bash
# Probe (stdout only)
python3 single_test_runner.py configs/qwen3.5-9b-q8/hello-world-baseline

# Probe with completion text saved to a file
python3 single_test_runner.py configs/qwen3.5-9b-q8/hello-world-baseline \
  --full-output /tmp/completion.txt

# Recorded run
python3 single_test_runner.py configs/qwen3.5-9b-q8/hello-world-baseline --save-result

# Server-only debug
python3 single_test_runner.py configs/qwen3.5-9b-q8/hello-world-baseline --test-server
```

## Results

With `--save-result`, files land in `results/` as:

`{YYYYMMDDTHHMMSSZ}_{slug}.json`

Example: `20260516T134500Z_qwen3.5-9b-q8-hello-world-baseline.json`

Slug rules:

- With `config_dir` under `configs/`: hyphen-join path segments relative to `configs/` (e.g. `configs/qwen3.5-9b-q8/hello-world-baseline` → `qwen3.5-9b-q8-hello-world-baseline`).
- With `config_dir` outside `configs/`: hyphen-join all directory segments of the resolved absolute path (e.g. `/tmp/my-run` → `tmp-my-run`).
- Without `config_dir`: sanitized stem of the model GGUF filename (e.g. `Qwen_Qwen3.5-9B-Q8_0`).

Slugs are sanitized for filenames (unsafe characters become underscores).

Each JSON document includes:

- `run_id`, `started_at`, `finished_at`, `status`, `error`, `notes`
- `server_config` and `client_config` (embedded copies of what ran)
- `metadata` (`server_version` from `llama-server --version`, `gpu_name` and `driver_version` from `nvidia-smi`; when `config_dir` is under `configs/`, also `config_path` e.g. `qwen3.5-9b-q8/hello-world-baseline/` plus `model` and `variant` from the path; each field is `null` when unavailable)
- `metrics` (see table below)

Default probe stdout prints every metric key below (rounded, `n/a` when missing), then a blank line and three headline lines derived from `wall_time_s` (end-to-end time), `peak_vram_mb` (peak VRAM as GiB), and `decode_tok_s` (generation throughput). Completion text is not printed; use `--full-output FILE` on a probe run to save it. With `--save-result`, stdout is the run summary (run id, model, result path, then the same metrics block and headlines) unless `--quiet`. The JSON `metrics` object keeps full floating-point values for every field.

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


## Critical gotchas

These are easy to miss from a quick read of the code:

- **Model load time:** Recorded as `server_ready_s` (subprocess start to first health 200). The first start can take several minutes. If load exceeds `ready_timeout_s`, the run fails even though the server might still be loading. Raise `ready_timeout_s` for large models or slow disks.
- **Token counts on this stack:** Many `llama-server` builds omit `usage` on stream chunks. The client reads `timings` (`prompt_n`, `predicted_n`) from the final chunk instead. TTFT is time to the first non-empty delta on `content`, `reasoning_content`, or `text`; models that stream reasoning before visible content can show a lower TTFT than "first answer token."
- **Clean stop:** Use Ctrl+C in the terminal for a controlled interrupt. With `--save-result`, the runner records `status: error` and still writes JSON when possible. Killing the process from outside (or a hard external timeout) may leave `llama-server` running in the background.
- **No model flag in args:** Putting `-m` or `--model` in `server.yaml` `args` is rejected; the runner appends `-m` with the `model` path.
- **Removed `label` field:** `server.yaml` must not contain `label`; use a config directory so the result slug reflects the variant path.
