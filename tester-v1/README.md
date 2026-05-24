# Tester v1

Single-run harness for `llama-server`: start the server (`server.yaml`), run one streaming chat completion (`client.yaml`), print all fourteen metrics plus a headline footer on stdout, then tear down.

- Pass `--save-result` with a variant directory to write JSON under `results/{model}/{variant}/` and print the full run summary.
- Add `--include-output` to print completion text on stdout; with `--save-result`, also write `{timestamp}-output.txt` beside the JSON.

Repo-root CLIs wrap this harness: [tester_v1_run_test.py](../tester_v1_run_test.py), [tester_v1_calibrate.py](../tester_v1_calibrate.py), [launch_server.py](../launch_server.py).

## Requirements

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

## Usage

Run from the repo root:

```bash
# print help
./tester_v1_run_test.py

# bench run for qwen3.5-9b-q8
./tester_v1_run_test.py models/qwen3.5-9b-q8/bench/

# bench run with 24 MoE experts offloaded to CPU
./tester_v1_run_test.py models/qwen3.6-35b-a3b-ud-q4-k-xl/bench/ --n-cpu-moe 24

# save detailed results JSON to results/
./tester_v1_run_test.py models/qwen3.5-9b-q8/bench/ --save-result

# print completion text on stdout
./tester_v1_run_test.py models/qwen3.5-9b-q8/bench/ --include-output
```

| Flag               | Effect                                                                                          |
| ------------------ | ----------------------------------------------------------------------------------------------- |
| `--save-result`    | Write `results/{model}/{variant}/{timestamp}.json`; print run summary on stdout                 |
| `--quiet`          | With `--save-result`: one `Wrote ...` line on stderr instead of stdout summary                  |
| `--include-output` | Print completion text after metrics; with `--save-result`, also writes `{timestamp}-output.txt` |
| `--n-cpu-moe N`    | Append `--n-cpu-moe N` and `--n-gpu-layers 999` (MoE models)                                    |

## Unit tests

From the repo root:

```bash
./tester-v1/run_python_unit_tests.py
```

Or from `tester-v1/`:

```bash
./run_python_unit_tests.py
```

## Config layout

```text
models/                              # repo root
  qwen3.5-9b-q8/
    model.yaml                       # GGUF path
    bench/                           # server.yaml + client.yaml for routine runs

tester-v1/
  calibration-tests/                 # shared calibration probes (not model configs)
    calibration-footprint/
    calibration-ctx-probe/
    hello-world-baseline/
    model.yaml                       # structural template for scaffolding
  results/                           # gitignored harness output
  src/                               # harness implementation
```

Each model dir under [models/](../models/) needs `model.yaml`. Variant dirs (e.g. `bench/`) hold `server.yaml` and `client.yaml`. Shared calibration probe YAML lives under [calibration-tests/](calibration-tests/); see [calibration-tests/README.md](calibration-tests/README.md).

**model.yaml:** `model` is the GGUF path. The runner injects `-m`; do not set `-m` or `--model` in `server.yaml`.

**server.yaml:** `args` (`args: |`, one flag per line), optional `binary`, `ready_timeout_s` (default 120), `ready_poll_interval_s` (default 0.5).

**client.yaml:** `base_url` (default `http://127.0.0.1:8080`), `messages`, `params`, optional `timeout_s` (default 600). The runner always sets `stream: true`. Keep `base_url` in sync with `--port` in server args.

## Launch server

[launch_server.py](../launch_server.py) starts `llama-server` from a model's `bench/` config, waits for health, and holds until Ctrl+C. Server stdout/stderr are not captured.

```bash
./launch_server.py models/qwen3.5-9b-q8/
./launch_server.py models/qwen3.6-35b-a3b-ud-q4-k-xl/ --n-cpu-moe 24
```

You can also pass a variant directory directly (e.g. `models/qwen3.5-9b-q8/bench/`).

## Calibration

[tester_v1_calibrate.py](../tester_v1_calibrate.py) runs three shared probes (footprint, ctx-probe, hello-world-baseline) from `calibration-tests/` and prints a VRAM and throughput summary on six lines by default. Footprint and ctx-probe use different `--ctx-size` values; idle VRAM delta estimates KV cost per token.

When calibration runs with MoE CPU offload (`./tester_v1_calibrate.py models/qwen3.6-35b-a3b-ud-q4-k-xl --n-cpu-moe 24`, same pattern as Usage), the trailing summary block grows from six lines to seven. The extra line comes from the footprint probe only: `* Model System RAM: {N.NN} GB` when RSS sampling succeeds, or `* Model System RAM: unavailable` when it fails. Ctx-probe does not contribute this value.

```bash
./tester_v1_calibrate.py models/qwen3.5-9b-q8
./tester_v1_calibrate.py models/qwen3.6-35b-a3b-ud-q4-k-xl --n-cpu-moe 24
./tester_v1_calibrate.py models/qwen3.5-9b-q8 --save-result
```

`--margin-mib` (default 100) subtracts a VRAM safety buffer from the KV budget. Estimated max context is VRAM-derived and can exceed the model native cap. Copy summary lines into the repo root README. New models: [create-model-configs](../.cursor/skills/create-model-configs/SKILL.md) skill.

## Results

`results/` is gitignored. Probe JSON: `results/{model}/{variant}/{timestamp}.json`. Calibration with `--save-result` also writes `results/{model}/calibration-sessions/{session_id}.json`; MoE offload runs include `model_system_ram_mb` in the session `summary` when `--n-cpu-moe` was set.

Default stdout: all metrics (rounded), blank line, three headline lines (end-to-end time, peak VRAM, decode tok/s). With `--save-result`, stdout is a run summary unless `--quiet`.

## Metrics


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
| `idle_system_ram_mb` | Process RSS (MiB) for `llama-server` after load at idle, before the measured prompt (`VmRSS` from `/proc/{pid}/status`; same idle window as `idle_vram_mb`, which uses `nvidia-smi`). Only sampled when resolved server args include `--n-cpu-moe` (MoE CPU offload); `null` / n/a when not applicable or if sampling fails. |
| `peak_vram_mb`      | Peak GPU memory in MB during the measured prompt (background `nvidia-smi` poll; `null` if unavailable).          |
| `model_max_context` | Native context limit from `GET /v1/models` (`data[0].meta.n_ctx_train`; `null` if missing or unreadable).        |

