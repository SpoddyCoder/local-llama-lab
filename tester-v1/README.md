# Tester v1

Single-run harness for `llama-server`: start the server (`server.yaml`), run one streaming chat completion (`client.yaml`), print all thirteen metrics plus a headline footer on stdout, then tear down. 

- Pass `--save-result` with a config directory to write JSON under `results/{model}/{variant}/` and print the full run summary. 
- Add `--include-output` to print completion text on stdout; with `--save-result`, also write `{timestamp}-output.txt` beside the JSON.

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

```bash
# print help
./single_test_runner.py

# run the bench test for qwen3.5-9b-q8
./single_test_runner.py configs/qwen3.5-9b-q8/bench/

# run the bench test for qwen3.6-35b-a3b-ud-q4-k-xl with 24 experts offloaded to the CPU
./single_test_runner.py configs/qwen3.6-35b-a3b-ud-q4-k-xl/bench/ --n-cpu-moe 24

 # run the bench test and save detailed results JSON to results/
 ./single_test_runner.py configs/qwen3.5-9b-q8/bench/ --save-result

 # run the bench test and print completion text on stdout
 ./single_test_runner.py configs/qwen3.5-9b-q8/bench/ --include-output
```

## Unit tests

```bash
python3 -m unittest discover -s tests -v
```

## Config layout

```text
configs/
  reference/                 # shared probes (calibration + --server/--client overrides)
    calibration-footprint/
    calibration-ctx-probe/
    hello-world-baseline/
  qwen3.5-9b-q8/
    model.yaml               # GGUF path
    bench/                   # server.yaml + client.yaml for routine runs
```

Each model dir needs `model.yaml`. Variant dirs hold `server.yaml` and `client.yaml`. Shared probe YAML lives under `reference/` only; see [configs/reference/README.md](configs/reference/README.md). Do not pass `configs/reference/` as `config_dir`.

**model.yaml:** `model` is the GGUF path. The runner injects `-m`; do not set `-m` or `--model` in `server.yaml`.

**server.yaml:** `args` (`args: |`, one flag per line), optional `binary`, `ready_timeout_s` (default 120), `ready_poll_interval_s` (default 0.5).

**client.yaml:** `base_url` (default `http://127.0.0.1:8080`), `messages`, `params`, optional `timeout_s` (default 600). The runner always sets `stream: true`. Keep `base_url` in sync with `--port` in server args.

## Probes

`single_test_runner.py` starts `llama-server`, waits for health, runs one streaming chat completion, prints metrics and a three-line summary footer, then stops the server.

Routine bench runs use a variant dir (see Usage above). For a quick probe with shared reference YAML and no JSON:

```bash
./single_test_runner.py \
  --server configs/reference/hello-world-baseline/server.yaml \
  --client configs/reference/hello-world-baseline/client.yaml \
  --model-yaml configs/qwen3.5-9b-q8/model.yaml
```

Without `config_dir`, pass all three of `--server`, `--client`, and `--model-yaml`. `--save-result` requires `config_dir` and writes under `results/{model}/{variant}/`.


| Flag               | Effect                                                                                          |
| ------------------ | ----------------------------------------------------------------------------------------------- |
| `--save-result`    | Write `results/{model}/{variant}/{timestamp}.json`; print run summary on stdout                 |
| `--quiet`          | With `--save-result`: one `Wrote ...` line on stderr instead of stdout summary                  |
| `--include-output` | Print completion text after metrics; with `--save-result`, also writes `{timestamp}-output.txt` |
| `--test-server`    | Start server, wait for health, hold until Ctrl+C                                                |
| `--n-cpu-moe N`    | Append `--n-cpu-moe N` and `--n-gpu-layers 999` (MoE models)                                    |


## Calibration

`model_calibration.py` runs three reference probes (footprint, ctx-probe, hello-world-baseline) and prints a six-line VRAM and throughput summary. Footprint and ctx-probe use different `--ctx-size` values; idle VRAM delta estimates KV cost per token.

```bash
./model_calibration.py configs/qwen3.5-9b-q8
./model_calibration.py configs/qwen3.6-35b-a3b-ud-q4-k-xl --n-cpu-moe 24
./model_calibration.py configs/qwen3.5-9b-q8 --save-result
```

`--margin-mib` (default 100) subtracts a VRAM safety buffer from the KV budget. Estimated max context is VRAM-derived and can exceed the model native cap. Copy summary lines into the repo root README. New models: [create-model-configs](../.cursor/skills/create-model-configs/SKILL.md) skill.

## Results

`results/` is gitignored. Probe JSON: `results/{model}/{variant}/{timestamp}.json`. Calibration with `--save-result` also writes `results/{model}/calibration-sessions/{session_id}.json`.

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
| `peak_vram_mb`      | Peak GPU memory in MB during the measured prompt (background `nvidia-smi` poll; `null` if unavailable).          |
| `model_max_context` | Native context limit from `GET /v1/models` (`data[0].meta.n_ctx_train`; `null` if missing or unreadable).        |


