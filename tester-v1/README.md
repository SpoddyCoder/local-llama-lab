# Tester v1

Single-run harness for `llama-server`: start the server, run one streaming chat completion, write a JSON result, print a metrics summary, then tear down. For the full Phase 1 design (modules, acceptance criteria, verified behavior), see [docs/tester-v1-implementation-plan.md](../docs/tester-v1-implementation-plan.md).

## What it does

One invocation of `python src/python_runner.py`:

1. Load `server.yaml` and `client.yaml`
2. Start `llama-server` with the configured model and flags
3. Poll until the API is healthy
4. POST one streaming `/v1/chat/completions` request
5. Write `results/{timestamp}_{slug}.json`
6. Print metrics on stdout
7. Stop the server process

## Quick start

From `tester-v1/`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/python_runner.py
```

Unit tests:

```bash
python3 -m unittest discover -s tests -v
```

## Workflow

1. Tune the model and server flags in `server.yaml` (quant, context, GPU layers, port, and so on).
2. Tune the prompt and API params in `client.yaml` (`messages`, `params`).
3. Re-run `python src/python_runner.py`.
4. Compare new files under `results/` (timestamps and slugs distinguish runs).

Use `label` in `server.yaml` when you want a short, stable slug in filenames instead of the model file stem.

## Config essentials

Example files: `server.yaml`, `client.yaml` (README Test 1 baseline).

**server.yaml**

| Field | Role |
|-------|------|
| `model` | Path to the GGUF file (required; file must exist) |
| `args` | Extra `llama-server` flags only; do not pass `-m` or `--model` (the runner injects `-m`) |
| `label` | Optional; if set, used for the result filename slug instead of the model stem |
| `binary` | Server executable (default `llama-server`) |
| `ready_timeout_s` | Max seconds to wait for health (default 120) |
| `ready_poll_interval_s` | Poll interval (default 0.5) |

**client.yaml**

| Field | Role |
|-------|------|
| `base_url` | API root; must match the server port (default `http://127.0.0.1:8080`) |
| `messages` | Chat messages for the completion |
| `params` | OpenAI-style fields (`max_tokens`, `temperature`, and so on) |
| `timeout_s` | Max seconds for the HTTP stream (default 600) |

The runner always sets `stream: true` on the request body regardless of `params`.

If `server.yaml` sets `--port` / `-p`, keep `base_url` in sync (or omit port in `base_url` and let the runner align it when possible).

## CLI

Default (full loop):

```bash
python src/python_runner.py
```

Custom config paths:

```bash
python src/python_runner.py --server /path/to/server.yaml --client /path/to/client.yaml
```

Debug modes (same config loading, different behavior):

| Flag | Behavior |
|------|----------|
| `--test-server` | Start server, wait for health, hold until Ctrl+C (no completion, no JSON) |
| `--test-client` | Full server plus one streaming completion and metrics on stdout; no result JSON write |

## Results

Files land in `results/` as:

`{YYYYMMDDTHHMMSSZ}_{slug}.json`

`slug` is the sanitized `label` if set, otherwise the model file stem (e.g. `Qwen_Qwen3.5-9B-Q8_0`).

Each JSON document includes:

- `run_id`, `started_at`, `finished_at`, `status`, `error`, `notes`
- `server_config` and `client_config` (embedded copies of what ran)
- `metrics` (see table below)

Stdout prints a rounded subset (`wall_time_s`, `ttft_s`, token counts, `prefill_tok_s`, `decode_tok_s`). The JSON `metrics` object keeps full floating-point values for every field.

| Metric | What it measures |
|--------|------------------|
| `wall_time_s` | Total time from sending the request until the stream ends. |
| `ttft_s` | Time until the first streamed token (first non-empty `content`, `reasoning_content`, or `text` delta). |
| `completion_time_s` | `wall_time_s` minus `ttft_s`; time spent after the first token (decode phase). |
| `prompt_tokens` | Input token count from the server (`usage` or `timings.prompt_n`). |
| `completion_tokens` | Generated token count (`usage` or `timings.predicted_n`). |
| `total_tokens` | Prompt plus completion when both counts are known; otherwise `null`. |
| `prefill_tok_s` | `prompt_tokens` / `ttft_s`; approximate prompt-processing rate. |
| `decode_tok_s` | `completion_tokens` / `completion_time_s`; approximate generation rate after the first token. |
| `tokens_per_second` | `completion_tokens` / `wall_time_s`; end-to-end completion throughput including TTFT. |
| `peak_vram_mb` | Peak GPU memory in MB; not collected in Phase 1 (always `null`). |

## Critical gotchas

These are easy to miss from a quick read of the code:

- **Model load time:** The first start can take several minutes before `/health` returns 200. If load exceeds `ready_timeout_s`, the run fails even though the server might still be loading. Raise `ready_timeout_s` for large models or slow disks.
- **Token counts on this stack:** Many `llama-server` builds omit `usage` on stream chunks. The client reads `timings` (`prompt_n`, `predicted_n`) from the final chunk instead. TTFT is time to the first non-empty delta on `content`, `reasoning_content`, or `text`; models that stream reasoning before visible content can show a lower TTFT than "first answer token."
- **Clean stop:** Use Ctrl+C in the terminal for a controlled interrupt. The runner records `status: error` and still writes JSON when possible. Killing the process from outside (or a hard external timeout) may leave `llama-server` running in the background.
- **No model flag in args:** Putting `-m` or `--model` in `server.yaml` `args` is rejected; the runner appends `-m` with the `model` path.