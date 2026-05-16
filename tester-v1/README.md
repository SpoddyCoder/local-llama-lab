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
python3 src/python_runner.py test-configs/test1/qwen3.5-9b-q8/baseline
```

Unit tests:

```bash
python3 -m unittest discover -s tests -v
```

Without a config directory, the runner uses `server.yaml` and `client.yaml` in `tester-v1/` (same as before).

## Workflow

1. Pick or create a variant directory under `test-configs/` (see layout below). Each leaf holds `server.yaml` and `client.yaml` for one run configuration.
2. Tune model path, server flags, prompt, and API params in that pair of files.
3. Run the variant:

   ```bash
   python3 src/python_runner.py test-configs/test1/qwen3.5-9b-q8/baseline
   ```

4. Compare new files under `results/` (timestamps and path-derived slugs distinguish runs).

Use separate variant directories (for example `baseline` vs `no-mmap`) instead of editing root-level yamls when comparing configurations.

## test-configs layout

Variant configs live under `test-configs/`. The directory tree is organizational only; the runner does not interpret segment names beyond building the result slug.

Example:

```text
test-configs/
  test1/                    # test or experiment group
    qwen3.5-9b-q8/          # model family
      baseline/
        server.yaml
        client.yaml
      no-mmap/
        server.yaml
        client.yaml
```

Each leaf directory must contain both `server.yaml` and `client.yaml`. Intermediate folders (`test1`, `qwen3.5-9b-q8`, and so on) group related variants; they are not special-cased in code.

## Config essentials

Example files: [test-configs/test1/qwen3.5-9b-q8/baseline/server.yaml](test-configs/test1/qwen3.5-9b-q8/baseline/server.yaml), [test-configs/test1/qwen3.5-9b-q8/baseline/client.yaml](test-configs/test1/qwen3.5-9b-q8/baseline/client.yaml). Root [server.yaml](server.yaml) and [client.yaml](client.yaml) remain the default when no config directory is passed.

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

Default (root `server.yaml` / `client.yaml`, model file stem as slug):

```bash
python3 src/python_runner.py
```

Config directory (primary workflow; loads `config_dir/server.yaml` and `config_dir/client.yaml`, path-derived slug):

```bash
python3 src/python_runner.py test-configs/test1/qwen3.5-9b-q8/baseline
```

Override one or both config files while still using the config directory for the slug:

```bash
python3 src/python_runner.py test-configs/test1/qwen3.5-9b-q8/baseline \
  --server /path/to/server.yaml \
  --client /path/to/client.yaml
```

Custom paths without a config directory (model file stem as slug):

```bash
python3 src/python_runner.py --server /path/to/server.yaml --client /path/to/client.yaml
```

Debug modes (same config resolution as above; pass `config_dir` when testing a variant):


| Flag            | Behavior                                                                              |
| --------------- | ------------------------------------------------------------------------------------- |
| `--test-server` | Start server, wait for health, hold until Ctrl+C (no completion, no JSON)             |
| `--test-client` | Full server plus one streaming completion and metrics on stdout; no result JSON write |


Examples:

```bash
python3 src/python_runner.py test-configs/test1/qwen3.5-9b-q8/baseline --test-server
python3 src/python_runner.py test-configs/test1/qwen3.5-9b-q8/no-mmap --test-client
```

## Results

Files land in `results/` as:

`{YYYYMMDDTHHMMSSZ}_{slug}.json`

Example: `20260516T134500Z_test1-qwen3.5-9b-q8-baseline.json`

Slug rules:

- With `config_dir` under `test-configs/`: hyphen-join path segments relative to `test-configs/` (e.g. `test-configs/test1/qwen3.5-9b-q8/baseline` → `test1-qwen3.5-9b-q8-baseline`).
- With `config_dir` outside `test-configs/`: hyphen-join all directory segments of the resolved absolute path (e.g. `/tmp/my-run` → `tmp-my-run`).
- Without `config_dir`: sanitized stem of the model GGUF filename (e.g. `Qwen_Qwen3.5-9B-Q8_0`).

Slugs are sanitized for filenames (unsafe characters become underscores).

Each JSON document includes:

- `run_id`, `started_at`, `finished_at`, `status`, `error`, `notes`
- `server_config` and `client_config` (embedded copies of what ran)
- `metadata` (`server_version` from `llama-server --version`, `gpu_name` and `driver_version` from `nvidia-smi`; when `config_dir` is under `test-configs/`, also `test_config_path` e.g. `test1/qwen3.5-9b-q8/baseline/` plus `test_name`, `model`, and `test_config` from the first three path segments; each field is `null` when unavailable)
- `metrics` (see table below)

Stdout prints a rounded subset (`server_ready_s`, `wall_time_s`, `ttft_s`, token counts, `prefill_tok_s`, `decode_tok_s`, `idle_vram_mb`, `peak_vram_mb`). The JSON `metrics` object keeps full floating-point values for every field.


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
- **Clean stop:** Use Ctrl+C in the terminal for a controlled interrupt. The runner records `status: error` and still writes JSON when possible. Killing the process from outside (or a hard external timeout) may leave `llama-server` running in the background.
- **No model flag in args:** Putting `-m` or `--model` in `server.yaml` `args` is rejected; the runner appends `-m` with the `model` path.
- **Removed `label` field:** `server.yaml` must not contain `label`; use a config directory so the result slug reflects the variant path.
