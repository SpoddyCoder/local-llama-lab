# Tester v1 - Three-Phase Implementation Plan

Local model benchmark harness for `llama-server` (`llama.cpp`). Goal: run one configured experiment at a time, inspect results, tune server settings (model, quant, flags), then repeat without sweep automation until the workflow feels right.

## Status (2026-05-16)

**Phase 1 is implemented and usable.** The default command runs the full loop (server -> stream -> metrics on stdout -> teardown). Pass `--save-result` to also write JSON under `results/`.

```bash
cd tester-v1
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python single_test_runner.py
```

Dev mode: `--test-server` (hold until Ctrl+C). Unit tests: `python3 -m unittest discover -s tests -v`.

### Phase 1 checklist

| # | Item | Status |
|---|------|--------|
| 1 | `tester-v1/` tree and `requirements.txt` | Done |
| 2 | YAML loaders + validation + `model_slug` (`src/config.py`) | Done |
| 3 | Server subprocess + health poll + teardown (`src/server.py`) | Done |
| 4 | Streaming client + metrics (`src/client.py`, `src/metrics.py`) | Done |
| 5 | Result writer + filename generation (`src/results.py`) | Done |
| 6 | Stdout summary + CLI `--server` / `--client` (`single_test_runner.py`) | Done |
| 7 | Example `server.yaml` / `client.yaml` (README Test 1 baseline) | Done |
| 8 | Run instructions in root `README.md` | Done |

Orchestration lives in `src/runner.py`; `single_test_runner.py` at repo root is the CLI entrypoint.

### Phase 1 acceptance criteria

| Criterion | Status | Notes |
|-----------|--------|-------|
| Single command runs Qwen Q8_0 + Test 1 without manual server | Verified | Live runs ~12-24s after model load |
| Server not left running after exit | Verified | Teardown via `managed_server()` + signal handler |
| Result JSON with embedded configs + metrics | Verified | e.g. `results/20260516T142121Z_Qwen_Qwen3.5-9B-Q8_0.json` |
| Stdout summary matches JSON metrics | Verified | Rounded on stdout; full precision in JSON |
| Re-run after editing `server.yaml` -> new timestamped file | Verified | Two baseline JSONs differ by timestamp (`141819Z` vs `142121Z`); `make_run_id` embeds UTC time; unit tests pass |
| TTFT and token fields on successful stream | Verified | Uses `timings` fallback when `usage` absent (see below) |

### Phase 1 - remaining (optional polish)

Complete (2026-05-16): README baseline result, `server.yaml` `label`, re-run criterion verified, acceptance table updated.

### Verified on this machine (WSL + llama-server)

- **Health:** `GET http://127.0.0.1:8080/health` returns 200 (used as ready signal; `/v1/models` is fallback)
- **SSE tokens:** This build sends no `usage` object; final chunk has `timings.prompt_n` / `timings.predicted_n`
- **TTFT:** First non-empty delta may be `reasoning_content`, not `content` (Qwen 3.5)
- **Cold start:** First load of 9B Q8 can take minutes before `/health` succeeds; use Ctrl+C for clean teardown (external `timeout` may leave `llama-server` running)

---

## Decisions (locked for v1)

| Topic | Choice |
|--------|--------|
| Scope | Single run per script invocation; manual config edits between runs |
| Server lifecycle | Script starts server, waits until healthy, runs client, **always tears down** server (success, failure, Ctrl+C where practical) |
| Config format | **YAML** for both server and client configs |
| Server config shape | **Structured:** `model` path + `args` list (not a raw command string) |
| API | OpenAI-compatible `/v1/chat/completions` with `stream: true` (required for TTFT) |
| Results | With `--save-result`: `results/{timestamp}_{model_slug}.json`; JSON includes **copies of server + client config** used for the run |
| Output | Default: metrics (+ preview) on stdout. With `--save-result`: JSON file + run summary on stdout (or stderr one-liner with `--quiet`) |

---

## Target layout (Phase 1 - as built)

```text
local-model-tests/
|-- docs/
|   `-- tester-v1-implementation-plan.md
|-- tester-v1/
|   |-- server.yaml
|   |-- client.yaml
|   |-- requirements.txt
|   |-- src/
|   |   |-- config.py
|   |   |-- server.py
|   |   |-- client.py
|   |   |-- metrics.py
|   |   |-- results.py
|   |   |-- runner.py            # single-run orchestration
|   |   `-- calibration_cli.py
|   |-- single_test_runner.py   # CLI entrypoint
|   |-- model_calibration.py    # calibration CLI entrypoint
|   |-- tests/
|   |   |-- test_client_sse.py
|   |   `-- test_results.py
|   `-- results/               # *.json gitignored; .gitkeep tracked
|-- .gitignore
`-- README.md
```

Invoke from `tester-v1/`: `python single_test_runner.py`. Packaging (`pyproject.toml`, `python -m`) deferred.

---

## Phase 1 - Runnable single-run loop (design spec)

Implemented; see **Status** at the top for checklist, acceptance, and remaining polish.

### Objective

From `tester-v1/`, one command:

1. Load `server.yaml` and `client.yaml`
2. Start `llama-server` with `model` + `args`
3. Poll until the HTTP API is ready (or timeout with clear error)
4. Send one streaming chat completion
5. Print metrics (and completion preview) to stdout
6. Stop the server process (and process group if spawned in a new session)

With `--save-result`, step 5 becomes: record metrics + embedded configs -> `results/<timestamp>_<model_slug>.json`, then print run summary (or stderr one-liner with `--quiet`).

### Module boundaries (minimal, growth-friendly)

Keep logic in small units under `src/` (single file acceptable initially if functions are grouped by responsibility):

| Module / area | Responsibility |
|---------------|----------------|
| `config` | Load and validate YAML; resolve paths (`~` expansion); derive `model_slug` from model basename |
| `server` | Build argv: `llama-server` + `args` + `-m <model>`; subprocess start; health check; graceful shutdown |
| `client` | Streaming HTTP to `/v1/chat/completions`; TTFT + wall clock; parse usage from stream |
| `metrics` | Assemble metric dict from client timings + token counts |
| `results` | Build result document; write JSON; format stdout summary |
| `runner` | Orchestrate: load configs -> server -> client -> results -> teardown |

Future phases add modules (e.g. `vram`, `sweep`) without changing the single-run orchestration contract.

### Config schemas

#### `server.yaml`

```yaml
# Path to .gguf (supports ~)
model: ~/.cache/huggingface/hub/.../Qwen_Qwen3.5-9B-Q8_0.gguf

# llama-server flags EXCLUDING -m (added by runner)
args:
  - --host
  - "0.0.0.0"
  - --port
  - "8080"

# Optional: override binary name/path (default: llama-server on PATH)
# binary: llama-server

# Health check tuning
ready_timeout_s: 120
ready_poll_interval_s: 0.5
```

`label` in `server.yaml` was removed; result filenames use path-derived slugs when a config directory is passed (see [tester-v1/README.md](../tester-v1/README.md)).

Validation (Phase 1):

- `model` required; file should exist (warn or fail; recommend **fail fast**)
- `args` must not contain `-m` / `--model` (runner owns model injection)
- Port deduced from `args` if present (`--port` / `-p`); used for health URL

#### `client.yaml`

```yaml
# Base URL without trailing path (default: http://127.0.0.1:8080)
base_url: http://127.0.0.1:8080

# Chat messages (OpenAI format)
messages:
  - role: user
    content: |
      hi, write me hello world in 20 different programming languages.

# Passed through to API (stream forced true by runner)
params:
  max_tokens: 1024
  temperature: 0

# Request timeout (entire stream)
timeout_s: 600
```

Validation:

- `messages` non-empty
- `params` may omit `stream` (runner sets `stream: true`)

### Server lifecycle

1. **Build command:** `[binary, *args, "-m", resolved_model_path]`
2. **Start:** `subprocess.Popen` with `start_new_session=True` (or equivalent) so teardown can signal the whole group
3. **Ready check:** GET `{base_url}/health` or `{base_url}/v1/models` (whichever `llama-server` exposes; implementer verifies against installed build; prefer lightweight health if available)
4. **On ready failure:** log stderr tail if captured; exit non-zero; ensure process group killed
5. **Teardown:** `SIGTERM` -> wait up to N seconds -> `SIGKILL` if needed; run in `finally` block

**Ctrl+C:** register handler that triggers same teardown before exit.

### Client - streaming and metrics

**Endpoint:** `POST {base_url}/v1/chat/completions`

**Body:** `messages` + `params` + `"stream": true`

**Measurements:**

| Field | Source |
|--------|--------|
| `wall_time_s` | Monotonic clock: request start -> last stream chunk processed |
| `ttft_s` | Monotonic clock: request start -> first non-empty text in `choices[0].delta` (`content` or `reasoning_content`) |
| `prompt_tokens` | Final `usage.prompt_tokens` from stream (or last chunk that includes `usage`) |
| `completion_tokens` | `usage.completion_tokens` |
| `total_tokens` | `usage.total_tokens` or sum of above |
| `completion_time_s` | `wall_time_s - ttft_s` (approximate generation window) |
| `prefill_tok_s` | `prompt_tokens / ttft_s` if `ttft_s > 0` and tokens known |
| `decode_tok_s` | `completion_tokens / completion_time_s` if completion window > 0 |
| `tokens_per_second` | `completion_tokens / wall_time_s` (overall throughput; document definition in JSON) |

Use `httpx` with `stream=True` and SSE line parsing (`data: {...}`). Handle `[DONE]` sentinel.

**Implemented fallbacks** for this `llama-server` build:

- Token counts: prefer OpenAI `usage` on any chunk; else final chunk `timings.prompt_n` / `timings.predicted_n`
- No text-based token estimation

**Phase 1 deferral:** `peak_vram_mb` -> `null` in JSON (field present for schema stability).

### Result JSON document

**Path:** `results/{timestamp}_{model_slug}.json`

- `timestamp`: ISO 8601 local or UTC (pick one, document in README; recommend UTC `YYYYMMDDTHHMMSSZ`)
- `model_slug`: sanitized basename of model file (e.g. `Qwen_Qwen3.5-9B-Q8_0`); override with `server.label` if set (sanitized for filesystem)

**Top-level shape:**

```json
{
  "run_id": "20260516T134500Z_Qwen_Qwen3.5-9B-Q8_0",
  "started_at": "2026-05-16T13:45:00Z",
  "finished_at": "2026-05-16T13:47:12Z",
  "server_config": { },
  "client_config": { },
  "metrics": {
    "wall_time_s": 132.4,
    "ttft_s": 0.85,
    "completion_time_s": 131.55,
    "prompt_tokens": 42,
    "completion_tokens": 980,
    "total_tokens": 1022,
    "prefill_tok_s": 49.4,
    "decode_tok_s": 7.5,
    "tokens_per_second": 7.4,
    "peak_vram_mb": null
  },
  "status": "ok",
  "error": null,
  "notes": null
}
```

- `server_config` / `client_config`: exact parsed YAML as JSON-serializable dicts (after path resolution, store resolved `model` path)
- On failure: `status: "error"`, `error` message string, partial `metrics` if any

### Stdout summary

After a successful run, print a fixed, scannable block (example):

```text
Run: 20260516T134500Z_Qwen_Qwen3.5-9B-Q8_0
Model: .../Qwen_Qwen3.5-9B-Q8_0.gguf
Result: results/20260516T134500Z_Qwen_Qwen3.5-9B-Q8_0.json

wall_time_s       132.40
ttft_s              0.85
prompt_tokens        42
completion_tokens   980
prefill_tok_s       49.4
decode_tok_s         7.5
```

On error: print error and path to partial result if written.

### Dependencies

```text
httpx>=0.27
pyyaml>=6.0
```

Python **3.11+** recommended. `venv` + `requirements.txt` in `tester-v1/`.

### Gitignore

Done in repo root `.gitignore`: `tester-v1/results/*.json`, ignore other files under `results/` except `.gitkeep`; `__pycache__/`, `.venv/`.

### Phase 1 design reference

The sections above (objective through stdout summary) remain the **design spec** for Phase 1. Implementation status and checklist are at the top of this document.

---

## Phase 2 - Harness quality (high level)

Improve trust and ergonomics of single runs without adding sweep automation.

- **VRAM:** background poll `nvidia-smi` during run; set `peak_vram_mb` in metrics
- **Warmup:** optional `warmup: true` in client config: one discarded streaming request before measured run
- **Robustness:** clearer errors (server stderr snippet, connection refused, timeout); non-zero exit codes
- **Metadata:** optional fields in result JSON: `llama-server --version`, GPU name from `nvidia-smi`, driver version
- **Config UX:** optional `notes` in server or client YAML copied into result; config JSON schema comments in examples
- **Stdout:** optional snippet of first N chars of completion for sanity check (off by default)
- **README:** "how to compare two JSON files" (manual diff tips)

---

## Phase 3 - Scale and comparison (high level)

Add automation only when manual single runs feel sufficient.

- **Sweep runner:** expand server and/or client configs into lists/matrix; run serially with same lifecycle as Phase 1
- **CSV export:** append rows matching README log columns for spreadsheet analysis
- **Compare command:** load two result JSON files (or latest two in `results/`) and print delta table on stdout
- **Presets:** named server profiles (`fast`, `quality`, `low_vram`) as separate YAML files
- **CI / regression (optional):** threshold checks on `decode_tok_s` for a pinned model

Phase 3 must not remove the Phase 1 single-run path; sweeps wrap the same `server` -> `client` -> `results` core.

---

## Relationship to README Test 1

README defines the first real experiment (Qwen 9B Q8_0 baseline, fixed prompt, `max_tokens: 1024`, `temperature: 0`). `tester-v1/server.yaml` and `client.yaml` match that setup.

**Done:** README Test 1 baseline links `tester-v1/results/20260516T142121Z_Qwen_Qwen3.5-9B-Q8_0.json` with rounded metrics; `server.yaml` sets `label: qwen3.5-9b-q8-baseline` for shorter future result filenames.

Longer-term README log columns (`model`, `quant`, `server_flags`, ...) map cleanly to Phase 3 CSV export; Phase 1 JSON already carries full configs for manual inspection.
