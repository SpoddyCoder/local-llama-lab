# Local Model Testing (`llama.cpp`)
First forays into local model usage, this is about tuning and understanding the most efficient models and configurations for the llama.cpp server.


## Setup
Running on a WSL2 instance on Windows machine...

```bash
local-model-tests$ nvidia-smi
Sat May 16 13:40:46 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 595.71.05              Driver Version: 596.49         CUDA Version: 13.2     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 5080        On  |   00000000:01:00.0  On |                  N/A |
|  0%   44C    P5             29W /  378W |    1054MiB /  16303MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
```

## Dependencies

```bash
./wsl-builder dev-python python3
./wsl-builder.sh ai cuda132,llama-cpp,huggingface-cli
```

## Tester v1 (single-run harness)

From the repo root, use the Phase 1 harness under `tester-v1/`:

```bash
cd tester-v1
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/python_runner.py
```

That loads `server.yaml` and `client.yaml`, starts `llama-server`, runs one streaming completion, writes `results/{timestamp}_{model_slug}.json`, prints a metrics summary, and tears down the server. See `docs/tester-v1-implementation-plan.md` for config shape and result JSON fields.

## Tests

### Test 1 - First Steps
Simple smoke test prompt...

```text
hi, write me hello world in 20 different programming languages.
```

Test Script Config:
```
"max_tokens": 1024,
"temperature": 0
```

#### `Qwen_Qwen3.5-9B-Q8_0.gguf`

##### Baseline - no switches
Result ([tester-v1/results/20260516T142121Z_Qwen_Qwen3.5-9B-Q8_0.json](tester-v1/results/20260516T142121Z_Qwen_Qwen3.5-9B-Q8_0.json); run before `server.yaml` `label` was added):

```
Run: 20260516T142121Z_Qwen_Qwen3.5-9B-Q8_0
Result: tester-v1/results/20260516T142121Z_Qwen_Qwen3.5-9B-Q8_0.json

wall_time_s       11.71
ttft_s             0.20
prompt_tokens         24
completion_tokens   1024
prefill_tok_s      117.70
decode_tok_s        89.02
tokens_per_second   87.47
```

Full server and client config for this run are in the JSON (`server_config`, `client_config`). New runs use `server.yaml` `label: qwen3.5-9b-q8-baseline`, so result filenames become `{timestamp}_qwen3.5-9b-q8-baseline.json` instead of the model stem slug; existing JSON names are unchanged.