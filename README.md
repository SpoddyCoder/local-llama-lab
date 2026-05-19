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

## Models

Download a model from Hugging Face...

```bash
hf download bartowski/Qwen_Qwen3.5-9B-GGUF Qwen_Qwen3.5-9B-Q8_0.gguf
```

Note: this example shows just one quantized model, if you omit the 2nd argument the whole repo is downloaded (all variants of the model - normally huge!)

* [Qwen3.6-27B-Q4](https://huggingface.co/unsloth/Qwen3.6-27B-GGUF/tree/main)
  * Cutting edge [description]
  * Dense model
  * Setup:
    * Doesn't really fit tbh, stressed up to 8192 tokens before it crashes.
  * Performance:
    * Generation throughput: ~27 tok/s
    * Estimated Max Context: 8192 tokens
    * Model Max Context: TODO tokens
    * Model VRAM: 15.1 GiB
    * KV VRAM: Too small to measure accurately
    * GGUF on disk: TODO GB
* [Qwen3.5-9B-Q8](https://huggingface.co/bartowski/Qwen_Qwen3.5-9B-GGUF/tree/main)
  * Broad multilingual support (100+ languages) and highly competitive, generalist benchmark scores.
  * Dense model
  * Setup:
    * Vanilla, model fits easily into VRAM
  * Performance:
    * Generation throughput: ~91 tok/s
    * Estimated Max Context: 150823 tokens
    * Model Max Context: 262144 tokens
    * Model VRAM: 10337 MiB
    * KV VRAM: 4430 MiB
    * GGUF on disk: 9.55 GB
* [Gemma-4-E4B-IT-Q8](https://huggingface.co/google/gemma-4-e4b-it-gguf/tree/main)
  * MoE model
  * Outstanding for local developer setups that require tool-calling capabilities and structured outputs.
  * Setup:
    * Vanilla, model fits easily into VRAM
  * Performance:
    * Generation throughput: ~114 tok/s
    * Estimated Max Context: 446749 tokens
    * Model Max Context: 131072 tokens
    * Model VRAM: 7022 MiB
    * KV VRAM: 7745 MiB
    * GGUF on disk: 8.03 GB

## Tester v1 (single-run harness)

From the repo root, use the harness under `tester-v1/`. Python 3 is required. Check third-party deps (skip install if this prints `deps ok`):

```bash
cd tester-v1
python3 -c "import httpx, yaml" 2>/dev/null && echo "deps ok" || echo "need install"
```

If you see `need install`, use a venv (avoids PEP 668 errors on system `pip`):

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

See [tester-v1/README.md](tester-v1/README.md). Saved probe JSON lives under [tester-v1/results/{model}/{variant}/](tester-v1/results/); see that README for [results](tester-v1/README.md#results) and [calibration-sessions](tester-v1/README.md#model-calibration).

### Run a server

Start a server using one of the variant configs (`client.yaml` is unused in this mode):

```bash
cd tester-v1
./single_test_runner.py configs/qwen3.5-9b-q8/hello-world-baseline --test-server
```

Use a browser to view the llama web UI while it is running (port depends on `server.yaml`):

[https://localhost:8080](https://localhost:8080)

### Probe runs

Configuratuion pairs (`server.yaml` and `client.yaml`) live under [tester-v1/configs/](tester-v1/configs/). 

For a new model, run calibration first (default: probe stdout plus four-line summary; no JSON):

```bash
cd tester-v1
./model_calibration.py configs/qwen3.5-9b-q8
./model_calibration.py configs/gemma-4-e4b-it-q8
```

Add `--save-result` on calibration or single runs when you want probe JSON under `tester-v1/results/{model}/{variant}/` (calibration also writes `calibration-sessions/{session_id}.json`).

Then run a probe variant (default: metrics on stdout, no JSON file):

```bash
./single_test_runner.py configs/qwen3.5-9b-q8/hello-world-baseline
./single_test_runner.py configs/gemma-4-e4b-it-q8/hello-world-baseline
```

To keep a JSON artifact under `tester-v1/results/{model}/{variant}/`, add `--save-result`.

### Tests
* [hello-world.md](hello-world.md) - simple smoke prompt.

## Key Learnings
* Quantization is important for squeezing larger models into a consumer graphics card:
  * FP16 - near full quality, considered lossless, huge
  * Q8 - considered relatively lossless, >2x smaller
  * Q4 - quality reduced, answers slip, >4x smaller
  * Q2 - quality significantly reduced, answers slip more, very model dependant how they react to this much quantization, >8x smaller
  * TurboQuant - relatively new way to quantize with minimal quality loss.
  * _K_XL models - use mixed quants and can be slightly better that the same standard quantized model
    * The difference is more noticeable at greater quantizations, ie Q4_K_XL vs Q4_0 will often be a much bigger gain than Q8_K_XL vs Q8_0
* Server Flags
  * `--fit off` - disable llama.cpp auto VRAM fitting on load (on by default).
    * Fit can shrink context or move layers to CPU to avoid OOM; on large models that often costs a lot of tok/s.
    * Use when you already set `--ctx-size`.
  * `--no-mmap` - force preload of model immediately into memory, to avoid disk reads during usage
    * Negatively affects startup time tho.
  * `--n-gpu-layers 999 --n-cpu-moe 41` - use with MoE models, put the small fast firing stuff on gpu and the bulky experts on cpu
    * Tune 42 down to use more gpu vram (more experts on vram)
    * Any VRAM not used by the model is used by the KV cache (context length), so you should wnat to leave 1-4Gb free.
  * `--cache-type-k turbo4 --cache-type-v turbo3` - use turbo4 for cache keys and turbo3 for cache values (TurboQuant).
    * Asymmetry can be useful if the model uses grouped query attention (8:1 ratio on qwen3.6) which means the keys can take heavier compression than the values.
    * Doesn't appear to be available in the WSL fork of llama.cpp yet
  * `--ngl 20` - first 20 layers go on GPU, rest on CPU (not fast! but useful for testing)
