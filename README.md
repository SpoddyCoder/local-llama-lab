# Local Model Testing (`llama.cpp`)
First forays into local model usage, this is about tuning and understanding the most efficient models and configurations for the llama.cpp server.

## Key Highlights / Conclusions
* Quantization is important for squeezing things into a consumer graphics card:
  * FP16 - near full quality, considered lossless, huge
  * Q8 considered relatively lossless, >2x smaller
  * Q4 quality reduced, answers slip, >4x smaller
  * Q2 quality significantly reduced, answers slip more, very model dependant how they react to this much quantization, >8x smaller
  * TurboQuant is relatively new way to quantize with minimal quality loss.
* Performance: Server Flags...
  * `--no-mmap` - force preload of model immediately into memory, to avoid disk reads during usage
    * Negatively affects startup time tho.
* Limited VRAM: Server Flags to help...
  * `--ngl 20` - first 20 layers go on GPU, rest on CPU (not fast! but useful for testing)
  * `--n-gpu-layers 999 --n-cpu-moe 41` - use MoE models, put the small fast firing stuff on gpu and the bulky experts on cpu
    * Tune 42 down to use more gpu vram (more experts on vram)
    * Any VRAM not used by the model is used by the KV cache (context length), so you should wnat to leave 1-4Gb free.
  * `--cache-type-k turbo4 --cache-type-v turbo3` - use turbo4 for cache keys and turbo3 for cache values (TurboQuant).
    * Asymmetry can be useful if the model uses grouped query attention (8:1 ratio on qwen3.6) which means the keys can take heavier compression than the values.
  * ``

### Further Things to Explore
* Optimisation for fitting large models too big for the GPU VRAM...
  * Use MoE models (not dense models), put small experts on gpus


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

Note: this example shows just one quantized model, if yu omit the 2nd argumanet the whole repo is downloaded (all variants of the model - normally huge!)

* [Qwen_Qwen3.5-9B-Q8_0.gguf](https://huggingface.co/bartowski/Qwen_Qwen3.5-9B-GGUF/tree/main)
  * Broad multilingual support (100+ languages) and highly competitive, generalist benchmark scores.
  * Vanilla setup:
    * 9.55 GB on disk
    * 10337 MiB model VRAM
    * 4430 MiB KV VRAM (assuming 16Gb card)
    * Estimated Max Context: 150823 tokens
* [Gemma 4: E4B-IT-Q8](https://huggingface.co/google/gemma-4-e4b-it-gguf/tree/main)
  * Outstanding for local developer setups that require tool-calling capabilities and structured outputs.
  * Vanilla setup:
    * 8.03 GB on disk
    * 7022 MiB model VRAM
    * 7745 MiB KV VRAM (assuming 16Gb card)
    * Estimated Max Context: 446749 tokens

## Tester v1 (single-run harness)

From the repo root, use the harness under `tester-v1/`:

```bash
cd tester-v1
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

See [tester-v1/README.md](tester-v1/README.md). Results for various cases are stored in [tester-v1/results/](tester-v1/results/)

## Tests
For a new model, run the calibration tests first...

```bash
python3 model_calibration.py test-configs/test1/qwen3.5-9b-q8
```

Then run tests cases...

```bash
python3 single_test_runner.py test-configs/test1/qwen3.5-9b-q8/baseline/
```

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

* GGUF on disk: 9.55 GB
* Model VRAM: 10337 MiB
* KV VRAM: 4430 MiB
* Estimated Max Context: 150823 tokens

##### `baseline` (no switches)
```
server_ready_s    3.12
wall_time_s       11.66
ttft_s            0.18
prompt_tokens     24
completion_tokens 1024
prefill_tok_s     133.26
decode_tok_s      89.23
idle_vram_mb      15134
peak_vram_mb      15222
```

##### `--no-mmap`
```
server_ready_s    21.34
wall_time_s       11.56
ttft_s            0.25
prompt_tokens     24
completion_tokens 1024
prefill_tok_s     95.35
decode_tok_s      90.54
idle_vram_mb      15133
peak_vram_mb      15203
```

#### `Gemma 4: E4B-IT-Q8_0.gguf`

* GGUF on disk: 8.03 GB
* Model VRAM: 7022 MiB
* KV VRAM: 7745 MiB
* Estimated Max Context: 446749 tokens

##### `baseline` (no switches)
```
server_ready_s    2.58
wall_time_s       8.66
ttft_s            0.28
prompt_tokens     30
completion_tokens 1024
prefill_tok_s     106.54
decode_tok_s      122.19
idle_vram_mb      9035
peak_vram_mb      9107
```

##### `--no-mmap`
```
server_ready_s    4.58
wall_time_s       8.61
ttft_s            0.24
prompt_tokens     30
completion_tokens 1024
prefill_tok_s     124.71
decode_tok_s      122.40
idle_vram_mb      9054
peak_vram_mb      9126
```