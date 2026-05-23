# Local Model Testing (`llama.cpp`)
First forays into local model usage, this is about tuning and understanding the most efficient models and configurations for the llama.cpp server.

## Setup
Relatively modest for AI work, but it's surprisingly viable. 
Running on a WSL2 instance on a Windows host with 32Gb sys RAM (24Gb allocated to WSL) and a 5080 with 16Gb VRAM...

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

local-model-tests/$ free -h
               total        used        free      shared  buff/cache   available
Mem:            25Gi       1.4Gi        10Gi       3.2Mi        13Gi        23Gi
Swap:           64Gi       351Mi        63Gi
```

## Dependencies
```bash
./wsl-builder.sh dev-python python3
./wsl-builder.sh ai cuda132,llama-cpp,huggingface-cli
```

## Models
Download a model from Hugging Face...

```bash
hf download bartowski/Qwen_Qwen3.5-9B-GGUF Qwen_Qwen3.5-9B-Q8_0.gguf
```
Note: if you omit the 2nd argument the whole repo is downloaded (all variants of the model - normally huge!)

### MoE Models
* [Gemma-4-E4B-IT-Q8](https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF/blob/main/gemma-4-E4B-it-Q8_0.gguf)
  * Released: April 2026, outstanding for local developer setups that require tool-calling capabilities and structured outputs.
  * Generation throughput: ~114 tok/s
  * Estimated Max Context: 130028 tokens
  * Model Max Context: 131072 tokens
  * Model VRAM: 5781 MiB
  * KV VRAM: 10422 MiB
  * GGUF on disk: 8.03 GB
* [Qwen3.6-35B-A3B-UD-Q4-K-XL](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/blob/main/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf)
  * Released: April 2026
  * Setup: `--n-gpu-layers 999 --n-cpu-moe 24` to fit on card and leave 5GB for KV cache.
  * Generation throughput: ~47 tok/s
  * Estimated Max Context: 69409 tokens
  * Model Max Context: 262144 tokens
  * Model VRAM: 10869 MiB
  * KV VRAM: 5334 MiB
  * GGUF on disk: 22.36 GB

### Dense Models
* [Qwen3.5-9B-Q8](https://huggingface.co/bartowski/Qwen_Qwen3.5-9B-GGUF/blob/main/Qwen_Qwen3.5-9B-Q8_0.gguf)
  * Released: March 2026, broad multilingual support (100+ languages) and highly competitive, generalist benchmark scores.
  * Generation throughput: ~86 tok/s
  * Estimated Max Context: 79443 tokens
  * Model Max Context: 262144 tokens
  * Model VRAM: 9001 MiB
  * KV VRAM: 7202 MiB
  * GGUF on disk: 9.55 GB
* [Qwen3.6-27B-Q4](https://huggingface.co/unsloth/Qwen3.6-27B-GGUF/blob/main/Qwen3.6-27B-Q4_0.gguf)
  * Released: April 2026, barely fits in 16Gb, no room for context, but I wanted to test it, so I did!
  * Generation throughput: ~46 tok/s
  * Estimated Max Context: 9176 tokens
  * Model Max Context: 262144 tokens
  * Model VRAM: 15584 MiB
  * KV VRAM: 619 MiB
  * GGUF on disk: 15.79 GB


## Tester v1
* Calibration tests are used to test overall throughput and rough expected max context window given a 16Gb card.
* See [tester-v1/README.md](tester-v1/README.md) for more details and requirements.

### Run a server

```bash
cd tester-v1
./single_test_runner.py configs/qwen3.5-9b-q8/bench/ --test-server
```

Use a browser to access the llama web UI while it is running (port depends on `server.yaml` but is typically 8080):

[https://localhost:8080](https://localhost:8080)

### Calibration Probe Runs

```bash
# standard dense model
./model_calibration.py configs/qwen3.5-9b-q8

# for MoE models, define how many experts you want to lay off the CPU (to fit large models on small cards)...
./model_calibration.py configs/qwen3.6-35b-a3b-ud-q4-k-xl/ --n-cpu-moe 24
```

* Each model has `model.yaml` under [tester-v1/configs/{model}/](tester-v1/configs/).
* The test probe `client.yaml` & `server.yaml`'s are under [tester-v1/configs/reference/](tester-v1/configs/reference/). 
* For MoE models, pass `--n-cpu-moe N` on calibration (also works for single_test_runner)
* Add `--save-result` to save detailed output JSON to `tester-v1/results/{model}/`.

## Key Learnings

### MoE vs dense

Model filenames often encode the architecture:

* **Dense** (e.g. `Qwen3.5-9B`): every parameter runs on every token. Simpler to load; VRAM scales with total size.
* **MoE** (e.g. `35B-A3B`): many experts, but only a few fire per token (~3B active here). Delivers near-large-model quality at small-model speed, but you may need `--n-cpu-moe` to split expert weights between GPU and CPU on a 16Gb card.

### Reading quant names

Quantization shrinks weights so bigger models fit on consumer GPUs. Names look cryptic; they are mostly `{prefix}-{family}_{tier}`.

**Bit width (the number):**

* `FP16` / `BF16`: near full quality; huge files. Use when VRAM is not a constraint and you want baseline fidelity.
* `Q8`: ~8 bits per weight; often hard to tell from full precision. Best quality-to-size ratio when you have headroom.
* `Q4`: ~4 bits; answers can slip on hard tasks. The usual tradeoff for fitting 20B+ models locally.
* `Q2`: ~2 bits; quality drops sharply and varies by model. Last resort when nothing else fits.

**Prefix:**

* `UD-` (Unsloth Dynamic): mixed precision per layer, tuned with calibration data. Better chat/coding quality at the same nominal Q4 size, at the cost of slightly slower inference.

**Family:**

* `Q4_K`: standard llama.cpp K-quants; mixed block sizes inside the file. Predictable, well-tested 4-bit format.
* `IQ4`: importance quants; lean harder on calibration to preserve quality at lower size. Smallest files in the ~4-bit class (e.g. `IQ4_XS` ~18 GB vs `Q4_K_XL` ~23 GB on the same model).
* `MXFP4_MOE`: microscaling FP4 aimed at MoE expert weights. Tuned for sparse expert layers rather than uniform `Q4_K` blocks.

**Tier suffix (`S` / `M` / `L` / `XL` / `XS` / `NL`):**

* `S` (small): most compressed in that family. Saves disk and VRAM but expect more quality loss.
* `M` (medium): balanced default and a safe general-purpose pick when you are unsure.
* `L` (large): less compression than `S`/`M`; a step up in quality within the same Q4 family.
* `XL`: not "extra large file" but a smart mix that keeps sensitive tensors at `Q5`/`Q6` while the rest stays `Q4`. Best quality in the Q4 class; Unsloth's usual recommendation over plain `Q4_K_M`.
* `XS` (I-quants only): extra-small; most aggressive IQ compression. Maximum headroom for context on tight VRAM.
* `NL` (I-quants only): non-linear dequant scheme; slightly larger than `XS`. Often a better speed/quality tradeoff than `XS` at similar size.

**Other:**

* `TurboQuant`: newer KV-cache compression (`turbo3` / `turbo4`), not the weight file itself. Stretches context without re-downloading a different GGUF.

Example: `Qwen3.6-35B-A3B-UD-Q4_K_XL` = MoE model, Unsloth Dynamic mixed Q4 quant, XL tier (best Q4 quality).

### Multi-Token Prediction (MTP)

Some repos are labelled **MTP-GGUF** (e.g. `unsloth/Qwen3.6-35B-A3B-MTP-GGUF`). The file includes extra prediction heads baked into the model.

Why: llama.cpp can draft several tokens ahead and verify them in one pass, giving roughly **1.5-2x faster generation** with no accuracy loss when enabled.

Requires an MTP GGUF and server flags:

```bash
--spec-type draft-mtp --spec-draft-n-max 2
```

Without those flags you carry the extra weights but get no speed benefit.

### Server Config
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
