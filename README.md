# Local Model Testing (`llama.cpp`)

First forays into local model usage, this is about tuning and understanding the most efficient models and configurations for the llama.cpp server.

System is relatively modest for AI work, but surprisingly viable for cutting edge open-source models...

- WSL2 instance on a Windows host
- 32Gb System RAM, 26Gb allocated to WSL
- 5080 with 16Gb VRAM
- See [system setup](#system-setup) for more details.

## Dependencies

```bash
./wsl-builder.sh dev-python python3
./wsl-builder.sh ai cuda132, llama-cpp, huggingface-cli
```

## Models

| Model                                                                                                                                                                                                                 | Throughput · context                                                    | Memory · setup                                                                                             |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| [Qwen3.6-27B-Q4](https://huggingface.co/unsloth/Qwen3.6-27B-GGUF/blob/main/Qwen3.6-27B-Q4_0.gguf)<br>Dense 27B<br>Strong coding/reasoning; not usable on 16 GB<br>LiveCodeBench: **83.9%**<br>April 2026                             | Throughput: **~46 tok/s**<br>Est. context: **9K**<br>Max context: **262K**    | Model VRAM: **15.22 GB**<br>KV VRAM: **0.60 GB**<br>Disk: **15.79 GB**                                           |
| [Qwen3.5-9B-Q8](https://huggingface.co/bartowski/Qwen_Qwen3.5-9B-GGUF/blob/main/Qwen_Qwen3.5-9B-Q8_0.gguf)<br>Dense 9B<br>201 languages, multimodal<br>LiveCodeBench: **82.7%**<br>March 2026                            | Throughput: **~86 tok/s**<br>Est. context: **79K**<br>Max context: **262K**   | Model VRAM: **8.79 GB**<br>KV VRAM: **7.03 GB**<br>Disk: **9.55 GB**                                             |
| [Qwen3.6-35B-A3B-UD-Q4-K-XL](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/blob/main/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf)<br>MoE (35B / 3B active)<br>Agentic coding leader<br>LiveCodeBench: **80.4%**<br>April 2026 | Throughput: **~52 tok/s**<br>Est. context: **69K**<br>Max context: **262K**   | Model VRAM: **10.50 GB**<br>System RAM: **11.97 GB**<br>KV VRAM: **5.33 GB**<br>Disk: **22.36 GB**<br>`--n-cpu-moe 24` |
| [Gemma-4-E4B-IT-Q8](https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF/blob/main/gemma-4-E4B-it-Q8_0.gguf)<br>Effective 4.5B (E4B)<br>Multimodal, tool-calling<br>LiveCodeBench: **52.0%**<br>April 2026                            | Throughput: **~114 tok/s**<br>Est. context: **130K**<br>Max context: **131K** | Model VRAM: **5.65 GB**<br>KV VRAM: **10.18 GB**<br>Disk: **8.03 GB**                                            |
| [Qwen3-Coder-30B-A3B-UD-Q5-K-XL](https://huggingface.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF/blob/main/Qwen3-Coder-30B-A3B-Instruct-UD-Q5_K_XL.gguf)<br>MoE (30B / 3B active)<br>Agentic coding<br>LiveCodeBench: **TODO** | Throughput: **~39 tok/s**<br>Est. context: **34K**<br>Max context: **262K** | Model VRAM: **11.78 GB**<br>System RAM: **10.11 GB**<br>KV VRAM: **4.05 GB**<br>Disk: **21.74 GB**<br>`--n-cpu-moe 24` |

* LiveCodeBench v6 scores are from official vendor model cards ([LiveCodeBench](https://livecodebench.github.io/)): base BF16 weights, not a local GGUF run. 
* For coding, local quants are usually close:
  * Q8 and Q6 within about 1-3% of BF16
  * Q4_K_XL and UD-Q4 within about 3-8% (ranking tends to hold; absolute numbers shift). 
  * See [gguf-bench quant curves](https://gguf-bench.com/) for llama-server GGUF scores by quant level.

Download a model from Hugging Face using its config under [models/](models/):

```bash
./download_model.py models/qwen3.5-9b-q8/
./download_model.py models/qwen3.5-9b-q8/ --dry-run
```

* Each model's [model.yaml](models/qwen3.5-9b-q8/model.yaml) has a `hf-download` block with 
* `./download_model.py` reads the `repo` and `file` keys from the `model.yaml`, runs `hf download` and updates yaml if necessary.

---

## Probe

* Calibration tests measure throughput and rough max context window on a 16 GB card.
* See [probe/README.md](probe/README.md) for probe runner details.
* Each model under [models/](models/) has `model.yaml` (GGUF path, `hf-download` repo/file) and `server.yaml` (base llama-server load profile) at the model root.
* Scaffold new models from [probe/templates/](probe/templates/).
  * Cursor skill `create-model-configs` can do this automatically - just tell it which model + variant you want and it'll do the rest.

### Run Server From Model Config

```bash
./launch_server.py models/qwen3.5-9b-q8/
```

Use a browser to access the llama web UI while it is running (port depends on `server.yaml` but is typically 8080):

[https://localhost:8080](https://localhost:8080)

### Calibration Tests

```bash
# standard dense model
./probe_calibrate.py models/qwen3.5-9b-q8

# MoE models: offload experts to CPU (to fit large models on small cards)
./probe_calibrate.py models/qwen3.6-35b-a3b-ud-q4-k-xl/ --n-cpu-moe 24
```

- Shared calibration probe YAML lives under [probe/calibration-probes/](probe/calibration-probes/).
- For MoE models, pass `--n-cpu-moe N` on calibration and probe run (replaces any existing MoE offload flags in merged server config).
- Add `--save-result` to save detailed output JSON to `probe/results/{model}/`.

### `llama-bench` CLI Wrapper

* Runs `llama-bench` from a model config dir. 
* Each model has `llama_bench.yaml` at the model root (alongside `server.yaml`). 
* Throughput numbers in the [models table](#models) still come from probe calibration and full CLI runs (`--save-result`), not llama-bench (different measurement).

```bash
./llama_bench.py models/qwen3.5-9b-q8/
```

See [probe/templates/llama_bench.yaml](probe/templates/llama_bench.yaml) for the config file format.

### Huggingface CLI
Some useful commands...

```bash
# download an entire repo (all variants, normally huge!)
hf download unsloth/Qwen3.6-35B-A3B-MTP-GGUF

# download a quant variant
hf download unsloth/Qwen3.6-35B-A3B-MTP-GGUF/blob/main/Qwen3.6-35B-A3B-MTP-GGUF-Q4_K_XL.gguf

# list model in local cache
hf cache ls --filter "type=model" --sort size:desc
hf cache ls --filter "type=model" --sort size:desc -q   # just the model name
hf cache ls --filter "type=model" --sort size:desc --json | jq . # formatted json

# delete a model (deletes all variants)
hf delete unsloth/Qwen3.6-35B-A3B-MTP-GGUF

# remove a specific variant
hf cache rm <revision_hash>

# remove detactched/orphan variants
hf cache prune
```

---

## Key Learnings

### Model types

Model filenames often encode the architecture:

- **Dense** (e.g. `Qwen3.5-9B`, `Qwen3.6-27B`): every parameter runs on every token. Simpler to load; VRAM scales with total size.
- **MoE** (e.g. `35B-A3B`): many expert sub-networks, but only a few activate per token (~3B active here). Near-large-model quality at small-model speed; on 16 GB you may need `--n-cpu-moe` to offload experts to CPU RAM.
- **Effective (E)** (e.g. `Gemma-4-E4B`): "E" means effective parameters (PLE, not MoE). Per-layer embedding tables hold most weights (~4.5B effective, ~8B on disk). Tuned for on-device and edge deployment: fast decode, modest VRAM, strong tool use for the size. Loads and runs like a dense model in llama.cpp.

### Reading quant names

Quantization shrinks weights so bigger models fit on consumer GPUs. Names look cryptic; they are mostly `{prefix}-{family}_{tier}`.

**Bit width (the number):**

- `FP16` / `BF16`: near full quality; huge files. Use when VRAM is not a constraint and you want baseline fidelity.
- `Q8`: ~8 bits per weight; often hard to tell from full precision. Best quality-to-size ratio when you have headroom.
- `Q4`: ~4 bits; answers can slip on hard tasks. The usual tradeoff for fitting 20B+ models locally.
- `Q2`: ~2 bits; quality drops sharply and varies by model. Last resort when nothing else fits.

**Prefix:**

- `UD-` (Unsloth Dynamic): mixed precision per layer, tuned with calibration data. Better chat/coding quality at the same nominal Q4 size, at the cost of slightly slower inference.

**Family:**

- `Q4_K`: standard llama.cpp K-quants; mixed block sizes inside the file. Predictable, well-tested 4-bit format.
- `IQ4`: importance quants; lean harder on calibration to preserve quality at lower size. Smallest files in the ~4-bit class (e.g. `IQ4_XS` ~18 GB vs `Q4_K_XL` ~23 GB on the same model).
- `MXFP4_MOE`: microscaling FP4 aimed at MoE expert weights. Tuned for sparse expert layers rather than uniform `Q4_K` blocks.

**Tier suffix (`S` / `M` / `L` / `XL` / `XS` / `NL`):**

- `S` (small): most compressed in that family. Saves disk and VRAM but expect more quality loss.
- `M` (medium): balanced default and a safe general-purpose pick when you are unsure.
- `L` (large): less compression than `S`/`M`; a step up in quality within the same Q4 family.
- `XL`: not "extra large file" but a smart mix that keeps sensitive tensors at `Q5`/`Q6` while the rest stays `Q4`. Best quality in the Q4 class; Unsloth's usual recommendation over plain `Q4_K_M`.
- `XS` (I-quants only): extra-small; most aggressive IQ compression. Maximum headroom for context on tight VRAM.
- `NL` (I-quants only): non-linear dequant scheme; slightly larger than `XS`. Often a better speed/quality tradeoff than `XS` at similar size.

**Other:**

- `TurboQuant`: newer KV-cache compression (`turbo3` / `turbo4`), not the weight file itself. Stretches context without re-downloading a different GGUF.

Example: `Qwen3.6-35B-A3B-UD-Q4_K_XL` = MoE model, Unsloth Dynamic mixed Q4 quant, XL tier (best Q4 quality).

### Multi-Token Prediction (MTP)

Some repos are labelled **MTP-GGUF** (e.g. `unsloth/Qwen3.6-35B-A3B-MTP-GGUF`). The file includes extra prediction heads baked into the model.

With those flags enabled, llama.cpp drafts several tokens ahead and verifies them in one pass, giving roughly **1.5-2x faster generation** with no accuracy loss.

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
* `--jinja` - use the model's Jinja chat template from GGUF metadata (off by default).
  * Required for OpenAI-style tool calling (`tools` in `/v1/chat/completions`).
  * With tools in the request, llama-server prefers `tokenizer.chat_template.tool_use` when present.
* `--chat-template-file PATH` - override the embedded chat template with a local `.jinja` file.
  * Usually not needed; check `http://localhost:8080/props` after `--jinja` first.
  * Use when the GGUF template is wrong, missing tool support, or you need a known community override.
* `--flash-attn` - Flash Attention: faster attention and lower KV-cache VRAM use.
  * Helps fit longer context on the same GPU; not tool-calling-specific.
  * Needs a build with FA support; drop it if startup fails or output looks off.

---

## System Setup

WSL2, Ubuntu 24.04...

```
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

