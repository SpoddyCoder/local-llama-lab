
# Test 1 - First Steps
Simple smoke test prompt...

```text
hi, write me hello world in 20 different programming languages.
```

Test Script Config:
```
"max_tokens": 1024,
"temperature": 0
```

## `Qwen_Qwen3.5-9B-Q8_0.gguf`

* GGUF on disk: 9.55 GB
* Model VRAM: 10337 MiB
* KV VRAM: 4430 MiB
* Estimated Max Context: 150823 tokens

### `baseline` (no switches)

Default probe: metrics on stdout only. Use `--save-result` when you want a file under `results/`.

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

### `--no-mmap`
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

## `Gemma 4: E4B-IT-Q8_0.gguf`

* GGUF on disk: 8.03 GB
* Model VRAM: 7022 MiB
* KV VRAM: 7745 MiB
* Estimated Max Context: 446749 tokens

### `baseline` (no switches)

Default probe (same as Qwen above).

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

### `--no-mmap`
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