# Hello world probe run

Simple smoke test prompt, standard probe in `configs/reference/`.

```text
hi, write me hello world in 20 different programming languages.
```

`client.yaml`:

```text
"max_tokens": 1024,
"temperature": 0
```

From `tester-v1/`:

```bash
./single_test_runner.py configs/qwen3.5-9b-q8/hello-world-baseline
```

## Qwen3.5-9B-Q8

### hello-world-baseline
```text
End-to-end time         11.59 s
Peak VRAM               14.72 GiB
Generation throughput   89.84 tok/s
```

## Gemma 4 E4B-IT-Q8

### hello-world-baseline
```text
End-to-end time         9.18 s
Peak VRAM               8.82 GiB
Generation throughput   115.11 tok/s
```
