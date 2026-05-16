# Test 1 - First Steps

Smoke-test configs for [Test 1](../../../README.md#test-1---first-steps). Each model has four variants: `calibration-footprint`, `calibration-ctx-probe`, `baseline`, `no-mmap` (each with `server.yaml` and `client.yaml`).

Models: [qwen3.5-9b-q8](qwen3.5-9b-q8/), [gemma-4-e4b-it-q8](gemma-4-e4b-it-q8/). Update `model:` paths if your GGUF is not in the checked-in Hugging Face cache location.

From `tester-v1/`:

```bash
python3 model_calibration.py test-configs/test1/qwen3.5-9b-q8
python3 single_test_runner.py test-configs/test1/qwen3.5-9b-q8/baseline
```

Default run is a probe (stdout metrics only). Add `--save-result` when you want a file under `results/`.

See [tester-v1/README.md](../../README.md) for harness, calibration, and YAML details. New models: copy from [reference/reference-model](../reference/reference-model/), not from `test1/`.
