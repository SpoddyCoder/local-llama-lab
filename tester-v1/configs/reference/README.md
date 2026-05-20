# Reference test configs

`reference/` holds shared probe YAML used directly by the harness.

`model_calibration.py` and manual runs pass these files via `--server` and `--client` (for example `configs/reference/hello-world-baseline/server.yaml`). Model paths come from `configs/{model_slug}/model.yaml`.

## Variants

Current shared variants (three dirs, six YAML files): `calibration-footprint`, `calibration-ctx-probe`, `hello-world-baseline`. Each has `server.yaml` and `client.yaml` only. `server.yaml` files use an `args: |` block scalar (one `llama-server` flag per line). New dirs under `reference/` with both files are picked up when validating calibration.

[reference/model.yaml](model.yaml) is a structural example for scaffolding new models (GGUF path). The [create-model-configs](../../../.cursor/skills/create-model-configs/SKILL.md) skill writes live `model.yaml` under `configs/{model_slug}/` and does not duplicate variant YAML.

## Do not use as config_dir

Do not run `single_test_runner.py` or `model_calibration.py` with `config_dir` under `reference/`. Calibration uses `configs/{model}/{variant}/` only for result path layout; server and client settings always load from `configs/reference/{variant}/`.
