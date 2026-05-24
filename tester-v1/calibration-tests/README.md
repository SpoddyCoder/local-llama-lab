# Calibration test probes

`calibration-tests/` holds shared probe YAML used by [tester_v1_calibrate.py](../../tester_v1_calibrate.py). Calibration loads these files in-process; model paths come from `models/{model_slug}/model.yaml`.

## Variants

Current shared variants (three dirs, six YAML files): `calibration-footprint`, `calibration-ctx-probe`, `hello-world-baseline`. Each has `server.yaml` and `client.yaml` only. `server.yaml` files use an `args: |` block scalar (one `llama-server` flag per line). New dirs under `calibration-tests/` with both files are picked up when validating calibration.

[model.yaml](model.yaml) is a structural example for scaffolding new models (GGUF path). The [create-model-configs](../../.cursor/skills/create-model-configs/SKILL.md) skill writes live `model.yaml` under `models/{model_slug}/` and does not duplicate variant YAML.

## Not a run-test config_dir

Do not pass `calibration-tests/` (or a variant under it) to `tester_v1_run_test.py`. Routine runs use a model variant dir such as `models/{model_slug}/bench/`. Calibration creates ephemeral result paths under the model dir for layout only; probe server and client settings always load from `calibration-tests/{variant}/`.
