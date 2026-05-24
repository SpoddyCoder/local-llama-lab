# Calibration test probes

`calibration-tests/` holds shared probe YAML used by [tester_v1_calibrate.py](../../tester_v1_calibrate.py). Calibration loads these files in-process; model paths come from `models/{model_slug}/model.yaml` and the base load profile from `models/{model_slug}/server.yaml`.

## Variants

Current shared variants (three dirs, six YAML files): `calibration-footprint`, `calibration-ctx-probe`, `hello-world-baseline`. Each has `client.yaml` and a thin `server.yaml` (ctx-size override only). Full llama-server load settings come from the model's root `server.yaml`, merged at run time. New dirs under `calibration-tests/` with both files are picked up when validating calibration.

Scaffold new models from [tester-v1/templates/](../templates/) (`model.yaml`, `server.yaml`). [model.yaml](model.yaml) here is a structural reference only. The [create-model-configs](../../.cursor/skills/create-model-configs/SKILL.md) skill copies from `templates/`, not from this dir.

## Not a run-test config_dir

Do not pass `calibration-tests/` (or a variant under it) to `tester_v1_run_test.py`. Pass the model slug dir and select a probe with `--variant`:

```bash
./tester_v1_run_test.py models/{model_slug}/ --variant hello-world-baseline
```

Calibration merges each probe's thin `server.yaml` with the model's root `server.yaml` in-process. Saved results go to `tester-v1/results/{model}/{variant}/`, not under the model dir.
