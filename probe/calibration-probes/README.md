# Calibration test probes

`calibration-probes/` holds shared probe YAML used by [probe_calibrate.py](../../probe_calibrate.py). Calibration loads these files in-process; model paths come from `models/{model}/model.yaml` and the base load profile from `models/{model}/server.yaml`.

## Variants

Current shared variants (three dirs, six YAML files): `calibration-footprint`, `calibration-ctx-probe`, `hello-world-baseline`. Each has `client.yaml` and a thin `server.yaml` (ctx-size override only). Full llama-server load settings come from the model's root `server.yaml`, merged at run time. New dirs under `calibration-probes/` with both files are picked up when validating calibration.

Scaffold new models from [probe/templates/](../templates/) (`model.yaml`, `server.yaml`). [model.yaml](model.yaml) here is a structural reference only. The [create-model-configs](../../.cursor/skills/create-model-configs/SKILL.md) skill copies from `templates/`, not from this dir.

## Not a probe run config_dir

Do not pass `calibration-probes/` (or a variant under it) to `probe_run.py`. Pass the model directory and select a probe with `--variant`:

```bash
./probe_run.py models/{model}/ --variant hello-world-baseline
```

Calibration merges each probe's thin `server.yaml` with the model's root `server.yaml` in-process. Saved results go to `probe/results/{model}/{variant}/`, not under the model dir.
