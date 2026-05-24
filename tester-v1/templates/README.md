# Model config templates

Copy these when scaffolding a new slug under `models/`. Canonical source for new models; `calibration-tests/model.yaml` is a structural reference only.

| File | Use |
|------|-----|
| `model.yaml` | Copy to `models/{slug}/model.yaml` and set the GGUF path |
| `server.yaml` | Copy to `models/{slug}/server.yaml` (base llama-server load profile) |
| `sandbox/client.yaml` | Optional ad-hoc prompt; copy to `models/{slug}/sandbox/client.yaml` and run with `--variant sandbox` |
| `sandbox/server.yaml` | Optional thin server override (commented stub); copy with client when needed |

Calibration probes (`calibration-footprint`, `calibration-ctx-probe`, `hello-world-baseline`) stay in `tester-v1/calibration-tests/` — do not copy them into model dirs.

There is no `bench/` variant; routine runs use shared probe YAML via `--variant` (default `hello-world-baseline`).
