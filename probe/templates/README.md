# Config templates

Copy these when scaffolding a new model under `models/`. Canonical source for new models; `calibration-probes/model.yaml` is a structural reference only.

| File | Use |
|------|-----|
| `model.yaml` | Copy to `models/{model}/model.yaml` and set the GGUF path; optional `hf-download` (`repo`, `file`) for [download_model.py](../../download_model.py) |
| `server.yaml` | Copy to `models/{model}/server.yaml` (base llama-server load profile) |
| `sandbox/client.yaml` | Optional ad-hoc prompt; copy to `models/{model}/sandbox/client.yaml` and run with `--variant sandbox` |

Calibration probes (`calibration-footprint`, `calibration-ctx-probe`, `hello-world-baseline`) stay in `probe/calibration-probes/` — do not copy them into model dirs.
