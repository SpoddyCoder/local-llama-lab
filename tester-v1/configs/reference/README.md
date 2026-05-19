# Reference test configs

`reference/` is the only template source for new model configs. Scaffolding copies `model.yaml` and every variant subdirectory into `configs/{model_slug}/`, setting `model:` in `model.yaml` to the target GGUF path. Variant `server.yaml` and `client.yaml` files are copied unchanged.

Current variants (three dirs, six YAML files): `calibration-footprint`, `calibration-ctx-probe`, `hello-world-baseline`. Each has `server.yaml` and `client.yaml` only (no per-variant model path). Reference `server.yaml` templates use `args: |` block scalar (one `llama-server` flag per line). New dirs under `reference/` with both files are included automatically when scaffolding or syncing.

Do not run `single_test_runner.py` or `model_calibration.py` against paths under `reference/`. These files are templates only, not runnable test trees.

Reference `model.yaml` may point at any valid local GGUF (the checked-in path is a working example). The [create-model-configs](../../../.cursor/skills/create-model-configs/SKILL.md) skill writes `model.yaml` with the user's path and copies variant YAML from reference without editing server or client files.
