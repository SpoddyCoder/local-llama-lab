# Reference test configs

`reference/` is the only template source for shared probe YAML. Scaffolding a new model creates `configs/{model_slug}/model.yaml` with the user's GGUF path. Variant `server.yaml` and `client.yaml` stay here only; `model_calibration.py` passes them to `single_test_runner.py` via `--server` and `--client`.

Current variants (three dirs, six YAML files): `calibration-footprint`, `calibration-ctx-probe`, `hello-world-baseline`. Each has `server.yaml` and `client.yaml` only (no per-variant model path). Reference `server.yaml` templates use `args: |` block scalar (one `llama-server` flag per line). New dirs under `reference/` with both files are picked up automatically when validating calibration.

Do not run `single_test_runner.py` or `model_calibration.py` against paths under `reference/`. These files are templates only, not runnable test trees.

Reference `model.yaml` may point at any valid local GGUF (the checked-in path is a working example). The [create-model-configs](../../../.cursor/skills/create-model-configs/SKILL.md) skill writes live `model.yaml` with the user's path and does not copy variant YAML into model directories.
