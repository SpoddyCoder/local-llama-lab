# Reference test configs

`reference/{variant}/` is the only template source for new model configs. Every variant subdirectory here is a global template: scaffold copies all of them into `configs/{model_slug}/` and sets `model:` to the target GGUF path.

Current variants (three dirs, six YAML files): `calibration-footprint`, `calibration-ctx-probe`, `hello-world-baseline`. Each has `server.yaml` and `client.yaml`. Reference `server.yaml` templates use `args: |` block scalar (one `llama-server` flag per line). New dirs under `reference/` with both files are included automatically when scaffolding or syncing.

Do not run `single_test_runner.py` or `model_calibration.py` against paths under `reference/`. These files are templates only, not runnable test trees.

The `model:` field in the reference tree may point at any valid local GGUF (the checked-in path is a working example). The [create-model-configs](../../../.cursor/skills/create-model-configs/SKILL.md) skill replaces `model:` when copying into `configs/{model_slug}/`.
