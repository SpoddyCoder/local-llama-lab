# Reference test configs

`reference/{variant}/` is the only template source for new model configs. When scaffolding a model under `configs/{model_slug}/`, copy these eight YAML files (four variants, each with `server.yaml` and `client.yaml`) and set `model:` to the target GGUF path.

Variants: `calibration-footprint`, `calibration-ctx-probe`, `hello-world-baseline`, `hello-world-no-mmap`.

Do not run `single_test_runner.py` or `model_calibration.py` against paths under `reference/`. These files are templates only, not runnable test trees.

The `model:` field in the reference tree may point at any valid local GGUF (the checked-in path is a working example). The [create-model-configs](../../../.cursor/skills/create-model-configs/SKILL.md) skill replaces `model:` when copying into `configs/{model_slug}/`.
