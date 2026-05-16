# Reference test configs

`reference/reference-model/` is the only template source for new model configs. When scaffolding a model under `test-configs/{test_name}/`, copy these eight YAML files (four variants, each with `server.yaml` and `client.yaml`) and set `model:` to the target GGUF path.

Do not run `single_test_runner.py` or `model_calibration.py` against paths under `reference/`. These files are templates only, not runnable test trees.

The `model:` field in the reference tree may point at any valid local GGUF (the checked-in path is a working example). The create-new-model-test skill replaces `model:` when copying into `{test_name}/{model_slug}/`.
