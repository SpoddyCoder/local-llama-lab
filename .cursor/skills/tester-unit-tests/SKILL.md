---
name: tester-unit-tests
description: >-
  Write or change unittest tests under tester-v1/tests/. Use when adding unit
  tests, fixing test_runner_cli or results tests, or mocking save-result /
  include-output paths. Ensures tests do not write artifacts under tester-v1/results/.
---

# Tester unit tests

Follow [.cursor/rules/tester-unit-tests.mdc](../../rules/tester-unit-tests.mdc) whenever you add or edit files under `tester-v1/tests/`.

## Core rule

**Never write under `tester-v1/results/` from unit tests.** That tree is for real harness output only.

## Mocked save-result paths

If you mock `write_result` (or pass a fake result path) and the code under test may still write siblings (for example `completion_output_path` / `{stem}-output.txt`), put the path under `tempfile.TemporaryDirectory()`:

```python
with tempfile.TemporaryDirectory() as tmp:
    result_path = (
        Path(tmp)
        / "results"
        / model_slug
        / variant
        / "20260101T000000Z.json"
    )
    patch("runner.write_result", return_value=result_path),
    ...
```

Do **not** use `_TESTER_ROOT / "results" / ...` as the mocked return value unless the test only reads and never triggers a real write from that path.

Canonical example: `test_include_output_with_save_result_writes_output_txt` in `tester-v1/tests/test_runner_cli.py`.

## Reads vs writes

| Action | Location |
|--------|----------|
| Read fixture JSON, assert parsers | Inline dicts/strings or files under `tempfile` only |
| Write JSON, `-output.txt`, session files | Use `tempfile` (or patch the write) |

## Running tests

From `tester-v1/`:

```bash
./run_python_unit_tests.py
```

Do not use harness CLIs with `--save-result` to verify unit-test changes unless the user asks (see `tester-harness-cli.mdc`).
