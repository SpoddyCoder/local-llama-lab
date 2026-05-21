---
name: tester-unit-tests
description: >-
  Write or change unittest tests under tester-v1/tests/. Ensures tests do not
  write artifacts under tester-v1/results/. Use when adding tests or mocking
  save-result / include-output paths.
---

# Tester unit tests

Follow `.cursor/rules/tester-unit-tests.mdc` for all edits under `tester-v1/tests/`.

Run from `tester-v1/`:

```bash
./run_python_unit_tests.py
```

Do not use harness CLIs with `--save-result` to verify test changes unless the user asks (`tester-harness-cli.mdc`).
