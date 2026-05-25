---
name: probe-unit-tests
description: >-
  Write or change unittest tests under probe/tests/. Ensures tests do not
  write artifacts under probe/results/. Use when adding tests or mocking
  save-result / include-output paths.
---

# Probe unit tests

Follow `.cursor/rules/probe-unit-tests.mdc` for all edits under `probe/tests/`.

Run from `probe/`:

```bash
./run_python_unit_tests.py
```

Do not use probe CLIs with `--save-result` to verify test changes unless the user asks (`probe-cli.mdc`).
