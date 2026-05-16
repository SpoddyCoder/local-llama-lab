# Tester v1: Phase 2 and 3 roadmap

High-level direction only. Phase 1 (single run, JSON, teardown) is documented in [tester-v1/README.md](../tester-v1/README.md) and specified in [tester-v1-implementation-plan.md](tester-v1-implementation-plan.md).

## Phase 2: Harness quality

Improve trust and ergonomics of one-off runs. No sweep automation.

- VRAM polling during the run; populate `peak_vram_mb` in metrics
- Optional warmup request in client config (discarded before the measured run)
- Clearer errors (server stderr tail, connection refused, timeouts) and consistent exit codes
- Extra metadata in result JSON (server version, GPU name, driver)
- Optional `notes` in YAML copied into the result; commented example configs
- Optional short completion preview on stdout (off by default)
- Doc section on comparing two JSON files manually

## Phase 3: Scale and comparison

Add automation when manual single runs are no longer enough.

- Sweep runner: expand server and/or client settings into a matrix; run serially with the same lifecycle as Phase 1
- CSV export aligned with README experiment log columns
- Compare command: two result files or latest pair in `results/`, delta table on stdout
- Named server preset YAML files (`fast`, `quality`, `low_vram`, and similar)
- Optional CI thresholds on metrics such as `decode_tok_s` for a pinned model

Phase 3 wraps the Phase 1 core (`server` to `client` to `results`). Manual `python src/python_runner.py` stays the default until you want sweeps or batch compare.

For detailed Phase 2/3 notes and Phase 1 acceptance criteria, see [tester-v1-implementation-plan.md](tester-v1-implementation-plan.md).
