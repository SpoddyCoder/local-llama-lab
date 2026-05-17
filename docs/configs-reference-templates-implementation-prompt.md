# Implementation prompt: reference templates and sync rules

Use this document as the full spec for one implementation pass. Do not run `model_calibration.py` or `single_test_runner.py` with `--save-result` unless the user explicitly asks.

## Goal

Align the repo with this model:

- **`tester-v1/configs/reference/`** — global templates only. Every variant subdirectory here is copied to **all** live models when scaffolding or when reference changes propagate.
- **`tester-v1/configs/{model_slug}/`** — per-model tree. May contain **extra** variant dirs that are **not** in `reference/` (model-specific experiments). Those are never deleted by reference sync.
- **Remove** `hello-world-no-mmap` from `reference/` only. **Keep** `hello-world-no-mmap/` on existing live models (`qwen3.5-9b-q8`, `gemma-4-e4b-it-q8`).

No backward compatibility for old skill wording, four-variant lists, or `test-configs` paths.

---

## 1. Filesystem: remove reference variant only

Delete:

- `tester-v1/configs/reference/hello-world-no-mmap/server.yaml`
- `tester-v1/configs/reference/hello-world-no-mmap/client.yaml`
- the empty `hello-world-no-mmap/` directory

**Do not** delete or modify:

- `tester-v1/configs/qwen3.5-9b-q8/hello-world-no-mmap/`
- `tester-v1/configs/gemma-4-e4b-it-q8/hello-world-no-mmap/`

After this, `reference/` should contain exactly three variant dirs:

- `calibration-footprint/`
- `calibration-ctx-probe/`
- `hello-world-baseline/`

Each with `server.yaml` and `client.yaml`.

---

## 2. Update `.cursor/skills/create-model-configs/SKILL.md`

Rewrite reference/scaffold sections to match:

### Core rule

**Every variant subdirectory under `tester-v1/configs/reference/` is a global template.** When scaffolding a new model, copy **all** of them (not a fixed list). If the user adds a new dir under `reference/`, it is by definition meant for every model.

### Scaffolding behavior

1. List `tester-v1/configs/reference/*/` — include only subdirs that contain both `server.yaml` and `client.yaml`.
2. For each such variant, create `tester-v1/configs/{model_slug}/{variant}/` with the same two files.
3. Copy template content verbatim except set `model:` in each `server.yaml` to the user's `gguf_path` (preserve `~` if provided).
4. Do not copy from live model dirs (`qwen3.5-9b-q8/`, etc.).

### Remove outdated content

- Remove "four variants", fixed lists naming `hello-world-no-mmap`, and the variant table row for `hello-world-no-mmap`.
- Remove "create all four variants" wording; use "create all reference variants" or "one dir per reference variant".
- Frontmatter `description:` should say scaffold copies **all** reference variants (calibration + hello-world-baseline as of this change; dynamic thereafter).

### Optional note in skill (short)

Model-only variants (e.g. `hello-world-no-mmap` on one model) are added manually under that model's slug; they do not live in `reference/` unless promoted there.

### Variant summary table

Keep a table describing **current** reference variants (three rows after this task). Note that the table is illustrative and reference enumeration is authoritative.

| Variant | Purpose (brief) |
|---------|-----------------|
| `calibration-footprint/` | VRAM cal, `-c 4096`, `--fit off` |
| `calibration-ctx-probe/` | VRAM cal, `-c 16384`, `--fit off` |
| `hello-world-baseline/` | Smoke probe, default server flags |

### Workflow step 3

"List subdirs under `configs/reference/`" and create the same set under `{model_slug}/`.

---

## 3. Update `.cursor/rules/configs-reference.mdc`

Expand the existing rule (or split if cleaner). Globs today: `tester-v1/configs/**`, `.cursor/skills/create-model-configs/**`. Keep those; ensure reference sync behavior is documented here.

### Template source (unchanged intent)

`tester-v1/configs/reference/` is canonical. create-model-configs copies from here only.

### What belongs in reference

- **In reference:** variants that should exist on **every** model (scaffold + propagate).
- **Not in reference:** model-specific experiments (e.g. `complex-webapp-build` on one slug only). Do **not** add those to `reference/` unless the user intends global rollout.

### Promoting / demoting variants

- **Promote to global:** add variant under `reference/`, then propagate to all models (see sync below).
- **Demote from global:** remove variant from `reference/` only. **Do not** delete that variant from live models automatically; model-only copies may remain (e.g. `hello-world-no-mmap` on Qwen/Gemma after this task).

### Reverse sync (old rule — replace)

Remove or replace the bullet that says: when you change a variant under a real model, update `reference/`. New rule:

- **Default:** edit `reference/` first for shared behavior; propagate to models.
- **Exception:** one-off tweaks on a single model stay on that model only and must not be copied into `reference/` unless the user promotes them.

### Do not run on reference (keep)

Do not run harness CLIs on paths under `configs/reference/`.

---

## 4. New rule: reference → all models sync

Create **`.cursor/rules/configs-reference-sync.mdc`** (or equivalent name).

```yaml
---
description: Propagate configs/reference template changes to all live model directories
globs: tester-v1/configs/reference/**
alwaysApply: false
---
```

### When this rule applies

Whenever the user or agent **edits, adds, or removes** files under `tester-v1/configs/reference/` (including after removing `hello-world-no-mmap`), run the propagation workflow below unless the user explicitly says not to sync.

### Live model set

All immediate subdirectories of `tester-v1/configs/` **except** `reference/`. Current slugs: `qwen3.5-9b-q8`, `gemma-4-e4b-it-q8` (discover dynamically; do not hardcode).

### Propagation workflow

For each live `{model_slug}/`:

1. **Add missing variants:** For each `reference/{variant}/` with both YAML files, if `{model_slug}/{variant}/` is missing, create it by copying from reference and set `model:` in `server.yaml` to the **existing** value from any sibling variant on that model, or from an existing `server.yaml` on that model if present; if the model dir is new/empty, ask the user for `gguf_path` before writing.

2. **Update existing reference-backed variants:** For each `reference/{variant}/` that also exists under `{model_slug}/{variant}/`, update `server.yaml` and `client.yaml` from reference **except** preserve the current `model:` line in `server.yaml` on the live copy.

3. **Do not delete** variant directories on live models when they are absent from `reference/`. Model-specific dirs (e.g. `hello-world-no-mmap`) stay.

4. **Do not overwrite** `model:` from reference into live models.

### User notification (required)

After propagation, report explicitly:

- Which `model_slug` directories were touched.
- Per model: which variants were **created** vs **updated** (file paths).
- Which reference variants were **not** propagated because of missing `model:` (if any — should ask user).
- Reminder: variants removed from `reference/` were **not** deleted from live models.

Example:

```text
Synced reference → models:
- qwen3.5-9b-q8: updated hello-world-baseline (server.yaml, client.yaml)
- gemma-4-e4b-it-q8: updated hello-world-baseline (server.yaml, client.yaml)
Not deleted: hello-world-no-mmap on both models (not in reference).
```

### Overwrite policy

When reference content changes, **overwrite** live `server.yaml` / `client.yaml` for variants that exist in both reference and that model (except `model:`). User accepts that reference is source of truth for shared variants; model-specific dirs are untouched.

### Link from `configs-reference.mdc`

Add a short pointer: "When `reference/` changes, follow configs-reference-sync.mdc."

### Update `tester-harness-cli.mdc` if needed

Only if paths or rule names are cited there; keep `configs/reference/` as do-not-run for harness.

---

## 5. Documentation updates

Update stale "four variants" / global `hello-world-no-mmap` wording. Keep docs accurate for **optional** per-model `hello-world-no-mmap`.

| File | Changes |
|------|---------|
| [`tester-v1/configs/reference/README.md`](tester-v1/configs/reference/README.md) | Three variants, six YAML files; all dirs under `reference/` are global templates; scaffold copies all. |
| [`tester-v1/README.md`](tester-v1/README.md) | Layout diagram: `reference/` has three variants; live models may have extra dirs. Skill scaffolds all reference variants. `hello-world-no-mmap` optional per model. |
| [`README.md`](README.md) | Probe runs: global templates = whatever is in `reference/`; note Qwen/Gemma may still have extra variants like `hello-world-no-mmap`. |
| [`hello-world.md`](hello-world.md) | Clarify `hello-world-baseline` is the standard smoke variant; `hello-world-no-mmap` is optional and model-specific (link existing paths). |

Follow [`.cursor/rules/documentation-style.mdc`](../.cursor/rules/documentation-style.mdc): no emojis, no em-dashes.

---

## 6. Out of scope

- Python harness changes (`tester-v1/src/`) — not required; calibration still uses fixed dir names `calibration-footprint` and `calibration-ctx-probe`.
- Deleting or migrating `tester-v1/results/*.json`.
- Running GPU benchmarks.
- Adding new model-specific variants unless needed for doc examples.

---

## 7. Verification

1. `tester-v1/configs/reference/` has exactly three variant dirs; no `hello-world-no-mmap`.
2. `qwen3.5-9b-q8/hello-world-no-mmap/` and `gemma-4-e4b-it-q8/hello-world-no-mmap/` still exist with valid YAML.
3. Ripgrep repo (exclude `results/` if desired): no `four variant` / `eight YAML` in reference README; skill does not require `hello-world-no-mmap` in reference.
4. `python3 -m unittest discover -s tests -v` from `tester-v1/` — all pass.
5. `.cursor/rules/configs-reference-sync.mdc` exists and globs `tester-v1/configs/reference/**`.
6. Skill states: copy **all** reference variant subdirs when scaffolding.

---

## 8. Summary for implementer

| Action | Item |
|--------|------|
| Delete | `configs/reference/hello-world-no-mmap/` |
| Keep | Live model `hello-world-no-mmap/` dirs |
| Rewrite | `create-model-configs` skill — dynamic all-reference copy |
| Rewrite | `configs-reference.mdc` — promote/demote, no auto-delete on demote |
| Create | `configs-reference-sync.mdc` — propagate reference → all models, notify user |
| Update | reference README, tester-v1 README, root README, hello-world.md |
