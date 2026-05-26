# Berkeley Function Call Leaderboard (BFCL)
Tool calling is where small local models often fall down.
BFCL allows us to benchmark our models.

Current limitations:
* Only a small number of older models are supported natively.
* Newer models need:
  * TODO


## Install
```bash
./wsl-builder.sh ai-resources bfcl-eval

INFO: Note: to use BFCL eval:
INFO:     conda activate bfcl-eval
INFO:     edit ~/ai-resources/gorilla/berkeley-function-call-leaderboard/.env (API keys and config)
INFO:     bfcl generate --model MODEL_NAME --test-category TEST_CATEGORY
INFO: Optional: pip install -e .[oss_eval_vllm] or -e .[oss_eval_sglang] for self-hosted models
```

## Setup
The project repo contains a symlink to the bfcl repo...

```bash
ll bfcl/
total 268
drwxr-xr-x  7 fernpa fernpa   4096 May 25 22:20 ./
drwxr-xr-x 12 fernpa fernpa   4096 May 25 19:53 ../
-rw-r--r--  1 fernpa fernpa   1629 May 25 21:19 .env
drwxr-xr-x  2 fernpa fernpa   4096 May 25 22:21 .file_locks/
-rw-r--r--  1 fernpa fernpa  44359 May 25 19:53 CHANGELOG.md
-rw-r--r--  1 fernpa fernpa   9293 May 25 19:53 CONTRIBUTING.md
-rw-r--r--  1 fernpa fernpa   3106 May 25 19:53 LOG_GUIDE.md
-rw-r--r--  1 fernpa fernpa  17311 May 25 19:53 README.md
-rw-r--r--  1 fernpa fernpa  21993 May 25 19:53 SUPPORTED_MODELS.md
-rw-r--r--  1 fernpa fernpa   3551 May 25 19:53 TEST_CATEGORIES.md
-rw-r--r--  1 fernpa fernpa 115465 May 25 19:53 architecture_diagram.png
drwxr-xr-x  8 fernpa fernpa   4096 May 25 19:55 bfcl_eval/
drwxr-xr-x  2 fernpa fernpa   4096 May 25 19:53 bfcl_eval.egg-info/
-rw-r--r--  1 fernpa fernpa    498 May 25 19:53 openfunctions_evaluation.py
-rw-r--r--  1 fernpa fernpa   2013 May 25 19:53 pyproject.toml
drwxr-xr-x  4 fernpa fernpa   4096 May 25 21:26 result/
drwxr-xr-x  4 fernpa fernpa   4096 May 25 22:21 score/
-rw-r--r--  1 fernpa fernpa    671 May 25 22:20 test_case_ids_to_generate.json
```

## Running Tests

* Edit the `test_case_ids_to_generate.json` file to configure the tests to run.
* Current set is AI generated, criteria: good tests for a coding agent. 15-20mins to run.

Start a lllama server...

```bash
./launch_server.py models/qwen3-8b-ud-q8-k-xl/
```

Run tests (takes ages)...

```bash
bfcl generate \
  --model Qwen/Qwen3-8B-FC \
  --run-ids \
  --skip-server-setup \
  --allow-overwrite \
  --num-threads 1   # IMPORTANT: matches the llama server.yaml config, otherwise tests will fail giving a false negative.
```

Evaluate tests...

```bash
bfcl evaluate \
  --model Qwen/Qwen3-8B-FC \
  --test-category simple_python,irrelevance,parallel,multiple,live_simple,multi_turn_base,multi_turn_miss_func \
  --partial-eval
```

Ask the AI to summarise the output files in `results/Qwen_Qwen3-8B-FC/` if you want.


## Model Results:

### Qwen3-8B-FC

```bash
🦍 Model: Qwen_Qwen3-8B-FC
🔍 Running test: multi_turn_miss_func
✅ Test completed: multi_turn_miss_func. 🎯 Accuracy: 40.00%
🔍 Running test: multi_turn_base
✅ Test completed: multi_turn_base. 🎯 Accuracy: 62.50%
🔍 Running test: live_simple
✅ Test completed: live_simple. 🎯 Accuracy: 46.15%
🔍 Running test: multiple
✅ Test completed: multiple. 🎯 Accuracy: 87.50%
🔍 Running test: irrelevance
✅ Test completed: irrelevance. 🎯 Accuracy: 71.43%
🔍 Running test: simple_python
✅ Test completed: simple_python. 🎯 Accuracy: 93.33%
🔍 Running test: parallel
✅ Test completed: parallel. 🎯 Accuracy: 100.00%
```

## Current Test Config

```json
{
    "simple_python": [
        "simple_python_0",
        "simple_python_27",
        "simple_python_53",
        "simple_python_80",
        "simple_python_107",
        "simple_python_133",
        "simple_python_160",
        "simple_python_187",
        "simple_python_213",
        "simple_python_240",
        "simple_python_267",
        "simple_python_293",
        "simple_python_320",
        "simple_python_347",
        "simple_python_387"
    ],
    "simple_java": [],
    "simple_javascript": [],
    "irrelevance": [
        "irrelevance_0",
        "irrelevance_32",
        "irrelevance_64",
        "irrelevance_96",
        "irrelevance_128",
        "irrelevance_160",
        "irrelevance_192"
    ],
    "parallel": [
        "parallel_0",
        "parallel_27",
        "parallel_53",
        "parallel_80",
        "parallel_107",
        "parallel_133",
        "parallel_160",
        "parallel_187"
    ],
    "multiple": [
        "multiple_0",
        "multiple_27",
        "multiple_53",
        "multiple_80",
        "multiple_107",
        "multiple_133",
        "multiple_160",
        "multiple_187"
    ],
    "parallel_multiple": [],
    "live_simple": [
        "live_simple_141-94-0",
        "live_simple_142-94-1",
        "live_simple_143-95-0",
        "live_simple_144-95-1",
        "live_simple_145-95-2",
        "live_simple_152-95-9",
        "live_simple_153-95-10",
        "live_simple_158-95-15",
        "live_simple_159-95-16",
        "live_simple_166-99-0",
        "live_simple_167-99-1",
        "live_simple_168-99-2",
        "live_simple_169-99-3"
    ],
    "live_multiple": [],
    "live_parallel": [],
    "live_parallel_multiple": [],
    "live_irrelevance": [],
    "live_relevance": [],
    "multi_turn_base": [
        "multi_turn_base_0",
        "multi_turn_base_7",
        "multi_turn_base_13",
        "multi_turn_base_20",
        "multi_turn_base_27",
        "multi_turn_base_33",
        "multi_turn_base_40",
        "multi_turn_base_47"
    ],
    "multi_turn_miss_func": [
        "multi_turn_miss_func_0",
        "multi_turn_miss_func_40",
        "multi_turn_miss_func_80",
        "multi_turn_miss_func_120",
        "multi_turn_miss_func_180"
    ],
    "multi_turn_miss_param": [],
    "multi_turn_long_context": [],
    "multi_turn_composite": [],
    "memory_kv": [],
    "memory_vector": [],
    "memory_rec_sum": [],
    "web_search_base": [],
    "web_search_no_snippet": [],
    "format_sensitivity": []
}
```