# Third-concept PoC TDD log

## Red

Command (working directory: `skill/crossrag`):

```text
python -m unittest discover -s tests -v
```

Exit code: `1`

Relevant output:

```text
test_json_is_utf8_even_when_console_default_is_not (...) ... ok
test_default_hash_wedge_is_all_zero_and_reported (...) ... ok
test_no_warning_when_only_an_unselected_candidate_has_wedge (...) ... ok
test_no_warning_when_wedge_varies (...) ... ok
test_warning_covers_all_candidates_not_only_selected (...) ... ok
test_warning_names_the_encoder (...) ... ok
test_third_concept (unittest.loader._FailedTest.test_third_concept) ... ERROR
ModuleNotFoundError: No module named 'third_concept'
Ran 7 tests in 0.202s
FAILED (errors=1)
```

The six pre-existing tests passed. The new test module failed before collection because the production module did not yet exist.

### Legacy snapshot sensitivity check

The fixture was copied to a unique temporary directory, `default.similar.selected[0].score` was changed from `0.59519` to `999`, and the comparison was expected to fail. The temporary directory was removed by the same one-off command.

```text
AssertionError: snapshot mismatch: default.similar.selected[0].score 0.59519 != 999
EXIT_CODE=1
TEMP_COPY_REMOVED=True
```

## Green

Command (working directory: `skill/crossrag`):

```text
python -m unittest discover -s tests -v
```

Exit code: `0`

```text
Ran 19 tests in 1.787s
OK
```

This includes all six pre-existing tests and 13 third-concept contract/regression tests. A cached semantic-model smoke run with `HF_HUB_OFFLINE=1` also exited `0` before the real-corpus runs.

## Real-corpus verification

Corpus: `D:\develop\works\crossrag\crossrag_past_engineering_corpus.json` (51 items)

All commands used `--limit 5 --seed 0`. Every output parsed as JSON and reported `corpus_size=51` and `corpus_id=26636196c50a5d08`.

| Command | Exit | Output |
|---|---:|---|
| `python scripts/third_concept_cli.py --A ヨーグルト --B 梅シロップ --corpus D:\develop\works\crossrag\crossrag_past_engineering_corpus.json --encoder hash --limit 5 --seed 0` | 0 | `D:\inbox\crossrag-third-concept\t1\yoghurt_ume_hash.json` |
| `python scripts/third_concept_cli.py --A 入眠障害 --B 条件反射 --corpus D:\develop\works\crossrag\crossrag_past_engineering_corpus.json --encoder hash --limit 5 --seed 0` | 0 | `D:\inbox\crossrag-third-concept\t1\sleep_conditioning_hash.json` |
| `python scripts/third_concept_cli.py --A "H2O MEMORY" --B TRAJEXA --corpus D:\develop\works\crossrag\crossrag_past_engineering_corpus.json --encoder hash --limit 5 --seed 0` | 0 | `D:\inbox\crossrag-third-concept\t1\h2o_trajexa_hash.json` |
| `HF_HUB_OFFLINE=1 python scripts/third_concept_cli.py --A ヨーグルト --B 梅シロップ --corpus D:\develop\works\crossrag\crossrag_past_engineering_corpus.json --encoder semantic --model intfloat/multilingual-e5-small --limit 5 --seed 0` | 0 | `D:\inbox\crossrag-third-concept\t1\yoghurt_ume_semantic.json` |
| `HF_HUB_OFFLINE=1 python scripts/third_concept_cli.py --A 入眠障害 --B 条件反射 --corpus D:\develop\works\crossrag\crossrag_past_engineering_corpus.json --encoder semantic --model intfloat/multilingual-e5-small --limit 5 --seed 0` | 0 | `D:\inbox\crossrag-third-concept\t1\sleep_conditioning_semantic.json` |
| `HF_HUB_OFFLINE=1 python scripts/third_concept_cli.py --A "H2O MEMORY" --B TRAJEXA --corpus D:\develop\works\crossrag\crossrag_past_engineering_corpus.json --encoder semantic --model intfloat/multilingual-e5-small --limit 5 --seed 0` | 0 | `D:\inbox\crossrag-third-concept\t1\h2o_trajexa_semantic.json` |

Observed limitations: the engineering/record-oriented corpus produced obviously off-topic retrieved and math candidates for the food pair under both encoders. The sleep pair remained mostly adjacent rather than a clear combination-dependent third concept. H2O MEMORY/TRAJEXA had more recognisable thematic overlap, especially with the semantic encoder, but this is an inspection note rather than evidence of quality. No run emitted a geometry warning.

## Final regression

Command (working directory: `skill/crossrag`):

```text
python -m unittest discover -s tests -v
```

Exit code: `0`

```text
Ran 19 tests in 2.028s
OK
```

Additional final checks:

- `python -m py_compile scripts/third_concept.py scripts/third_concept_cli.py` — exit `0`.
- All six final real-corpus output files parsed as JSON with `corpus_size=51`, `corpus_id=26636196c50a5d08`, and `params.vocab_size=0`.

## T3 compare

### Red

Command (working directory: `skill/crossrag`):

```text
python -m unittest discover -s tests -v
```

Exit code: `1`

Relevant output:

```text
test_compare_third_concept (unittest.loader._FailedTest.test_compare_third_concept) ... ERROR
ModuleNotFoundError: No module named 'compare_third_concept'
Ran 20 tests in 4.718s
FAILED (errors=1)
```

The 19 existing tests passed; only the new comparison module import failed before implementation.

### Green

Command (working directory: `skill/crossrag`):

```text
python -m unittest discover -s tests -v
```

Exit code: `0`

```text
Ran 26 tests in 3.260s
OK
```

This includes the 19 existing tests and 7 new comparison tests. The separate compile check `python -m py_compile scripts/compare_third_concept.py` also exited `0`.

### Real-corpus comparison runs

Corpus: `D:\develop\works\crossrag\crossrag_past_engineering_corpus.json` (51 items, `corpus_id=26636196c50a5d08`)

All commands used `--limit 5 --seed 0`. The PowerShell runner captured stdout as UTF-8 without BOM at the listed output path. Semantic commands ran with `HF_HUB_OFFLINE=1`; the model was loaded only from the local cache. Each command exited `0`.

| Command | Exit | Output |
|---|---:|---|
| `python scripts/compare_third_concept.py --A ヨーグルト --B 梅シロップ --corpus D:\develop\works\crossrag\crossrag_past_engineering_corpus.json --encoder hash --limit 5 --seed 0 --format json` | 0 | `D:\inbox\crossrag-third-concept\t3\yoghurt_ume_hash.json` |
| `python scripts/compare_third_concept.py --A ヨーグルト --B 梅シロップ --corpus D:\develop\works\crossrag\crossrag_past_engineering_corpus.json --encoder hash --limit 5 --seed 0 --format markdown` | 0 | `D:\inbox\crossrag-third-concept\t3\yoghurt_ume_hash.md` |
| `HF_HUB_OFFLINE=1 python scripts/compare_third_concept.py --A ヨーグルト --B 梅シロップ --corpus D:\develop\works\crossrag\crossrag_past_engineering_corpus.json --encoder semantic --model intfloat/multilingual-e5-small --limit 5 --seed 0 --format json --sheet-dir D:\inbox\crossrag-third-concept\t3\sheets\yoghurt_ume` | 0 | `D:\inbox\crossrag-third-concept\t3\yoghurt_ume_semantic.json` |
| `HF_HUB_OFFLINE=1 python scripts/compare_third_concept.py --A ヨーグルト --B 梅シロップ --corpus D:\develop\works\crossrag\crossrag_past_engineering_corpus.json --encoder semantic --model intfloat/multilingual-e5-small --limit 5 --seed 0 --format markdown --sheet-dir D:\inbox\crossrag-third-concept\t3\sheets\yoghurt_ume` | 0 | `D:\inbox\crossrag-third-concept\t3\yoghurt_ume_semantic.md` |
| `python scripts/compare_third_concept.py --A 入眠障害 --B 条件反射 --corpus D:\develop\works\crossrag\crossrag_past_engineering_corpus.json --encoder hash --limit 5 --seed 0 --format json` | 0 | `D:\inbox\crossrag-third-concept\t3\sleep_conditioning_hash.json` |
| `python scripts/compare_third_concept.py --A 入眠障害 --B 条件反射 --corpus D:\develop\works\crossrag\crossrag_past_engineering_corpus.json --encoder hash --limit 5 --seed 0 --format markdown` | 0 | `D:\inbox\crossrag-third-concept\t3\sleep_conditioning_hash.md` |
| `HF_HUB_OFFLINE=1 python scripts/compare_third_concept.py --A 入眠障害 --B 条件反射 --corpus D:\develop\works\crossrag\crossrag_past_engineering_corpus.json --encoder semantic --model intfloat/multilingual-e5-small --limit 5 --seed 0 --format json --sheet-dir D:\inbox\crossrag-third-concept\t3\sheets\sleep_conditioning` | 0 | `D:\inbox\crossrag-third-concept\t3\sleep_conditioning_semantic.json` |
| `HF_HUB_OFFLINE=1 python scripts/compare_third_concept.py --A 入眠障害 --B 条件反射 --corpus D:\develop\works\crossrag\crossrag_past_engineering_corpus.json --encoder semantic --model intfloat/multilingual-e5-small --limit 5 --seed 0 --format markdown --sheet-dir D:\inbox\crossrag-third-concept\t3\sheets\sleep_conditioning` | 0 | `D:\inbox\crossrag-third-concept\t3\sleep_conditioning_semantic.md` |
| `python scripts/compare_third_concept.py --A "H2O MEMORY" --B TRAJEXA --corpus D:\develop\works\crossrag\crossrag_past_engineering_corpus.json --encoder hash --limit 5 --seed 0 --format json` | 0 | `D:\inbox\crossrag-third-concept\t3\h2o_trajexa_hash.json` |
| `python scripts/compare_third_concept.py --A "H2O MEMORY" --B TRAJEXA --corpus D:\develop\works\crossrag\crossrag_past_engineering_corpus.json --encoder hash --limit 5 --seed 0 --format markdown` | 0 | `D:\inbox\crossrag-third-concept\t3\h2o_trajexa_hash.md` |
| `HF_HUB_OFFLINE=1 python scripts/compare_third_concept.py --A "H2O MEMORY" --B TRAJEXA --corpus D:\develop\works\crossrag\crossrag_past_engineering_corpus.json --encoder semantic --model intfloat/multilingual-e5-small --limit 5 --seed 0 --format json --sheet-dir D:\inbox\crossrag-third-concept\t3\sheets\h2o_trajexa` | 0 | `D:\inbox\crossrag-third-concept\t3\h2o_trajexa_semantic.json` |
| `HF_HUB_OFFLINE=1 python scripts/compare_third_concept.py --A "H2O MEMORY" --B TRAJEXA --corpus D:\develop\works\crossrag\crossrag_past_engineering_corpus.json --encoder semantic --model intfloat/multilingual-e5-small --limit 5 --seed 0 --format markdown --sheet-dir D:\inbox\crossrag-third-concept\t3\sheets\h2o_trajexa` | 0 | `D:\inbox\crossrag-third-concept\t3\h2o_trajexa_semantic.md` |

Blind materials were written to these directories, each containing `blind_sheet.md` and `blind_key.json`:

- `D:\inbox\crossrag-third-concept\t3\sheets\yoghurt_ume`
- `D:\inbox\crossrag-third-concept\t3\sheets\sleep_conditioning`
- `D:\inbox\crossrag-third-concept\t3\sheets\h2o_trajexa`

The local semantic model emitted a load-report warning that `embeddings.position_ids` was unexpected. It did not change any exit code; all six JSON outputs parsed and reported the expected corpus ID and size.

### Final regression

Command (working directory: `skill/crossrag`):

```text
python -m unittest discover -s tests -v
```

Exit code: `0`

```text
Ran 26 tests in 2.315s
OK
```
