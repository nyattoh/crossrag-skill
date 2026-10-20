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

## T4 hypothesis runs

Working directory for all commands below: `skill/crossrag`.

Corpus: `D:\develop\works\crossrag\crossrag_past_engineering_corpus.json` (`corpus_id=26636196c50a5d08`, 51 items).

### Commands and exit codes

| Command | Exit | Output |
|---|---:|---|
| `python scripts/third_concept_cli.py --A 'ヨーグルト' --B '梅シロップ' --corpus 'D:\develop\works\crossrag\crossrag_past_engineering_corpus.json' --encoder hash --limit 5 --seed 0 --hypotheses '..\..\docs\third_concept_hypotheses\yogurt_ume.json'` | 0 | `D:\inbox\crossrag-third-concept\t4\yogurt_ume_hash_hyp.json` |
| `$env:HF_HUB_OFFLINE='1'; python scripts/third_concept_cli.py --A 'ヨーグルト' --B '梅シロップ' --corpus 'D:\develop\works\crossrag\crossrag_past_engineering_corpus.json' --encoder semantic --model 'intfloat/multilingual-e5-small' --limit 5 --seed 0 --hypotheses '..\..\docs\third_concept_hypotheses\yogurt_ume.json'` | 0 | `D:\inbox\crossrag-third-concept\t4\yogurt_ume_semantic_hyp.json` |
| `python scripts/third_concept_cli.py --A '入眠障害' --B '条件反射' --corpus 'D:\develop\works\crossrag\crossrag_past_engineering_corpus.json' --encoder hash --limit 5 --seed 0 --hypotheses '..\..\docs\third_concept_hypotheses\insomnia_conditioning.json'` | 0 | `D:\inbox\crossrag-third-concept\t4\insomnia_conditioning_hash_hyp.json` |
| `$env:HF_HUB_OFFLINE='1'; python scripts/third_concept_cli.py --A '入眠障害' --B '条件反射' --corpus 'D:\develop\works\crossrag\crossrag_past_engineering_corpus.json' --encoder semantic --model 'intfloat/multilingual-e5-small' --limit 5 --seed 0 --hypotheses '..\..\docs\third_concept_hypotheses\insomnia_conditioning.json'` | 0 | `D:\inbox\crossrag-third-concept\t4\insomnia_conditioning_semantic_hyp.json` |
| `python scripts/third_concept_cli.py --A 'H2O MEMORY' --B 'TRAJEXA' --corpus 'D:\develop\works\crossrag\crossrag_past_engineering_corpus.json' --encoder hash --limit 5 --seed 0 --hypotheses '..\..\docs\third_concept_hypotheses\h2o_trajexa.json'` | 0 | `D:\inbox\crossrag-third-concept\t4\h2o_trajexa_hash_hyp.json` |
| `$env:HF_HUB_OFFLINE='1'; python scripts/third_concept_cli.py --A 'H2O MEMORY' --B 'TRAJEXA' --corpus 'D:\develop\works\crossrag\crossrag_past_engineering_corpus.json' --encoder semantic --model 'intfloat/multilingual-e5-small' --limit 5 --seed 0 --hypotheses '..\..\docs\third_concept_hypotheses\h2o_trajexa.json'` | 0 | `D:\inbox\crossrag-third-concept\t4\h2o_trajexa_semantic_hyp.json` |
| `python scripts/third_concept_cli.py --A 'ヨーグルト' --B '梅シロップ' --emit-prompt` | 0 | `D:\inbox\crossrag-third-concept\t4\yogurt_ume_prompt.txt` |
| `python scripts/third_concept_cli.py --A 'ヨーグルト' --B '梅シロップ' --corpus 'D:\develop\works\crossrag\crossrag_past_engineering_corpus.json' --encoder hash --limit 5 --seed 0 --emit-prompt --with-math` | 0 | `D:\inbox\crossrag-third-concept\t4\yogurt_ume_prompt_with_math.txt` |
| `python scripts/third_concept_cli.py --A '入眠障害' --B '条件反射' --emit-prompt` | 0 | `D:\inbox\crossrag-third-concept\t4\insomnia_conditioning_prompt.txt` |
| `python scripts/third_concept_cli.py --A '入眠障害' --B '条件反射' --corpus 'D:\develop\works\crossrag\crossrag_past_engineering_corpus.json' --encoder hash --limit 5 --seed 0 --emit-prompt --with-math` | 0 | `D:\inbox\crossrag-third-concept\t4\insomnia_conditioning_prompt_with_math.txt` |
| `python scripts/third_concept_cli.py --A 'H2O MEMORY' --B 'TRAJEXA' --emit-prompt` | 0 | `D:\inbox\crossrag-third-concept\t4\h2o_trajexa_prompt.txt` |
| `python scripts/third_concept_cli.py --A 'H2O MEMORY' --B 'TRAJEXA' --corpus 'D:\develop\works\crossrag\crossrag_past_engineering_corpus.json' --encoder hash --limit 5 --seed 0 --emit-prompt --with-math` | 0 | `D:\inbox\crossrag-third-concept\t4\h2o_trajexa_prompt_with_math.txt` |

The three semantic runs used only the local model cache. Each emitted an `embeddings.position_ids | UNEXPECTED` load report and still exited `0`. All six hypothesis JSON files parsed, reported `corpus_size=51` and `corpus_id=26636196c50a5d08`, and emitted no geometry warning.

### SHA-256

| Output | SHA-256 |
|---|---|
| `h2o_trajexa_hash_hyp.json` | `dda8cca32ea44ac3fce4cdd30af9f83e9fe326a9a7d16bf59c28e9e3f70bbdeb` |
| `h2o_trajexa_prompt_with_math.txt` | `7d980f632be12100f94c901f8b2cbeaca8352ebd8597cde1c1b1ac9dd46e812c` |
| `h2o_trajexa_prompt.txt` | `e25a070aadf5315521d74cca8f01cc68aa5324e4f1a45262cf81cd3969da6612` |
| `h2o_trajexa_semantic_hyp.json` | `574dbf7d0deeb883ab9ddc7d76e8e9fd1f3dafefbe437c922afc44c0594834ea` |
| `insomnia_conditioning_hash_hyp.json` | `59a2dd944c1f094be25ad8b947699d46471623c69e7f3c68c498c59f0f53ad04` |
| `insomnia_conditioning_prompt_with_math.txt` | `b0dcdb49c832b44c42bd8866c7f5c8d67ec67b3458022f4fa6c263b05aa7a5de` |
| `insomnia_conditioning_prompt.txt` | `d387b30e50ed43dc92be2b4c86c7a3b3a5a2bde147c7f7738d476b06118a80a7` |
| `insomnia_conditioning_semantic_hyp.json` | `05e02679b2992b420f92174fd0e3f9c2770c8fd006cb6bcd84e32b21240706d0` |
| `yogurt_ume_hash_hyp.json` | `a8e12ac4cd7b98eef6b6b934a1ece39ab5f18c9cc6388367edec306fddc30f36` |
| `yogurt_ume_prompt_with_math.txt` | `781f7843637a388ec0bd28de3975e2db16aff29e05644c46107b349ca25caaff` |
| `yogurt_ume_prompt.txt` | `6600f15ee08fa4e80f3370f163228f4f9660069c016f36c58070165e84b093a9` |
| `yogurt_ume_semantic_hyp.json` | `bc26cdee460de9cdff50175496e08bd09161bc8b0afbfaafdc267b99d8e29957` |

### Full regression

Command:

```text
python -m unittest discover -s tests -v
```

Exit code: `0`

```text
Ran 26 tests in 2.156s
OK
```

## F1 review fixes

### Red

Command (working directory: `skill/crossrag`):

```text
python -m unittest discover -s tests -v
```

Exit code: `1`

```text
Ran 40 tests in 3.330s
FAILED (failures=13)
```

Failing tests and representative lines:

```text
test_all_methods_select_the_requested_number_of_corpus_entries
  Items in the second set but not the first: 'contains_input'
test_blind_sheet_is_method_and_score_blind_and_key_maps_back
  'Input A: `H2O MEMORY`' not found
test_third_concept_methods_exclude_inputs_but_baselines_mark_them
  0 unexpectedly found in {0, 1, 3, 6, 7}
test_whitespace_normalization_is_recorded_and_applied_to_every_method
  'H2O/MEMORY' != 'H2O_MEMORY'
test_geometry_treats_near_parallel_plane_as_degenerate_by_one_minus_rho
  0.9999999111821575 != 0.0
test_boolean_llm_self_assessment_is_rejected
  ValueError not raised
test_json_schema
  encoder.applied_to was not None for hash
test_limit_must_be_positive (limit=0)
test_limit_must_be_positive (limit=-1)
  ValueError not raised
test_rank_entries_excludes_text_and_raw_cosine_input_matches
  excluded rows were returned
test_shared_mean_keeps_same_concept_scores_equal_and_hypothesis_stable
  math and llm_hypothesis scores differed
test_zero_emergence_does_not_claim_geometric_ab_dependence
  required wording was absent
test_cli_rejects_non_positive_limit
  return code 0 == 0
```

The existing UTF-8, wedge-warning, legacy engine/CLI snapshot, prefix, leakage, schema, and origin-separation tests passed in the same RED run.

### Green

Command (working directory: `skill/crossrag`):

```text
python -m unittest discover -s tests -v
```

Exit code: `0`

```text
Ran 40 tests in 2.752s
OK
```

Additional checks:

| Command | Exit |
|---|---:|
| `python -m py_compile scripts/third_concept.py scripts/third_concept_cli.py scripts/compare_third_concept.py` | 0 |
| `git diff --check` | 0 |

### v2 real-corpus rerun

All 24 requested CLI/comparison commands exited `0`. Semantic commands ran with `HF_HUB_OFFLINE=1`. The exact command templates, case substitutions, per-output exit codes, and SHA-256 values are recorded in `docs/third_concept_report.md` sections 7.1–7.3. Outputs are under `D:\inbox\crossrag-third-concept\v2\`; the corpus remained `corpus_id=26636196c50a5d08`, 51 entries.

The semantic runs again emitted `embeddings.position_ids | UNEXPECTED`; JSON and Markdown generation still exited `0`. The earlier `t1`, `t3`, and `t4` numeric outputs are superseded by v2.
