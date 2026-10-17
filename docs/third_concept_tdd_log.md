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

Pending implementation.

## Real-corpus verification

Pending implementation.

## Final regression

Pending implementation.
