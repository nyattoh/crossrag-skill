"""Contract tests for the seven-way third-concept comparison."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
FIXTURES = Path(__file__).resolve().parent / "fixtures"
sys.path.insert(0, str(SCRIPTS))

from engine import run  # noqa: E402
from run import load_corpus  # noqa: E402
from third_concept import corpus_id, make_encoder  # noqa: E402
from compare_third_concept import (  # noqa: E402
    METHOD_NAMES,
    compare_methods,
    write_blind_materials,
)
import compare_third_concept as comparison  # noqa: E402


class RecordingModel:
    def __init__(self):
        self.calls = []

    def encode(self, texts, normalize_embeddings, show_progress_bar):
        self.calls.append(list(texts))
        rows = []
        for text in texts:
            raw = np.array([len(text), sum(map(ord, text)) % 17 + 1, text.count(":") + 1], dtype=float)
            rows.append(raw / np.linalg.norm(raw))
        return np.asarray(rows)


class FixedEncoder:
    name = "fixed"
    model_id = "fixed"
    prefix = None

    def __init__(self, vectors):
        self.vectors = {key: np.asarray(value, dtype=float) for key, value in vectors.items()}

    def encode_query(self, texts):
        return np.asarray([self.vectors[text] for text in texts])

    def encode_passage(self, texts):
        return np.asarray([self.vectors[text] for text in texts])

    def encode(self, texts):
        return self.encode_query(texts)


class CompareThirdConceptTest(unittest.TestCase):
    def setUp(self):
        self.corpus_path = FIXTURES / "synthetic_corpus.json"
        self.corpus = load_corpus(self.corpus_path)

    def compare(self, A="H2O MEMORY", B="TRAJEXA", limit=3, seed=7):
        encoder = make_encoder("hash")
        return compare_methods(
            A,
            B,
            self.corpus,
            encoder,
            limit=limit,
            seed=seed,
            corpus_path=self.corpus_path,
        )

    def test_all_methods_select_the_requested_number_of_corpus_entries(self):
        result = self.compare(limit=4)
        self.assertEqual(
            tuple(result["methods"]),
            ("similar", "orthogonal", "wedge", "random", "third-concept", "mean", "third-concept-no-wedge"),
        )
        self.assertEqual(tuple(result["methods"]), METHOD_NAMES)
        for method in result["methods"].values():
            self.assertEqual(len(method["selected"]), min(4, len(self.corpus)))
            self.assertEqual(
                set(method["means"]),
                {"s_A", "s_B", "plane_projection", "residual", "emergence_norm", "score_third"},
            )
        for name, method in result["methods"].items():
            for selected in method["selected"]:
                expected = {"corpus_index", "a", "b", "metrics"}
                if name in ("similar", "orthogonal", "wedge", "random"):
                    expected.add("contains_input")
                self.assertEqual(set(selected), expected)

    def test_common_encoder_corpus_and_limit_are_recorded_once(self):
        result = self.compare(limit=2)
        self.assertEqual(result["encoder"]["name"], make_encoder("hash").name)
        self.assertEqual(result["encoder"]["model_id"], "hash-ngram")
        self.assertIsNone(result["encoder"]["prefix"])
        self.assertEqual(result["corpus_id"], corpus_id(self.corpus, self.corpus_path))
        self.assertEqual(result["corpus_size"], len(self.corpus))
        self.assertEqual(result["params"], {"limit": 2, "seed": 7})
        for method in result["methods"].values():
            self.assertNotIn("encoder", method)
            self.assertNotIn("corpus_id", method)
            self.assertNotIn("limit", method)

    def test_baselines_are_exactly_engine_run_selections(self):
        encoder = make_encoder("hash")
        result = compare_methods(
            "H2O MEMORY",
            "TRAJEXA",
            self.corpus,
            encoder,
            limit=5,
            seed=0,
            corpus_path=self.corpus_path,
        )
        context = "H2O_MEMORY、TRAJEXA"
        for name in ("similar", "orthogonal", "wedge", "random"):
            expected = run(context, self.corpus, name, limit=5, encoder=encoder)
            self.assertEqual(
                [item["corpus_index"] for item in result["methods"][name]["selected"]],
                [item["index"] for item in expected["selected"]],
            )

    def test_whitespace_normalization_is_recorded_and_applied_to_every_method(self):
        result = self.compare(A="H2O/MEMORY", B="TRA JEXA", limit=3)
        self.assertEqual(
            result["input_normalization"],
            {
                "rule": r"[、,，/／\s]+ -> _",
                "A": {"original": "H2O/MEMORY", "used": "H2O_MEMORY"},
                "B": {"original": "TRA JEXA", "used": "TRA_JEXA"},
            },
        )
        self.assertEqual(result["input"], {"A": "H2O/MEMORY", "B": "TRA JEXA"})
        encoder = make_encoder("hash")
        for name in ("similar", "orthogonal", "wedge", "random"):
            expected = run("H2O_MEMORY、TRA_JEXA", self.corpus, name, limit=3, encoder=encoder)
            self.assertEqual(
                [item["corpus_index"] for item in result["methods"][name]["selected"]],
                [item["index"] for item in expected["selected"]],
            )

    def test_third_concept_methods_exclude_inputs_but_baselines_mark_them(self):
        result = self.compare(A="ヨーグルト", B="梅シロップ", limit=5)
        for name in ("third-concept", "mean", "third-concept-no-wedge"):
            self.assertNotIn(0, {item["corpus_index"] for item in result["methods"][name]["selected"]})
            self.assertNotIn(1, {item["corpus_index"] for item in result["methods"][name]["selected"]})
        baseline_items = [item for name in ("similar", "orthogonal", "wedge", "random") for item in result["methods"][name]["selected"]]
        self.assertTrue(any(item["contains_input"] for item in baseline_items))
        self.assertTrue(any("not exclusion-filtered" in note for note in result["notes"]))
        self.assertTrue(any("seed" in note and "engine" in note and "random" in note for note in result["notes"]))

    def test_dict_rows_use_the_validated_corpus_path(self):
        dict_rows = [{"a": a, "b": b, "line": line} for a, b, line in self.corpus]
        result = compare_methods("湯気", "睡眠", dict_rows, make_encoder("hash"), limit=2)
        self.assertEqual(result["corpus_size"], len(dict_rows))

    def test_jaccard_matrix_is_symmetric_and_has_unit_diagonal(self):
        result = self.compare()
        for left in METHOD_NAMES:
            self.assertEqual(result["jaccard"][left][left], 1.0)
            for right in METHOD_NAMES:
                self.assertEqual(result["jaccard"][left][right], result["jaccard"][right][left])

    def test_blind_sheet_is_method_and_score_blind_and_key_maps_back(self):
        result = self.compare(limit=4, seed=11)
        with tempfile.TemporaryDirectory() as first_tmp, tempfile.TemporaryDirectory() as second_tmp:
            first = write_blind_materials(result, self.corpus, first_tmp)
            second = write_blind_materials(result, self.corpus, second_tmp)
            first_sheet = Path(first["sheet"]).read_text(encoding="utf-8")
            second_sheet = Path(second["sheet"]).read_text(encoding="utf-8")
            first_key = json.loads(Path(first["key"]).read_text(encoding="utf-8"))
            second_key = json.loads(Path(second["key"]).read_text(encoding="utf-8"))

        self.assertEqual(first_sheet, second_sheet)
        self.assertEqual(first_key, second_key)
        self.assertIn("未実施", first_sheet)
        self.assertIn("Input A: `H2O MEMORY`", first_sheet)
        self.assertIn("Input B: `TRAJEXA`", first_sheet)
        for forbidden in (*METHOD_NAMES, "score", "スコア"):
            self.assertNotIn(forbidden, first_sheet)
        self.assertIn("Unexpectedness", first_sheet)
        self.assertIn("Coherence", first_sheet)
        self.assertIn("Emergence", first_sheet)
        self.assertIn("Testability", first_sheet)
        self.assertEqual(first_key["seed"], 11)
        self.assertEqual(first_key["corpus_id"], result["corpus_id"])
        selected_indices = {
            item["corpus_index"]
            for method in result["methods"].values()
            for item in method["selected"]
        }
        self.assertEqual(
            {item["corpus_index"] for item in first_key["entries"].values()},
            selected_indices,
        )
        for blind_id, item in first_key["entries"].items():
            self.assertRegex(blind_id, r"^E\d{2}$")
            expected_methods = [
                name
                for name in METHOD_NAMES
                if item["corpus_index"]
                in {selected["corpus_index"] for selected in result["methods"][name]["selected"]}
            ]
            self.assertEqual(item["methods"], expected_methods)

    def test_deterministic_and_cli_emits_utf8_under_cp932(self):
        self.assertEqual(self.compare(seed=13), self.compare(seed=13))
        env = dict(os.environ, PYTHONIOENCODING="cp932")
        env.pop("PYTHONUTF8", None)
        command = [
            sys.executable,
            str(SCRIPTS / "compare_third_concept.py"),
            "--A",
            "湯気",
            "--B",
            "睡眠",
            "--corpus",
            str(self.corpus_path),
            "--encoder",
            "hash",
            "--limit",
            "2",
            "--format",
            "json",
        ]
        completed = subprocess.run(command, cwd=ROOT, capture_output=True, env=env, check=True)
        output = json.loads(completed.stdout.decode("utf-8"))
        self.assertEqual(output["input"], {"A": "湯気", "B": "睡眠"})


class ComparisonV3Test(unittest.TestCase):
    def setUp(self):
        self.corpus_path = FIXTURES / "synthetic_corpus.json"
        self.corpus = load_corpus(self.corpus_path)

    def test_evaluation_schema_preserves_legacy_top_level_contract(self):
        result = compare_methods(
            "H2O MEMORY", "TRAJEXA", self.corpus, make_encoder("hash"),
            limit=3, seed=7, corpus_path=self.corpus_path, null_samples=12,
        )
        self.assertEqual(tuple(result["methods"]), METHOD_NAMES)
        self.assertEqual(result["params"], {"limit": 3, "seed": 7})
        self.assertIn("evaluation", result)
        evaluation = result["evaluation"]
        self.assertEqual(
            set(evaluation),
            {"params", "baseline_source", "methods", "ratio_means", "jaccard", "wedge_ablation", "null", "excluded_count", "eligible_count", "notes"},
        )
        self.assertEqual(set(evaluation["methods"]), {"third-concept-ratio"})
        self.assertEqual(set(evaluation["methods"]["third-concept-ratio"]["means"]), set(comparison.METRIC_NAMES))
        self.assertEqual(set(evaluation["jaccard"]), set(METHOD_NAMES) | {"third-concept-ratio"})
        json.dumps(result, ensure_ascii=False, allow_nan=False)

    def test_stats_helpers_cover_ties_bounds_and_undefined_cases(self):
        kendall_tau_b = getattr(comparison, "kendall_tau_b")
        midrank_percentile = getattr(comparison, "midrank_percentile")
        self.assertEqual(kendall_tau_b([1, 2, 3], [1, 2, 3]), 1.0)
        self.assertEqual(kendall_tau_b([1, 2, 3], [3, 2, 1]), -1.0)
        self.assertEqual(kendall_tau_b([1, 1, 2], [1, 1, 3]), 1.0)
        # One joint tie is ignored; P=4 and Q=1, so tau_b = (4-1) / 5.
        self.assertAlmostEqual(kendall_tau_b([1, 1, 2, 3], [1, 1, 3, 2]), 3 / 5)
        # P=2, Q=0, T_x=1, T_y=0, so tau_b = 2 / sqrt((2+0)*(2+1)).
        self.assertAlmostEqual(kendall_tau_b([1, 1, 2], [1, 2, 3]), 2 / np.sqrt(6))
        self.assertIsNone(kendall_tau_b([1, 1, 1], [1, 2, 3]))
        self.assertIsNone(kendall_tau_b([1], [1]))
        self.assertEqual(midrank_percentile([2, 2, 2], 2), 50.0)
        self.assertEqual(midrank_percentile([2, 2, 2], 3), 100.0)
        self.assertEqual(midrank_percentile([2, 2, 2], 1), 0.0)

    def _ablation_result(self, candidates):
        A = [1.0, 0.0, 0.0]
        B = [0.0, 1.0, 0.0]
        compensation = -np.sum(np.asarray([A, B, *candidates]), axis=0)
        lines = [f"node-{index}" for index in range(len(candidates))] + ["counter-vector"]
        vectors = {"left-anchor": A, "right-anchor": B, **dict(zip(lines, [*candidates, compensation]))}
        return compare_methods(
            "left-anchor", "right-anchor", lines, FixedEncoder(vectors), limit=1, seed=3,
            p_min=0.0, null_samples=8,
        )

    def test_wedge_ratio_ablation_can_differ_from_balance(self):
        result = self._ablation_result([
            [0.141421, 0.141421, 0.979796],
            [0.7, 0.1, 0.707107],
        ])
        self.assertFalse(result["evaluation"]["wedge_ablation"]["third-concept-ratio"]["topk_identical_to_balance"])

    def test_wedge_ratio_ablation_can_match_balance(self):
        result = self._ablation_result([
            [0.5, 0.5, 0.707107],
            [0.141421, 0.141421, 0.979796],
        ])
        self.assertTrue(result["evaluation"]["wedge_ablation"]["third-concept-ratio"]["topk_identical_to_balance"])

    def test_wedge_ablation_tau_uses_each_score_against_the_named_baseline(self):
        candidates = [
            [0.10, 0.60, 0.793725],
            [0.35, 0.35, 0.868907],
            [0.55, 0.20, 0.811788],
            [0.45, 0.45, 0.771362],
        ]
        A = np.array([1.0, 0.0, 0.0])
        B = np.array([0.0, 1.0, 0.0])
        compensation = -np.sum(np.asarray([A, B, *candidates]), axis=0)
        lines = [f"tau-node-{index}" for index in range(len(candidates))] + ["tau-counter-vector"]
        vectors = {"tau-left": A, "tau-right": B, **dict(zip(lines, [*candidates, compensation]))}
        captured = []
        original = comparison.shared_geometry_space

        def recording_space(*args, **kwargs):
            space = original(*args, **kwargs)
            captured.append(space)
            return space

        with mock.patch.object(comparison, "shared_geometry_space", side_effect=recording_space):
            result = compare_methods(
                "tau-left", "tau-right", lines, FixedEncoder(vectors), limit=2, seed=3,
                p_min=0.0, null_samples=8,
            )

        self.assertEqual(len(captured), 1)
        space = captured[0]
        metrics = comparison.geometry(
            space["a_vector"], space["b_vector"], space["passage_vectors"],
            mu=space["mu"], p_min=0.0,
        )
        excluded = comparison.exclusion_mask(space, "tau-left", "tau-right")
        eligible = [index for index, value in enumerate(excluded) if not value]
        expected_by_method = {
            "third-concept": "score_third",
            "third-concept-ratio": "score_ratio",
        }
        baselines = {
            "kendall_tau_b_vs_balance": "balance",
            "kendall_tau_b_vs_no_wedge": "score_no_wedge",
        }
        for method, score_name in expected_by_method.items():
            for output_name, baseline_name in baselines.items():
                expected = comparison.kendall_tau_b(
                    [metrics[index][score_name] for index in eligible],
                    [metrics[index][baseline_name] for index in eligible],
                )
                self.assertNotEqual(comparison._round_optional(expected), 1.0)
                self.assertEqual(
                    result["evaluation"]["wedge_ablation"][method][output_name],
                    comparison._round_optional(expected),
                )

    def test_plain_text_corpora_work_in_api_and_cli(self):
        plain = ["PLAIN-LINE-01", {"line": "PLAIN-LINE-02"}, "PLAIN-LINE-03"]
        result = compare_methods("steam", "sleep", plain, make_encoder("hash"), limit=2, null_samples=8)
        self.assertEqual(tuple(result["methods"]), ("similar", "random", "third-concept", "mean", "third-concept-no-wedge"))
        self.assertEqual(result["evaluation"]["baseline_source"], "shared-space")
        self.assertNotIn("orthogonal", result["methods"])
        self.assertTrue(any("orthogonal" in note and "wedge" in note for note in result["evaluation"]["notes"]))
        for method in result["methods"].values():
            self.assertEqual(set(method["selected"][0]), {"corpus_index", "line", "metrics", "contains_input"})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "plain.json"
            path.write_text(json.dumps(plain), encoding="utf-8")
            completed = subprocess.run(
                [sys.executable, str(SCRIPTS / "compare_third_concept.py"), "--A", "steam", "--B", "sleep", "--corpus", str(path), "--encoder", "hash", "--limit", "2", "--null-samples", "8"],
                cwd=ROOT, capture_output=True, check=False,
            )
        self.assertEqual(completed.returncode, 0, completed.stderr.decode("utf-8", errors="replace"))
        self.assertEqual(json.loads(completed.stdout.decode("utf-8"))["evaluation"]["baseline_source"], "shared-space")

    def test_sentence_input_keeps_original_for_geometry_and_normalises_baselines(self):
        model = RecordingModel()
        encoder = make_encoder("semantic", "intfloat/multilingual-e5-small", model=model)
        result = compare_methods(
            "first sentence / original", "second sentence original", self.corpus[:3], encoder,
            limit=2, input_kind="sentence", p_min=0.0, null_samples=6,
        )
        flattened = [text for call in model.calls for text in call]
        self.assertIn("query: first sentence / original", flattened)
        self.assertIn("query: second sentence original", flattened)
        self.assertTrue(any("first_sentence_original" in text for text in flattened))
        self.assertEqual(result["input_normalization"]["applies_to"], "engine-baselines-only")

    def test_exclude_filters_third_concept_family_and_marks_baselines(self):
        result = compare_methods(
            "湯気", "睡眠", self.corpus, make_encoder("hash"), limit=len(self.corpus),
            exclude=["発酵"], p_min=0.0, null_samples=8,
        )
        family = ("third-concept", "mean", "third-concept-no-wedge")
        for name in family:
            self.assertTrue(all("発酵" not in self.corpus[item["corpus_index"]][2] for item in result["methods"][name]["selected"]))
        ratio = result["evaluation"]["methods"]["third-concept-ratio"]["selected"]
        self.assertTrue(all("発酵" not in self.corpus[item["corpus_index"]][2] for item in ratio))
        baseline_items = [item for name in METHOD_NAMES[:4] for item in result["methods"][name]["selected"]]
        self.assertTrue(any("発酵" in self.corpus[item["corpus_index"]][2] and item["contains_input"] for item in baseline_items))

    def test_compare_pair_shares_mu_pool_and_records_conditions(self):
        compare_pair = getattr(comparison, "compare_pair")
        captured = []
        original = comparison.shared_geometry_space

        def recording_space(*args, **kwargs):
            space = original(*args, **kwargs)
            captured.append(np.asarray(space["mu"]))
            return space

        with mock.patch.object(comparison, "shared_geometry_space", side_effect=recording_space):
            result = compare_pair(
                ("steam", "sleep"), ("steam rises slowly", "sleep restores memory"),
                self.corpus, make_encoder("hash"), 2, 5, self.corpus_path, ["発酵"], None, 8,
            )
        self.assertEqual(result["mode"], "word-vs-sentence")
        self.assertGreaterEqual(len(captured), 2)
        for mu in captured[1:]:
            self.assertTrue(np.allclose(captured[0], mu))
        self.assertEqual(
            result["paired"]["shared_conditions"]["mu_query_texts"],
            ["steam", "sleep", "steam rises slowly", "sleep restores memory"],
        )
        expected_methods = set(result["word"]["methods"]) | {"third-concept-ratio"}
        self.assertEqual(set(result["paired"]["per_method"]), expected_methods)
        self.assertEqual(result["word"]["evaluation"]["eligible_count"], result["sentence"]["evaluation"]["eligible_count"])

    def test_compare_pair_canonicalises_all_mu_query_texts_once(self):
        compare_pair = getattr(comparison, "compare_pair")
        captured = []
        original = comparison.shared_geometry_space

        def recording_space(*args, **kwargs):
            space = original(*args, **kwargs)
            captured.append(space)
            return space

        with mock.patch.object(comparison, "shared_geometry_space", side_effect=recording_space):
            result = compare_pair(
                ("H2O MEMORY", "TRAJEXA"),
                (" 湯気が窓を曇らせる ", " 眠る前に灯りを消す "),
                self.corpus, make_encoder("hash"), 2, 5, self.corpus_path, [], 0.0, 8,
            )

        self.assertEqual(len(captured), 2)
        self.assertTrue(np.allclose(captured[0]["mu"], captured[1]["mu"]))
        self.assertEqual(
            result["paired"]["shared_conditions"]["mu_query_texts"],
            ["H2O_MEMORY", "TRAJEXA", "湯気が窓を曇らせる", "眠る前に灯りを消す"],
        )

    def test_compare_pair_tau_uses_the_run_spaces_and_metrics(self):
        compare_pair = getattr(comparison, "compare_pair")
        captured = []
        original = comparison.shared_geometry_space

        def recording_space(*args, **kwargs):
            space = original(*args, **kwargs)
            captured.append(space)
            return space

        with mock.patch.object(comparison, "shared_geometry_space", side_effect=recording_space):
            result = compare_pair(
                ("H2O MEMORY", "TRAJEXA"),
                (" 湯気が窓を曇らせる ", " 眠る前に灯りを消す "),
                self.corpus, make_encoder("hash"), 2, 5, self.corpus_path, [], 0.0, 8,
            )

        self.assertEqual(len(captured), 2)
        eligible = range(len(self.corpus))
        metrics = [
            comparison.geometry(
                space["a_vector"], space["b_vector"], space["passage_vectors"],
                mu=space["mu"], p_min=0.0,
            )
            for space in captured
        ]
        expected = comparison.kendall_tau_b(
            [metrics[0][index]["score_third"] for index in eligible],
            [metrics[1][index]["score_third"] for index in eligible],
        )
        self.assertEqual(
            result["paired"]["per_method"]["third-concept"]["kendall_tau_b"],
            comparison._round_optional(expected),
        )

    def test_compare_pair_uses_union_of_word_and_sentence_exclusions(self):
        compare_pair = getattr(comparison, "compare_pair")
        corpus = ["sentence-only", "eligible-one", "eligible-two"]
        vectors = {
            "word-left": [1.0, 0.0, 0.0, 0.0],
            "word-right": [0.0, 1.0, 0.0, 0.0],
            "prefix sentence-only suffix": [0.0, 0.0, 1.0, 0.0],
            "sentence-right": [0.0, 0.0, 0.0, 1.0],
            "sentence-only": [0.5, 0.5, 0.5, 0.5],
            "eligible-one": [0.6, 0.2, 0.3, 0.7],
            "eligible-two": [0.2, 0.7, 0.6, 0.3],
        }
        result = compare_pair(
            ("word-left", "word-right"),
            ("prefix sentence-only suffix", "sentence-right"),
            corpus, FixedEncoder(vectors), len(corpus), 4, None, [], 0.0, 7,
        )

        family = ("third-concept", "mean", "third-concept-no-wedge")
        for run_name in ("word", "sentence"):
            run = result[run_name]
            for method in family:
                selected = run["methods"][method]["selected"]
                self.assertNotIn(0, [item["corpus_index"] for item in selected])
            ratio = run["evaluation"]["methods"]["third-concept-ratio"]["selected"]
            self.assertNotIn(0, [item["corpus_index"] for item in ratio])
            self.assertEqual(run["evaluation"]["eligible_count"], 2)
        self.assertEqual(result["paired"]["shared_conditions"]["eligible_count"], 2)

    def test_paired_cli_rejects_sheet_dir(self):
        completed = subprocess.run(
            [
                sys.executable, str(SCRIPTS / "compare_third_concept.py"),
                "--A", "H2O MEMORY", "--B", "TRAJEXA",
                "--A-sentence", "steam sentence", "--B-sentence", "sleep sentence",
                "--corpus", str(self.corpus_path), "--sheet-dir", "unused",
            ],
            cwd=ROOT, capture_output=True, check=False,
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("--sheet-dir", completed.stderr.decode("utf-8", errors="replace"))

    def test_aggregate_only_cli_removes_all_corpus_markers(self):
        pair = [
            {"a": "ZQX-A-01", "b": "ZQX-B-01", "line": "ZQX-LINE-01"},
            {"a": "ZQX-A-02", "b": "ZQX-B-02", "line": "ZQX-LINE-02"},
        ]
        plain = [{"line": "ZQX-PLAIN-01"}, {"line": "ZQX-PLAIN-02"}]
        with tempfile.TemporaryDirectory() as tmp:
            cases = []
            for name, corpus, extra in (
                ("pair", pair, []),
                ("plain", plain, []),
                ("paired", pair, ["--A-sentence", "steam sentence", "--B-sentence", "sleep sentence"]),
            ):
                path = Path(tmp) / f"{name}.json"
                path.write_text(json.dumps(corpus), encoding="utf-8")
                base = [sys.executable, str(SCRIPTS / "compare_third_concept.py"), "--A", "steam", "--B", "sleep", "--corpus", str(path), "--limit", "2", "--null-samples", "5", *extra]
                visible = subprocess.run(base, cwd=ROOT, capture_output=True, check=True).stdout.decode("utf-8")
                hidden = subprocess.run([*base, "--aggregate-only"], cwd=ROOT, capture_output=True, check=True).stdout.decode("utf-8")
                cases.append((corpus, visible, hidden))
            invalid = subprocess.run([*base, "--aggregate-only", "--format", "markdown"], cwd=ROOT, capture_output=True, check=False)
        for corpus, visible, hidden in cases:
            markers = [value for row in corpus for value in row.values()]
            self.assertTrue(any(marker in visible for marker in markers))
            self.assertTrue(all(marker not in hidden for marker in markers))
            self.assertTrue(json.loads(hidden)["aggregate_only"])
        self.assertEqual(invalid.returncode, 2)

    def test_aggregate_only_cli_rejects_sheet_dir_before_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            sheet_dir = Path(tmp) / "blind-output"
            completed = subprocess.run(
                [
                    sys.executable, str(SCRIPTS / "compare_third_concept.py"),
                    "--A", "steam", "--B", "sleep", "--corpus", str(self.corpus_path),
                    "--aggregate-only", "--sheet-dir", str(sheet_dir), "--null-samples", "5",
                ],
                cwd=ROOT, capture_output=True, check=False,
            )
            self.assertEqual(completed.returncode, 2)
            self.assertIn(
                "--aggregate-only cannot be used with --sheet-dir",
                completed.stderr.decode("utf-8", errors="replace"),
            )
            self.assertFalse(sheet_dir.exists())

    def test_evaluation_is_deterministic_for_same_seed(self):
        kwargs = {"limit": 3, "seed": 17, "p_min": None, "null_samples": 12}
        first = compare_methods("湯気", "睡眠", self.corpus, make_encoder("hash"), **kwargs)
        second = compare_methods("湯気", "睡眠", self.corpus, make_encoder("hash"), **kwargs)
        self.assertEqual(first["evaluation"], second["evaluation"])


if __name__ == "__main__":
    unittest.main()
