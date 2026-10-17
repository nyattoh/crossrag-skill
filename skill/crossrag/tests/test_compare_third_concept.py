"""Contract tests for the seven-way third-concept comparison."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
