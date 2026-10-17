"""Contract and regression tests for the third-concept PoC."""
from __future__ import annotations

import base64
import json
import os
import re
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

from engine import DEFAULT_CORPUS, HashEncoder, run  # noqa: E402
from third_concept import (  # noqa: E402
    auto_p_min,
    build_hypothesis_prompt,
    exclusion_mask,
    explore,
    geometry,
    load_hypotheses,
    make_encoder,
    null_geometry,
    rank_entries,
    shared_geometry_space,
)
import third_concept as third_concept_module  # noqa: E402


def load_rows(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    return [tuple((row["a"], row["b"], row["line"])) if isinstance(row, dict) else tuple(row) for row in data]


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


class LegacyRegressionTest(unittest.TestCase):
    def test_engine_and_cli_match_snapshot(self):
        snapshot = json.loads((FIXTURES / "legacy_run_snapshot.json").read_text(encoding="utf-8"))
        corpora = {
            "default": DEFAULT_CORPUS,
            "example": load_rows(ROOT / "references" / "example-corpus.json"),
        }
        for corpus_name, corpus in corpora.items():
            for method in ("similar", "orthogonal", "wedge", "random"):
                expected = snapshot[corpus_name][method]
                self.assertEqual(run("犬、カレー、夕暮れ、帰宅", corpus, method, 0.55, 0.8, 5, HashEncoder()), expected)
                command = [sys.executable, str(SCRIPTS / "run.py"), "--context", "犬、カレー、夕暮れ、帰宅", "--method", method, "--limit", "5"]
                if corpus_name == "example":
                    command.extend(["--corpus", str(ROOT / "references" / "example-corpus.json")])
                actual = json.loads(subprocess.run(command, cwd=ROOT, capture_output=True, check=True).stdout.decode("utf-8"))
                for key in ("task_mode", "evidence_type", "instruction"):
                    actual.pop(key)
                self.assertEqual(actual, expected)


class EncoderAndGeometryTest(unittest.TestCase):
    def test_e5_prefixes_and_no_double_prefix(self):
        model = RecordingModel()
        encoder = make_encoder("semantic", "intfloat/multilingual-e5-small", model=model)
        encoder.encode_query(["A", "query: B"])
        encoder.encode_passage(["entry", "passage: ready"])
        encoder.encode(["concept"])
        self.assertEqual(model.calls[0], ["query: A", "query: B"])
        self.assertEqual(model.calls[1], ["passage: entry", "passage: ready"])
        self.assertEqual(model.calls[2], ["query: concept"])
        self.assertEqual(encoder.prefix, {"query": "query: ", "passage": "passage: "})

    def test_hash_encoder_sends_raw_text(self):
        encoder = make_encoder("hash")
        self.assertIsNone(encoder.prefix)
        self.assertTrue(np.array_equal(encoder.encode_query(["query: raw"]), HashEncoder().encode(["query: raw"])))

    def test_geometry_known_vectors(self):
        a = np.array([1.0, 0.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0, 0.0])
        bisector = np.array([1.0, 1.0, 0.0, 0.0]) / np.sqrt(2.0)
        rows = geometry(a, b, np.asarray([bisector, a, [0.5, 0.5, np.sqrt(0.5), 0.0]]))
        self.assertAlmostEqual(rows[0]["plane_projection"], 1.0)
        self.assertAlmostEqual(rows[0]["residual"], 0.0)
        self.assertAlmostEqual(rows[0]["emergence_norm"], 1.0)
        self.assertAlmostEqual(rows[1]["emergence_norm"], 0.0)
        self.assertAlmostEqual(rows[2]["balance"], 1.0)

    def test_balance_ignores_projection_pointing_away_from_A_and_B(self):
        a = np.array([1.0, 0.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0, 0.0])
        towards = [0.5, 0.5, np.sqrt(0.5), 0.0]
        away = [-0.5, -0.5, np.sqrt(0.5), 0.0]
        rows = geometry(a, b, np.asarray([towards, away]))
        self.assertAlmostEqual(rows[0]["balance"], 1.0)
        self.assertAlmostEqual(rows[1]["plane_projection"], rows[0]["plane_projection"])
        self.assertEqual(rows[1]["balance"], 0.0)
        self.assertEqual(rows[1]["score_third"], 0.0)

    def test_geometry_warns_for_degenerate_plane(self):
        rows = geometry([1, 0, 0], [1, 1e-10, 0], [[0, 1, 0]])
        self.assertEqual(rows[0]["emergence_norm"], 0.0)
        self.assertTrue(rows[0]["warnings"])

    def test_geometry_treats_near_parallel_plane_as_degenerate_by_one_minus_rho(self):
        rows = geometry([1, 0, 0], [1, 1e-4, 0], [[2, 1e-4, 0]])
        self.assertEqual(rows[0]["emergence_norm"], 0.0)
        self.assertTrue(rows[0]["warnings"])

    def test_emergence_requires_positive_similarity_to_both_inputs(self):
        rows = geometry([1, 0, 0], [0, 1, 0], [[1, -1, 0]])
        self.assertAlmostEqual(rows[0]["plane_projection"], 1.0)
        self.assertLess(rows[0]["s_B"], 0.0)
        self.assertEqual(rows[0]["emergence_norm"], 0.0)

    def test_geometry_supports_negative_input_correlation(self):
        rows = geometry([1, 0, 0], [-0.5, np.sqrt(0.75), 0], [[0.5, np.sqrt(0.75), 0]])
        self.assertLess(float(np.dot([1, 0, 0], [-0.5, np.sqrt(0.75), 0])), 0.0)
        self.assertGreater(rows[0]["emergence_norm"], 0.0)


class ExplorationContractTest(unittest.TestCase):
    def setUp(self):
        self.corpus_path = FIXTURES / "synthetic_corpus.json"
        self.corpus = load_rows(self.corpus_path)
        self.encoder = make_encoder("hash")

    def test_rank_methods(self):
        for method in ("third-concept", "third-concept-no-wedge", "mean"):
            ranked = rank_entries("湯気", "睡眠", self.corpus, self.encoder, method, 3)
            self.assertEqual(len(ranked), 3)
            self.assertIn("scores", ranked[0])

    def test_limit_must_be_positive(self):
        for limit in (0, -1):
            with self.subTest(limit=limit), self.assertRaisesRegex(ValueError, "limit must be at least 1"):
                explore("湯気", "睡眠", self.corpus, self.encoder, limit=limit)

    def test_excluded_count_does_not_depend_on_limit(self):
        small = explore("ヨーグルト", "梅シロップ", self.corpus, self.encoder, limit=1)
        large = explore("ヨーグルト", "梅シロップ", self.corpus, self.encoder, limit=20)
        self.assertEqual(small["params"]["excluded_inputs"], large["params"]["excluded_inputs"])

    def test_rank_entries_excludes_text_and_raw_cosine_input_matches(self):
        corpus = [("safe-a", "safe-b", "safe-line"), ("A", "other", "text")]
        texts = ["safe-aとsafe-b。safe-line", "Aとother。text"]
        encoder = FixedEncoder(
            {
                "A": [1, 0, 0],
                "B": [0, 1, 0],
                "safe-a": [0, 0, 1],
                "safe-b": [1, 1, 0],
                "other": [0, 1, 1],
                texts[0]: [0.99, 0.01, 0],
                texts[1]: [0, 0, 1],
            }
        )
        for method in ("third-concept", "third-concept-no-wedge", "mean"):
            self.assertEqual(rank_entries("A", "B", corpus, encoder, method, 5), [])

    def test_plain_text_rows_are_passages_without_pair_labels(self):
        # FixedEncoder raises KeyError if a passage is wrapped as "aとb。line".
        encoder = FixedEncoder({"A": [1, 0, 0], "B": [0, 1, 0], "文一": [0.5, 0.5, 0.7], "文二": [0.2, 0.6, 0.77]})
        data = explore("A", "B", ["文一", {"line": "文二"}], encoder, limit=5)
        self.assertEqual({candidate["origin"] for candidate in data["candidates"]}, {"retrieved"})
        self.assertEqual({candidate["concept"] for candidate in data["candidates"]}, {"文一", "文二"})
        self.assertEqual(data["params"]["vocab_size"], 0)

    def test_misspelled_pair_keys_are_rejected_not_read_as_plain_text(self):
        with self.assertRaisesRegex(ValueError, "corpus item"):
            explore("湯気", "睡眠", [{"aa": "夢", "bb": "記憶", "line": "夜の記録"}], self.encoder)

    def test_json_schema(self):
        data = explore("湯気", "睡眠", self.corpus, self.encoder, limit=5, corpus_path=self.corpus_path)
        self.assertTrue({"mode", "input", "encoder", "corpus_id", "corpus_size", "params", "origin_legend", "candidates", "warnings", "disclaimer"} <= data.keys())
        self.assertRegex(data["corpus_id"], r"^[0-9a-f]{16}$")
        self.assertTrue(data["disclaimer"])
        self.assertEqual(data["params"]["vocab_size"], 0)
        self.assertIsNone(data["encoder"]["applied_to"])
        required = {"id", "concept", "origin", "generation_method", "relation_to_A_alone", "relation_to_B_alone", "why_requires_AB", "uncertainty", "falsification_or_check", "scores", "source"}
        score_keys = {"s_A", "s_B", "plane_projection", "residual", "emergence_norm", "balance", "unexpectedness_h", "coherence_h", "score_third"}
        for candidate in data["candidates"]:
            self.assertTrue(required <= candidate.keys())
            self.assertIn(candidate["origin"], {"retrieved", "math", "llm_hypothesis", "idea"})
            self.assertEqual(set(candidate["scores"]["heuristic_math"]), score_keys)

    def test_input_concepts_and_substrings_are_excluded(self):
        hypotheses = {
            "input": {"A": "湯気", "B": "睡眠"},
            "generator": {"provider": "test", "model": "fake", "effort": "none", "prompt_sha256": "0", "date": "2026-10-18"},
            "hypotheses": [{"concept": "湯気", "kind": "idea", "explanation": "x", "relation_to_A_alone": "x", "relation_to_B_alone": "x", "why_requires_AB": "x", "uncertainty": "x", "falsification_or_check": "x", "llm_self_assessment": {"unexpectedness": 1, "coherence": 1, "emergence": 1, "testability": 1}}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "hypotheses.json"
            path.write_text(json.dumps(hypotheses, ensure_ascii=False), encoding="utf-8")
            loaded = load_hypotheses(path, "湯気", "睡眠")
            data = explore("湯気", "睡眠", self.corpus, self.encoder, limit=20, hypotheses=loaded, corpus_path=self.corpus_path)
        concepts = [candidate["concept"] for candidate in data["candidates"]]
        self.assertNotIn("湯気", concepts)
        self.assertNotIn("朝湯気立ち", concepts)
        self.assertGreater(data["params"]["excluded_inputs"], 0)

    def test_origins_are_not_mixed(self):
        offline = explore("湯気", "睡眠", self.corpus, self.encoder, limit=5, corpus_path=self.corpus_path)
        self.assertFalse({"llm_hypothesis", "idea"} & {item["origin"] for item in offline["candidates"]})
        self.assertTrue(all(item["scores"]["llm_self_assessment"] is None for item in offline["candidates"] if item["origin"] in {"math", "retrieved"}))
        hypotheses = {
            "input": {"A": "湯気", "B": "睡眠"},
            "generator": {"provider": "test", "model": "fake", "effort": "none", "prompt_sha256": "0", "date": "2026-10-18"},
            "hypotheses": [{"concept": "夜間蒸気計", "kind": "llm_hypothesis", "explanation": "x", "relation_to_A_alone": "x", "relation_to_B_alone": "x", "why_requires_AB": "x", "uncertainty": "x", "falsification_or_check": "x", "llm_self_assessment": {"unexpectedness": 2, "coherence": 3, "emergence": 4, "testability": 5}}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "hypotheses.json"
            path.write_text(json.dumps(hypotheses, ensure_ascii=False), encoding="utf-8")
            loaded = load_hypotheses(path, "湯気", "睡眠")
            combined = explore("湯気", "睡眠", self.corpus, self.encoder, hypotheses=loaded, corpus_path=self.corpus_path)
        llm_items = [item for item in combined["candidates"] if item["origin"] == "llm_hypothesis"]
        self.assertEqual([item["concept"] for item in llm_items], ["夜間蒸気計"])
        self.assertEqual(llm_items[0]["scores"]["llm_self_assessment"]["testability"], 5)

    def test_shared_mean_keeps_same_concept_scores_equal_and_hypothesis_stable(self):
        def hypothesis_items(concepts):
            return {
                "input": {"A": "湯気", "B": "睡眠"},
                "generator": {"provider": "test", "model": "fake", "effort": "none", "prompt_sha256": "0", "date": "2026-10-18"},
                "hypotheses": [
                    {"concept": concept, "kind": "llm_hypothesis", "explanation": "x", "relation_to_A_alone": "x", "relation_to_B_alone": "x", "why_requires_AB": "x", "uncertainty": "x", "falsification_or_check": "x", "llm_self_assessment": None}
                    for concept in concepts
                ],
            }

        one = explore("湯気", "睡眠", self.corpus, self.encoder, limit=20, vocab=["夜間蒸気計"], hypotheses=hypothesis_items(["夜間蒸気計"]))
        two = explore("湯気", "睡眠", self.corpus, self.encoder, limit=20, vocab=["夜間蒸気計"], hypotheses=hypothesis_items(["夜間蒸気計", "温度夢日誌"]))

        def score(data, origin, concept):
            return next(item["scores"]["heuristic_math"] for item in data["candidates"] if item["origin"] == origin and item["concept"] == concept)

        self.assertEqual(score(one, "math", "夜間蒸気計"), score(one, "llm_hypothesis", "夜間蒸気計"))
        self.assertEqual(score(one, "llm_hypothesis", "夜間蒸気計"), score(two, "llm_hypothesis", "夜間蒸気計"))

    def test_zero_emergence_does_not_claim_geometric_ab_dependence(self):
        data = explore("湯気", "睡眠", self.corpus, self.encoder, limit=20)
        zero_items = [item for item in data["candidates"] if item["scores"]["heuristic_math"]["emergence_norm"] == 0.0]
        self.assertTrue(zero_items)
        self.assertTrue(all("幾何上は A∧B への依存を示さない" in item["why_requires_AB"] for item in zero_items))

    def test_explore_encodes_corpus_entries_as_passages(self):
        model = RecordingModel()
        encoder = make_encoder("semantic", "intfloat/multilingual-e5-small", model=model)
        explore("湯気", "睡眠", self.corpus[:2], encoder, limit=1)
        expected = [f"passage: {a}と{b}。{line}" for a, b, line in self.corpus[:2]]
        self.assertIn(expected, model.calls)

    def test_hypothesis_input_mismatch_and_forbidden_word_are_rejected(self):
        base = {
            "input": {"A": "湯気", "B": "別入力"},
            "generator": {"provider": "test", "model": "fake"},
            "hypotheses": [],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "hypotheses.json"
            path.write_text(json.dumps(base, ensure_ascii=False), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not match"):
                load_hypotheses(path, "湯気", "睡眠")
            base["input"]["B"] = "睡眠"
            base["hypotheses"] = [{"concept": "候補", "kind": "idea", "explanation": "confirmed", "relation_to_A_alone": "x", "relation_to_B_alone": "x", "why_requires_AB": "x", "uncertainty": "x", "falsification_or_check": "x", "llm_self_assessment": None}]
            path.write_text(json.dumps(base, ensure_ascii=False), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "disallowed"):
                load_hypotheses(path, "湯気", "睡眠")

    def test_boolean_llm_self_assessment_is_rejected(self):
        data = {
            "input": {"A": "湯気", "B": "睡眠"},
            "generator": {"provider": "test", "model": "fake"},
            "hypotheses": [{"concept": "候補", "kind": "idea", "explanation": "x", "relation_to_A_alone": "x", "relation_to_B_alone": "x", "why_requires_AB": "x", "uncertainty": "x", "falsification_or_check": "x", "llm_self_assessment": {"unexpectedness": True, "coherence": 2, "emergence": 3, "testability": 4}}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "hypotheses.json"
            path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid llm_self_assessment"):
                load_hypotheses(path, "湯気", "睡眠")

    def test_no_center_changes_geometry_and_is_recorded(self):
        centred = explore("湯気", "睡眠", self.corpus, self.encoder, limit=2, center=True, corpus_path=self.corpus_path)
        raw = explore("湯気", "睡眠", self.corpus, self.encoder, limit=2, center=False, corpus_path=self.corpus_path)
        self.assertTrue(centred["params"]["center"])
        self.assertFalse(raw["params"]["center"])
        self.assertNotEqual(centred["candidates"][0]["scores"]["heuristic_math"], raw["candidates"][0]["scores"]["heuristic_math"])

    def test_forbidden_claim_wording_is_absent(self):
        data = explore("湯気", "睡眠", self.corpus, self.encoder, limit=5, corpus_path=self.corpus_path)
        text = json.dumps(data, ensure_ascii=False)
        self.assertIsNone(re.search(r"(?i)\b(?:verified|proven|confirmed)\b|証明済|検証済", text))


class LeakageAndCliTest(unittest.TestCase):
    @staticmethod
    def forbidden_terms():
        encoded = ("44OQ44OK44OK", "44Gw44Gq44Gq", "YmFuYW5h")
        return [base64.b64decode(value).decode("utf-8") for value in encoded]

    def assert_no_known_answer(self, text):
        lowered = text.casefold()
        for term in self.forbidden_terms():
            self.assertNotIn(term.casefold(), lowered)

    def test_known_answer_does_not_leak(self):
        self.assert_no_known_answer((FIXTURES / "synthetic_corpus.json").read_text(encoding="utf-8"))
        for path in SCRIPTS.iterdir():
            if path.is_file():
                self.assert_no_known_answer(path.read_text(encoding="utf-8", errors="ignore"))
        prompt = build_hypothesis_prompt("ヨーグルト", "梅シロップ")
        self.assert_no_known_answer(prompt)
        command = [sys.executable, str(SCRIPTS / "third_concept_cli.py"), "--A", "ヨーグルト", "--B", "梅シロップ", "--corpus", str(FIXTURES / "synthetic_corpus.json"), "--encoder", "hash"]
        output = subprocess.run(command, cwd=ROOT, capture_output=True, check=True).stdout.decode("utf-8")
        self.assert_no_known_answer(output)
        eval_path = os.environ.get("CROSSRAG_EVAL_CORPUS")
        if eval_path and Path(eval_path).is_file():
            self.assert_no_known_answer(Path(eval_path).read_text(encoding="utf-8"))

    def test_cli_is_utf8_under_cp932_and_emit_prompt(self):
        env = dict(os.environ, PYTHONIOENCODING="cp932")
        env.pop("PYTHONUTF8", None)
        command = [sys.executable, str(SCRIPTS / "third_concept_cli.py"), "--A", "湯気", "--B", "睡眠", "--corpus", str(FIXTURES / "synthetic_corpus.json"), "--encoder", "hash", "--limit", "2"]
        output = subprocess.run(command, cwd=ROOT, capture_output=True, env=env, check=True).stdout
        self.assertEqual(json.loads(output.decode("utf-8"))["input"], {"A": "湯気", "B": "睡眠"})
        prompt = subprocess.run([sys.executable, str(SCRIPTS / "third_concept_cli.py"), "--A", "湯気", "--B", "睡眠", "--emit-prompt"], cwd=ROOT, capture_output=True, check=True).stdout.decode("utf-8")
        self.assertIn("概念A「湯気」", prompt)
        self.assertIn("概念B「睡眠」", prompt)
        self.assertIn('{"hypotheses": [...]}', prompt)

    def test_cli_accepts_plain_text_corpus(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "plain.json"
            path.write_text(json.dumps(["湯気が窓を曇らせる", {"line": "眠る前に灯りを消す"}], ensure_ascii=False), encoding="utf-8")
            command = [sys.executable, str(SCRIPTS / "third_concept_cli.py"), "--A", "雨", "--B", "時計", "--corpus", str(path), "--encoder", "hash"]
            output = subprocess.run(command, cwd=ROOT, capture_output=True, check=True).stdout
        data = json.loads(output.decode("utf-8"))
        self.assertEqual(data["corpus_size"], 2)
        self.assertEqual({candidate["origin"] for candidate in data["candidates"]}, {"retrieved"})

    def test_cli_rejects_misspelled_pair_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "typo.json"
            path.write_text(json.dumps([{"aa": "夢", "bb": "記憶", "line": "夜の記録"}], ensure_ascii=False), encoding="utf-8")
            command = [sys.executable, str(SCRIPTS / "third_concept_cli.py"), "--A", "雨", "--B", "時計", "--corpus", str(path), "--encoder", "hash"]
            completed = subprocess.run(command, cwd=ROOT, capture_output=True)
        self.assertEqual(completed.returncode, 2)
        self.assertIn(b"corpus item", completed.stderr)

    def test_cli_rejects_non_positive_limit(self):
        command = [sys.executable, str(SCRIPTS / "third_concept_cli.py"), "--A", "湯気", "--B", "睡眠", "--limit", "0"]
        completed = subprocess.run(command, cwd=ROOT, capture_output=True)
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn(b"limit must be at least 1", completed.stderr)


class RatioGeometryTest(unittest.TestCase):
    def test_token_geometry_known_vectors_have_joint_support_and_mean_grouping(self):
        from third_concept import token_geometry

        result = token_geometry(
            np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]),
            np.array([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            mu=np.zeros(3),
        )
        expected_joint = (((1 + 1 / np.sqrt(2)) / 2) ** 2) ** 0.25
        expected_single = (1 / 4) ** 0.25
        self.assertAlmostEqual(result[0]["score"], expected_joint)
        self.assertAlmostEqual(result[1]["score"], expected_single)
        self.assertAlmostEqual(result[2]["score"], expected_single)
        self.assertEqual(result[3]["score"], 0.0)
        self.assertGreater(result[0]["score"], result[1]["score"])
        self.assertGreater(result[1]["score"], result[3]["score"])
        self.assertAlmostEqual(result[0]["a_side_support"], 1 / np.sqrt(2))
        self.assertAlmostEqual(result[0]["b_side_support"], 1 / np.sqrt(2))
        self.assertEqual(result[0]["between_direction_support"], 1.0)
        self.assertEqual(result[0]["plane_support"], 1.0)
        self.assertEqual(result[0]["aggregation"], "mean-vector-geometric-v2")
        pooled = token_geometry(
            np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]),
            [np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])],
            mu=np.zeros(3),
        )
        self.assertAlmostEqual(pooled[0]["score"], expected_joint)

    def test_token_geometry_is_scale_and_anchor_order_invariant_and_defines_zero(self):
        from third_concept import token_geometry

        a, b = np.array([1., 0., 0.]), np.array([0., 1., 0.])
        groups = np.array([[1., 1., 0.], [1., 0., 0.], [0., 1., 0.],
                           [0., 0., 1.], [0., 0., 0.]])
        baseline = token_geometry(a, b, groups, mu=np.zeros(3))
        scaled = token_geometry(7 * a, 0.25 * b, groups * np.array([3., 0.2, 8., 2., 4.])[:, None], mu=np.zeros(3))
        swapped = token_geometry(b, a, groups, mu=np.zeros(3))
        keys = ("score", "a_side_support", "b_side_support", "between_direction_support", "plane_support")
        for i in range(len(groups)):
            for key in keys:
                self.assertAlmostEqual(baseline[i][key], scaled[i][key])
                if key in ("score", "between_direction_support", "plane_support"):
                    self.assertAlmostEqual(baseline[i][key], swapped[i][key])
        self.assertEqual(baseline[-1]["score"], 0.0)
        self.assertTrue(all(baseline[-1][key] == 0.0 for key in keys))
        collinear = token_geometry(a, 2 * a, groups[:2], mu=np.zeros(3))
        self.assertEqual([row["score"] for row in collinear], [0.0, 0.0])

    def test_token_wedge_rejects_unknown_method_and_model_prefix_policy_is_fixed(self):
        from third_concept import make_encoder

        unknown = make_encoder("semantic", "unknown/model", model=RecordingModel(), prefix_policy="auto")
        self.assertIsNone(unknown.prefix)
        encoder = make_encoder("semantic", "intfloat/multilingual-e5-small", model=RecordingModel(), prefix_policy="prefix-free")
        encoder.encode_query(["A"])
        self.assertEqual(encoder.model.calls[0], ["A"])

    def test_passage_token_embeddings_keep_full_contextual_text(self):
        class TokenModel:
            def __init__(self):
                self.calls = []

            def encode(self, texts, **kwargs):
                self.calls.append((list(texts), kwargs))
                return [np.asarray([[1., 0., 0.], [0., 1., 0.]]) for _ in texts]

        model = TokenModel()
        encoder = make_encoder("semantic", "unknown/model", model=model)
        vectors = encoder.encode_passage_tokens(["雨音を聞きながら眠る"])
        self.assertEqual(model.calls[0][0], ["雨音を聞きながら眠る"])
        self.assertEqual(model.calls[0][1]["output_value"], "token_embeddings")
        self.assertEqual(vectors[0].shape, (2, 3))

    @staticmethod
    def anchors(rho, dimension=4):
        a = np.zeros(dimension)
        b = np.zeros(dimension)
        a[0] = 1.0
        b[:2] = [rho, np.sqrt(1.0 - rho * rho)]
        return a, b

    def test_emergence_ratio_matches_independent_in_plane_angles(self):
        cases = [
            (-0.6, 1.0, 1.0, 0.4),
            (-0.2, 2.0, 1.0, 1.3),
            (0.3, 1.0, 3.0, 0.8),
            (0.8, 2.0, 5.0, 2.0),
        ]
        for rho, alpha, beta, off_plane in cases:
            with self.subTest(rho=rho, alpha=alpha, beta=beta):
                a, b = self.anchors(rho)
                candidate = alpha * a + beta * b + np.array([0.0, 0.0, off_plane, 0.0])
                row = geometry(a, b, [candidate])[0]
                c = candidate / np.linalg.norm(candidate)
                e2 = (b - rho * a) / np.linalg.norm(b - rho * a)
                q = np.dot(c, a) * a + np.dot(c, e2) * e2
                q /= np.linalg.norm(q)
                expected = 1.0 - max(np.dot(q, a) ** 2, np.dot(q, b) ** 2)
                self.assertTrue(row["between_AB"])
                self.assertAlmostEqual(row["emergence_ratio"], expected)

    def test_ratio_is_independent_of_scale_and_off_plane_component(self):
        a, b = self.anchors(0.3)
        q = 2.0 * a + 3.0 * b
        candidates = [
            q,
            2.0 * q + np.array([0.0, 0.0, 3.0, 0.0]),
            0.5 * q + np.array([0.0, 0.0, 4.0, 0.0]),
            7.0 * q + np.array([0.0, 0.0, 1.0, 0.0]),
        ]
        rows = geometry(a, b, candidates)
        for row in rows[1:]:
            self.assertAlmostEqual(row["emergence_ratio"], rows[0]["emergence_ratio"])
        self.assertGreater(max(row["plane_projection"] for row in rows) - min(row["plane_projection"] for row in rows), 0.2)

    def test_lines_bisectors_and_in_cone_sweep(self):
        for rho in (-0.6, 0.0, 0.3, 0.8):
            with self.subTest(rho=rho):
                a, b = self.anchors(rho, dimension=3)
                bisector = (a + b) / np.linalg.norm(a + b)
                line_a, middle = geometry(a, b, [a, bisector])
                expected = (1.0 - rho) / 2.0
                self.assertEqual(line_a["emergence_ratio"], 0.0)
                self.assertAlmostEqual(middle["emergence_ratio"], expected)
                self.assertAlmostEqual(middle["emergence_ratio_norm"], 1.0)
                theta = np.arccos(rho)
                sweep = [[np.cos(phi), np.sin(phi), 0.0] for phi in np.linspace(0.0, theta, 241)]
                ratios = [row["emergence_ratio"] for row in geometry(a, b, sweep)]
                self.assertLessEqual(max(ratios), middle["emergence_ratio"] + 1e-12)
                self.assertAlmostEqual(max(ratios), middle["emergence_ratio"], places=12)

    def test_cone_gate_rejects_outside_direction_even_when_both_similarities_are_positive(self):
        rho = 0.8
        a, b = self.anchors(rho, dimension=3)
        theta = np.arccos(rho)
        outside = [np.cos(-theta), np.sin(-theta), 0.0]
        row = geometry(a, b, [outside])[0]
        self.assertGreater(row["s_A"], 0.0)
        self.assertGreater(row["s_B"], 0.0)
        self.assertFalse(row["between_AB"])
        self.assertEqual(row["emergence_ratio"], 0.0)
        self.assertGreater(row["emergence_norm"], 0.0)

    def test_ratio_edge_cases_p_floor_and_validation(self):
        degenerate = geometry([1, 0, 0], [1, 1e-10, 0], [[0, 1, 0]])[0]
        self.assertEqual(degenerate["emergence_ratio"], 0.0)
        self.assertFalse(degenerate["between_AB"])
        self.assertTrue(degenerate["warnings"])

        non_positive = geometry([1, 0, 0], [0, 1, 0], [[1, -1, 0], [-1, 1, 0]])
        self.assertTrue(all(row["emergence_ratio"] == 0.0 for row in non_positive))

        candidate = np.array([0.3, 0.3, np.sqrt(0.82)])
        below = geometry([1, 0, 0], [0, 1, 0], [candidate], p_min=0.5)[0]
        self.assertTrue(below["p_below_min"])
        self.assertGreater(below["emergence_ratio"], 0.0)
        self.assertEqual(below["score_ratio"], 0.5 * below["balance"])

        for invalid in (-0.1, 1.5, float("nan")):
            with self.subTest(p_min=invalid), self.assertRaises(ValueError):
                geometry([1, 0, 0], [0, 1, 0], [[1, 1, 0]], p_min=invalid)

    def test_p_min_does_not_change_existing_geometry_keys_or_values(self):
        existing_keys = {
            "s_A", "s_B", "plane_projection", "residual", "emergence_norm", "balance",
            "unexpectedness_h", "coherence_h", "score_third", "s_mean", "score_no_wedge", "warnings",
        }
        default = geometry([1, 0, 0], [0, 1, 0], [[0.4, 0.4, np.sqrt(0.68)]])[0]
        floored = geometry([1, 0, 0], [0, 1, 0], [[0.4, 0.4, np.sqrt(0.68)]], p_min=0.9)[0]
        self.assertEqual({key: default[key] for key in existing_keys}, {key: floored[key] for key in existing_keys})


class NullGeometryTest(unittest.TestCase):
    def setUp(self):
        self.a = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        self.b = np.array([0.2, np.sqrt(0.96), 0.0, 0.0, 0.0])
        self.passages = np.array([
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [0.6, -0.2, 0.1, 0.3, -0.4],
            [-0.3, 0.7, 0.2, -0.1, 0.5],
        ])
        self.mu = np.mean(np.vstack([self.a, self.b, self.passages]), axis=0)

    def test_null_geometry_is_seeded_bounded_and_has_expected_shape(self):
        one = null_geometry(self.a, self.b, self.passages, self.mu, n=37, seed=9)
        again = null_geometry(self.a, self.b, self.passages, self.mu, n=37, seed=9)
        other = null_geometry(self.a, self.b, self.passages, self.mu, n=37, seed=10)
        for kind in ("random", "shuffled"):
            for key in ("p", "emergence_ratio"):
                self.assertEqual(one[kind][key].shape, (37,))
                self.assertTrue(np.array_equal(one[kind][key], again[kind][key]))
                self.assertTrue(np.all((0.0 <= one[kind][key]) & (one[kind][key] <= 1.0)))
        self.assertFalse(np.array_equal(one["random"]["p"], other["random"]["p"]))
        self.assertFalse(np.array_equal(one["shuffled"]["p"], other["shuffled"]["p"]))
        self.assertEqual(auto_p_min(one), float(np.percentile(one["shuffled"]["p"], 95)))

    def test_null_geometry_uses_centred_unit_passages_without_recentering(self):
        n = 11
        seed = 23
        captured = []
        original_geometry = third_concept_module.geometry

        def recording_geometry(a_vec, b_vec, candidates, mu=None, p_min=0.0):
            captured.append(np.asarray(candidates, dtype=float).copy())
            return original_geometry(a_vec, b_vec, candidates, mu=mu, p_min=p_min)

        with mock.patch.object(third_concept_module, "geometry", side_effect=recording_geometry):
            actual = null_geometry(self.a, self.b, self.passages, self.mu, n=n, seed=seed)

        unit = lambda vector: np.asarray(vector, dtype=float) / np.linalg.norm(vector)
        a_c = unit(self.a - self.mu)
        b_c = unit(self.b - self.mu)
        centred_units = np.asarray([unit(vector - self.mu) for vector in self.passages])
        self.assertEqual(len(captured), 2)
        actual_shuffled = captured[1]
        for index, sample in enumerate(actual_shuffled):
            self.assertTrue(
                np.allclose(np.sort(sample), np.sort(centred_units[index % len(centred_units)]))
            )

        random_rows = np.random.default_rng(seed).standard_normal((n, self.a.size))
        random_samples = np.asarray([unit(vector) for vector in random_rows])
        shuffled_rng = np.random.default_rng(seed + 1)
        shuffled_samples = np.asarray([
            centred_units[index % len(centred_units)][shuffled_rng.permutation(self.a.size)]
            for index in range(n)
        ])
        for name, samples in (("random", random_samples), ("shuffled", shuffled_samples)):
            expected_rows = original_geometry(a_c, b_c, samples, mu=None)
            self.assertTrue(np.allclose(actual[name]["p"], [row["plane_projection"] for row in expected_rows]))
            self.assertTrue(np.allclose(actual[name]["emergence_ratio"], [row["emergence_ratio"] for row in expected_rows]))

    def test_null_geometry_rejects_invalid_size_and_empty_passages(self):
        with self.assertRaises(ValueError):
            null_geometry(self.a, self.b, self.passages, self.mu, n=0)
        with self.assertRaises(ValueError):
            null_geometry(self.a, self.b, np.empty((0, 5)), self.mu)


class RatioExplorationAndExclusionTest(unittest.TestCase):
    @staticmethod
    def hypothesis(concept):
        return {
            "concept": concept,
            "kind": "idea",
            "explanation": "x",
            "relation_to_A_alone": "x",
            "relation_to_B_alone": "x",
            "why_requires_AB": "x",
            "uncertainty": "x",
            "falsification_or_check": "x",
            "llm_self_assessment": None,
        }

    def test_auto_p_min_prevents_below_floor_ratio_from_driving_rank(self):
        low = np.array([0.3, 0.3, np.sqrt(0.82), 0.0])
        reliable = np.array([0.6, 0.6, np.sqrt(0.28), 0.0])
        encoder = FixedEncoder({"A": [1, 0, 0, 0], "B": [0, 1, 0, 0], "low": low, "firm": reliable})
        with mock.patch("third_concept.auto_p_min", return_value=0.8):
            data = explore("A", "B", ["low", "firm"], encoder, method="third-concept-ratio", center=False, null_samples=11)
        retrieved = [item for item in data["candidates"] if item["origin"] == "retrieved"]
        self.assertEqual(retrieved[0]["concept"], "firm")
        low_scores = next(item["scores"]["heuristic_ratio"] for item in retrieved if item["concept"] == "low")
        low_math = next(item["scores"]["heuristic_math"] for item in retrieved if item["concept"] == "low")
        self.assertTrue(low_scores["p_below_min"])
        self.assertGreater(low_scores["emergence_ratio_norm"], 0.0)
        self.assertAlmostEqual(low_scores["score_ratio"], 0.5 * low_math["balance"], delta=1e-6)
        self.assertEqual(data["params"]["p_min_source"], "auto:shuffled-p95")

    def test_sentence_input_needs_explicit_term_exclusion_across_all_origins(self):
        A = "犬が夕暮れに帰ってくる"
        B = "雨の記録"
        row = ("犬小屋", "安全", "犬を待つ灯り")
        passage = "犬小屋と安全。犬を待つ灯り"
        vectors = {
            A: [1, 0, 0, 0], B: [0, 1, 0, 0], "犬小屋": [0.4, 0.4, 0.8, 0],
            "安全": [0.2, 0.5, 0.8, 0], passage: [0.3, 0.3, 0.8, 0.3], "犬笛": [0.45, 0.35, 0.75, 0.2],
        }
        encoder = FixedEncoder(vectors)
        self.assertEqual(len(rank_entries(A, B, [row], encoder, "third-concept", 5, center=False)), 1)
        self.assertEqual(rank_entries(A, B, [row], encoder, "third-concept", 5, center=False, exclude=["犬"]), [])
        hypotheses = {"input": {"A": A, "B": B}, "generator": {"provider": "test"}, "hypotheses": [self.hypothesis("犬笛")]}
        unfiltered = explore(A, B, [row], encoder, limit=5, hypotheses=hypotheses, center=False, null_samples=7)
        self.assertTrue({"retrieved", "math", "idea"} <= {item["origin"] for item in unfiltered["candidates"]})
        filtered = explore(A, B, [row], encoder, limit=5, hypotheses=hypotheses, center=False, exclude=["犬"], null_samples=7)
        self.assertTrue(all("犬" not in item["concept"] for item in filtered["candidates"]))
        self.assertGreaterEqual(filtered["params"]["excluded_inputs"], 3)

    def test_ratio_method_orders_hypotheses_by_ratio_score(self):
        low = np.array([0.3, 0.3, np.sqrt(0.82)])
        high = np.array([0.6, 0.6, np.sqrt(0.28)])
        encoder = FixedEncoder({
            "left": [1, 0, 0], "right": [0, 1, 0], "source": [0.2, 0.2, np.sqrt(0.92)],
            "low": low, "firm": high,
        })
        hypotheses = {
            "input": {"A": "left", "B": "right"},
            "generator": {"provider": "test"},
            "hypotheses": [self.hypothesis("low"), self.hypothesis("firm")],
        }
        data = explore(
            "left", "right", ["source"], encoder, hypotheses=hypotheses, center=False,
            method="third-concept-ratio", p_min=0.8, null_samples=5,
        )
        self.assertEqual([item["concept"] for item in data["candidates"] if item["origin"] == "idea"], ["firm", "low"])

    def test_exclusion_normalisation_is_one_directional(self):
        encoder = FixedEncoder({
            "A": [1, 0, 0], "B": [0, 1, 0], "dog house": [0, 0, 1], "犬": [0.2, 0.3, 0.9],
        })
        dog_space = shared_geometry_space("A", "B", ["dog house"], encoder, center=False)
        self.assertEqual(exclusion_mask(dog_space, "A", "B", ["ＤＯＧ "]), [True])
        short_space = shared_geometry_space("A", "B", ["犬"], encoder, center=False)
        self.assertEqual(exclusion_mask(short_space, "A", "B", ["犬小屋"]), [False])

    def test_invalid_exclude_and_explicit_mask_contract(self):
        encoder = FixedEncoder({"A": [1, 0, 0], "B": [0, 1, 0], "A row": [0, 0, 1], "safe": [0.3, 0.4, 0.8]})
        corpus = ["A row", "safe"]
        space = shared_geometry_space("A", "B", corpus, encoder, center=False)
        for invalid in ("", "   ", 3):
            with self.subTest(term=invalid), self.assertRaisesRegex(ValueError, "exclude terms must be non-empty strings"):
                exclusion_mask(space, "A", "B", [invalid])
        ranked = rank_entries("A", "B", corpus, encoder, "third-concept", 5, center=False, space=space, excluded=[False, True])
        self.assertEqual([item["line"] for item in ranked], ["A row"])
        with self.assertRaises(ValueError):
            rank_entries("A", "B", corpus, encoder, "third-concept", 5, center=False, space=space, excluded=[False])

    def test_sentence_inputs_receive_one_query_prefix_in_explore(self):
        A = "犬が夕暮れに帰ってくる"
        B = "雨音を聞きながら眠る"
        model = RecordingModel()
        encoder = make_encoder("semantic", "intfloat/multilingual-e5-small", model=model)
        explore(A, B, ["synthetic passage"], encoder, limit=1, center=False, null_samples=5)
        sent = [text for call in model.calls for text in call]
        self.assertIn("query: " + A, sent)
        self.assertIn("query: " + B, sent)
        self.assertNotIn("query: query: " + A, sent)
        self.assertNotIn("query: query: " + B, sent)

    def test_candidate_ratio_scores_and_params_are_additions(self):
        encoder = FixedEncoder({"A": [1, 0, 0], "B": [0, 1, 0], "entry": [0.4, 0.4, np.sqrt(0.68)]})
        data = explore("A", "B", ["entry"], encoder, center=False, p_min=0.2, null_samples=3)
        scores = data["candidates"][0]["scores"]
        self.assertEqual(set(scores["heuristic_ratio"]), {"emergence_ratio", "emergence_ratio_norm", "score_ratio", "between_AB", "p_below_min"})
        self.assertEqual(data["params"]["method"], "third-concept")
        self.assertEqual(data["params"]["p_min"], 0.2)
        self.assertEqual(data["params"]["p_min_source"], "user")
        self.assertEqual(data["params"]["exclude"], [])
        self.assertEqual(data["params"]["null_samples"], 3)


class RatioCliTest(unittest.TestCase):
    def test_cli_ratio_options_are_recorded(self):
        command = [
            sys.executable, str(SCRIPTS / "third_concept_cli.py"), "--A", "雨", "--B", "時計",
            "--corpus", str(FIXTURES / "synthetic_corpus.json"), "--encoder", "hash",
            "--exclude", "犬", "--method", "third-concept-ratio", "--p-min", "0.25", "--null-samples", "9",
        ]
        data = json.loads(subprocess.run(command, cwd=ROOT, capture_output=True, check=True).stdout.decode("utf-8"))
        self.assertEqual(data["params"]["method"], "third-concept-ratio")
        self.assertEqual(data["params"]["p_min"], 0.25)
        self.assertEqual(data["params"]["exclude"], ["犬"])
        self.assertEqual(data["params"]["null_samples"], 9)

    def test_cli_bad_p_min_and_null_samples_exit_two(self):
        base = [sys.executable, str(SCRIPTS / "third_concept_cli.py"), "--A", "雨", "--B", "時計"]
        for args in (["--p-min", "1.5"], ["--p-min", "nan"], ["--null-samples", "0"]):
            with self.subTest(args=args):
                completed = subprocess.run(base + args, cwd=ROOT, capture_output=True)
                self.assertEqual(completed.returncode, 2)


if __name__ == "__main__":
    unittest.main()
