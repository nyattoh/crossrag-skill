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

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
FIXTURES = Path(__file__).resolve().parent / "fixtures"
sys.path.insert(0, str(SCRIPTS))

from engine import DEFAULT_CORPUS, HashEncoder, run  # noqa: E402
from third_concept import (  # noqa: E402
    build_hypothesis_prompt,
    explore,
    geometry,
    load_hypotheses,
    make_encoder,
    rank_entries,
)


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

    def test_geometry_warns_for_degenerate_plane(self):
        rows = geometry([1, 0, 0], [1, 1e-10, 0], [[0, 1, 0]])
        self.assertEqual(rows[0]["emergence_norm"], 0.0)
        self.assertTrue(rows[0]["warnings"])


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

    def test_json_schema(self):
        data = explore("湯気", "睡眠", self.corpus, self.encoder, limit=5, corpus_path=self.corpus_path)
        self.assertTrue({"mode", "input", "encoder", "corpus_id", "corpus_size", "params", "origin_legend", "candidates", "warnings", "disclaimer"} <= data.keys())
        self.assertRegex(data["corpus_id"], r"^[0-9a-f]{16}$")
        self.assertTrue(data["disclaimer"])
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


if __name__ == "__main__":
    unittest.main()
