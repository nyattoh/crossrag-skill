"""Offline CLI contracts for random-orthogonal serendipity retrieval."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
LAW_DIRECTIVE = "この世界が成立する奇妙な法則を一つ作れ"


class SerendipityCliTest(unittest.TestCase):
    def run_cli(self, *arguments):
        command = [sys.executable, str(SCRIPTS / "third_concept_cli.py"),
                   "--search-mode", "random-orthogonal", "--encoder", "hash", "--seed", "23"]
        return subprocess.run([*command, *arguments], cwd=ROOT, capture_output=True, check=False)

    def run_legacy(self, *arguments):
        command = [sys.executable, str(SCRIPTS / "third_concept_cli.py"),
                   "--A", "concept A", "--B", "concept B", "--encoder", "hash"]
        return subprocess.run([*command, *arguments], cwd=ROOT, capture_output=True, check=False)

    def test_repeated_contexts_select_isolated_mode_and_return_combined_json(self):
        with tempfile.TemporaryDirectory() as directory:
            corpus = Path(directory) / "corpus.json"
            corpus.write_text(json.dumps(["candidate one", "candidate two"], ensure_ascii=False), encoding="utf-8")
            completed = self.run_cli("--context", "context one", "--context", "context two",
                                     "--context", "context three", "--corpus", str(corpus))

        self.assertEqual(completed.returncode, 0, completed.stderr.decode("utf-8", errors="replace"))
        result = json.loads(completed.stdout.decode("utf-8"))
        self.assertEqual(result["mode"], "random-orthogonal")
        self.assertEqual(result["input"]["contexts"], ["context one", "context two", "context three"])
        self.assertEqual(result["params"]["seed"], 23)
        self.assertTrue(result["rays"])
        self.assertIn("combined_clusters", result)
        self.assertNotIn("concept_sets", result)
        self.assertNotIn("passage_sets", result)

    def test_requires_at_least_two_context_arguments(self):
        for contexts in ((), ("--context", "only one")):
            with self.subTest(contexts=contexts):
                completed = self.run_cli(*contexts)
                self.assertEqual(completed.returncode, 2)
                self.assertIn("at least two --context values", completed.stderr.decode("utf-8", errors="replace"))
                self.assertEqual(completed.stdout, b"")

    def test_awaiting_schema_has_exact_host_directive_and_contextual_prompt(self):
        with tempfile.TemporaryDirectory() as directory:
            corpus = Path(directory) / "corpus.json"
            corpus.write_text("[]", encoding="utf-8")
            completed = self.run_cli("--context", "one", "--context", "two", "--corpus", str(corpus))
        self.assertEqual(completed.returncode, 0, completed.stderr.decode("utf-8", errors="replace"))
        result = json.loads(completed.stdout.decode("utf-8"))
        self.assertEqual(result["search_status"], "unavailable")
        self.assertEqual(result["workflow_status"], "awaiting_law")
        self.assertFalse(result["finalized"])
        self.assertTrue(result["awaiting_host_law"])
        self.assertIsNone(result["law"])
        self.assertNotIn("law", result["input"])
        host_prompt = result["host_prompt"]
        self.assertEqual(host_prompt["required_directive"], LAW_DIRECTIVE)
        self.assertIsInstance(host_prompt["prompt"], str)
        self.assertTrue(host_prompt["prompt"].strip())
        self.assertEqual(host_prompt["contexts"], ["one", "two"])
        self.assertEqual(host_prompt["selected_clusters"], result["combined_clusters"])
        self.assertIsInstance(host_prompt["tone"], str)
        self.assertTrue(host_prompt["tone"].strip())
        self.assertIsInstance(host_prompt["tone_instruction"], str)
        self.assertTrue(host_prompt["tone_instruction"].strip())
        self.assertIsInstance(host_prompt["guardrails"], list)
        self.assertTrue(host_prompt["guardrails"])
        self.assertEqual(result["params"]["tone"], host_prompt["tone"])
        self.assertEqual(result["params"]["tone_instruction"], host_prompt["tone_instruction"])
        self.assertIn("one", host_prompt["prompt"])
        self.assertIn("two", host_prompt["prompt"])
        self.assertIn("selected_clusters", host_prompt["prompt"])
        self.assertIn("tone", host_prompt["prompt"])
        self.assertIn("tone_instruction", host_prompt["prompt"])
        self.assertIn("guardrails", host_prompt["prompt"])

    def test_nonempty_law_finalises_as_host_llm_without_api_call(self):
        law = "雨が降るたび、昨日の道が一つだけ移動する。"
        with tempfile.TemporaryDirectory() as directory:
            corpus = Path(directory) / "corpus.json"
            corpus.write_text(json.dumps(["candidate"], ensure_ascii=False), encoding="utf-8")
            completed = self.run_cli("--context", "one", "--context", "two", "--law", law, "--corpus", str(corpus))
        self.assertEqual(completed.returncode, 0, completed.stderr.decode("utf-8", errors="replace"))
        result = json.loads(completed.stdout.decode("utf-8"))
        self.assertTrue(result["finalized"])
        self.assertFalse(result["awaiting_host_law"])
        self.assertEqual(result["law"], law)
        self.assertEqual(result["provenance"]["law_source"], "host_llm")

    def test_serendipity_flag_accepts_inclusive_integer_range_and_rejects_invalid_values(self):
        for value in ("0", "60", "100"):
            with self.subTest(value=value):
                completed = self.run_cli("--context", "one", "--context", "two", "--serendipity", value)
                self.assertEqual(completed.returncode, 0, completed.stderr.decode("utf-8", errors="replace"))
                self.assertEqual(json.loads(completed.stdout.decode("utf-8"))["params"]["serendipity"], int(value))
        for value in ("-1", "101", "1.5"):
            with self.subTest(value=value):
                completed = self.run_cli("--context", "one", "--context", "two", "--serendipity", value)
                self.assertEqual(completed.returncode, 2)

    def test_random_orthogonal_accepts_explicit_ray_cluster_cosine_override(self):
        completed = self.run_cli("--context", "one", "--context", "two", "--ray-cluster-cosine", "0.9")
        self.assertEqual(completed.returncode, 0, completed.stderr.decode("utf-8", errors="replace"))
        params = json.loads(completed.stdout.decode("utf-8"))["params"]
        self.assertEqual(params["cosine_threshold"], 0.9)
        self.assertEqual(params["mapped_cosine_threshold"], 0.64)
        self.assertEqual(params["cosine_threshold_source"], "explicit_override")

    def test_random_orthogonal_rejects_legacy_concept_flags(self):
        for flag in ("--A", "--B", "--C"):
            with self.subTest(flag=flag):
                completed = self.run_cli("--context", "one", "--context", "two", flag, "legacy")
                self.assertEqual(completed.returncode, 2)

    def test_legacy_modes_reject_new_serendipity_flags_and_seed(self):
        for method in ("residual-cluster", "pca-cross"):
            for flag_args in (("--seed", "4"), ("--ray-count", "2"), ("--serendipity", "60"), ("--law", "law")):
                with self.subTest(method=method, flag_args=flag_args):
                    command = ["--search-mode", "ray", "--ray-method", method]
                    if method == "residual-cluster":
                        command.extend(("--C", "source text"))
                    completed = self.run_legacy(*command, *flag_args)
                    self.assertEqual(completed.returncode, 2)

    def test_third_concept_mode_rejects_serendipity_options(self):
        for flag_args in (("--ray-count", "2"), ("--serendipity", "60"), ("--law", "law")):
            with self.subTest(flag_args=flag_args):
                completed = self.run_legacy("--search-mode", "third-concept", *flag_args)
                self.assertEqual(completed.returncode, 2)

    def test_legacy_third_concept_accepts_seed_for_random_sampling(self):
        completed = self.run_legacy("--search-mode", "third-concept", "--method", "third-concept-ratio", "--seed", "4")
        self.assertEqual(completed.returncode, 0, completed.stderr.decode("utf-8", errors="replace"))
        self.assertTrue(json.loads(completed.stdout.decode("utf-8"))["candidates"])

    def test_old_ray_methods_reject_random_orthogonal_combination(self):
        for method in ("residual-cluster", "pca-cross"):
            with self.subTest(method=method):
                completed = self.run_legacy("--search-mode", "ray", "--ray-method", method, "--seed", "23")
                self.assertEqual(completed.returncode, 2)


if __name__ == "__main__":
    unittest.main()
