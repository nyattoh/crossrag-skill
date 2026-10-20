"""Offline CLI contracts for residual-cluster and legacy PCA rays."""
from __future__ import annotations

import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
FIXTURE = ROOT / "tests" / "fixtures" / "synthetic_corpus.json"
sys.path.insert(0, str(SCRIPTS))

import third_concept_cli as cli  # noqa: E402


class ResidualRayCliTest(unittest.TestCase):
    def run_cli(self, *arguments, with_c=True, ray=True):
        command = [
            sys.executable, str(SCRIPTS / "third_concept_cli.py"),
            "--A", "概念アルファ", "--B", "概念ベータ",
            "--corpus", str(FIXTURE), "--encoder", "hash",
        ]
        if ray:
            command.extend(("--search-mode", "ray"))
        if with_c:
            command.extend(("--C", "出典記事の本文"))
        return subprocess.run([*command, *arguments], cwd=ROOT, capture_output=True, check=False)

    def successful_json(self, *arguments, **kwargs):
        completed = self.run_cli(*arguments, **kwargs)
        self.assertEqual(completed.returncode, 0, completed.stderr.decode("utf-8", errors="replace"))
        return json.loads(completed.stdout.decode("utf-8"))

    def assert_rejected(self, *arguments, message, **kwargs):
        completed = self.run_cli(*arguments, **kwargs)
        self.assertEqual(completed.returncode, 2, completed.stderr.decode("utf-8", errors="replace"))
        self.assertIn(message, completed.stderr.decode("utf-8", errors="replace"))
        self.assertEqual(completed.stdout, b"")

    def test_default_residual_cluster_returns_utf8_json_and_shared_passage_coordinates(self):
        result = self.successful_json()

        self.assertEqual(result["mode"], "normal-ray")
        self.assertEqual(result["method"], "affine_residual_cluster")
        self.assertEqual(result["status"], "no_residual_clusters")
        self.assertEqual(result["corpus_size"], 10)
        self.assertEqual(result["clustering"]["candidate_count"], result["corpus_size"])
        self.assertEqual(result["clustering"]["cluster_count"], 10)
        self.assertEqual(result["input"]["C"], "出典記事の本文")
        self.assertEqual(result["params"]["encoding_path"], "encode_passage")
        self.assertEqual(result["params"]["origin"], "C")
        self.assertEqual(result["params"]["cluster_cosine"], 0.7)
        self.assertEqual(result["params"]["min_cluster_size"], 2)
        self.assertNotIn("concept_sets", result)
        self.assertNotIn("passage_sets", result)
        for warning in result["warnings"]:
            self.assertNotRegex(warning.lower(), r"query|passage|align|mismatch")

    def test_default_residual_cluster_requires_c(self):
        self.assert_rejected(with_c=False, message="residual-cluster ray search requires --C")

    def test_residual_cluster_rejects_legacy_only_flags_even_at_legacy_defaults(self):
        for arguments in (
            ("--no-center",), ("--vocab", "unused.json"), ("--vocab=",),
            ("--ray-cluster-gap", "0.25"), ("--ray-min-set-size", "2"),
        ):
            with self.subTest(arguments=arguments):
                self.assert_rejected(*arguments, message=f"residual-cluster ray search does not support {arguments[0].split('=')[0]}")

    def test_both_ray_methods_reject_ranking_and_prompt_flags(self):
        for method in ("residual-cluster", "pca-cross"):
            for arguments in (
                ("--emit-prompt",), ("--hypotheses", "unused.json"), ("--with-math",),
                ("--method", "third-concept-ratio"), ("--seed", "1"),
                ("--null-samples", "2"), ("--p-min", "0.1"),
            ):
                with self.subTest(method=method, arguments=arguments):
                    self.assert_rejected(
                        "--ray-method", method, *arguments, with_c=method == "residual-cluster",
                        message="ray search does not support",
                    )

    def test_explicit_zero_seed_matches_omitted_seed_for_both_ray_methods(self):
        for method in ("residual-cluster", "pca-cross"):
            with self.subTest(method=method):
                with_c = method == "residual-cluster"
                default_seed = self.successful_json("--ray-method", method, with_c=with_c)
                explicit_zero_seed = self.successful_json(
                    "--ray-method", method, "--seed", "0", with_c=with_c,
                )
                self.assertEqual(explicit_zero_seed, default_seed)

    def test_invalid_residual_parameters_route_to_parser_error(self):
        for arguments, message in (
            (("--ray-radius", "0"), "ray radius must be finite and positive"),
            (("--ray-cluster-cosine", "0"), "cluster cosine must be finite and in (0, 1]"),
            (("--ray-cluster-cosine", "1.5"), "cluster cosine must be finite and in (0, 1]"),
            (("--ray-min-cluster-size", "0"), "ray minimum cluster size must be at least 1"),
        ):
            with self.subTest(arguments=arguments):
                self.assert_rejected(*arguments, message=message)

    def test_c_is_rejected_outside_ray_mode(self):
        self.assert_rejected(ray=False, message="--C is only supported with --search-mode ray")

    def test_legacy_pca_cross_preserves_schema_and_defaults(self):
        result = self.successful_json("--ray-method", "pca-cross", with_c=False)

        self.assertEqual(result["mode"], "normal-ray")
        self.assertIn("concept_sets", result)
        self.assertIn("passage_sets", result)
        self.assertEqual(result["params"]["cluster_gap"], 0.25)
        self.assertEqual(result["params"]["min_set_size"], 2)
        self.assertNotIn("C", result["input"])

    def test_legacy_pca_cross_rejects_residual_only_flags(self):
        for arguments in (
            ("--C", "source"), ("--C=",),
            ("--ray-cluster-cosine", "0.7"), ("--ray-min-cluster-size", "2"),
        ):
            with self.subTest(arguments=arguments):
                self.assert_rejected(
                    "--ray-method", "pca-cross", *arguments, with_c=False,
                    message=f"pca-cross ray search does not support {arguments[0].split('=')[0]}",
                )

    def test_only_explicit_exclusions_remove_plain_rows_matching_inputs(self):
        rows = ["概念アルファ", "出典記事の本文", "exclude-only この行だけ除く", "無関係な候補文"]
        with tempfile.TemporaryDirectory() as directory:
            corpus_path = Path(directory) / "plain.json"
            corpus_path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
            unfiltered = self.successful_json("--corpus", str(corpus_path))
            excluded = self.successful_json("--corpus", str(corpus_path), "--exclude", "exclude-only")

        self.assertEqual(unfiltered["corpus_size"], len(rows))
        self.assertEqual(unfiltered["clustering"]["excluded_count"], 0)
        self.assertEqual(unfiltered["clustering"]["candidate_count"], len(rows))
        self.assertEqual(excluded["clustering"]["excluded_count"], 1)
        self.assertEqual(excluded["clustering"]["candidate_count"], len(rows) - 1)

    def test_limit_and_residual_cluster_controls_are_passed_through(self):
        result = self.successful_json(
            "--limit", "3", "--ray-cluster-cosine", "1", "--ray-min-cluster-size", "1",
        )

        self.assertEqual(result["params"]["max_rays"], 3)
        self.assertEqual(result["params"]["cluster_cosine"], 1)
        self.assertEqual(result["params"]["min_cluster_size"], 1)

    def test_main_returns_hits_using_only_encode_passage_for_every_vector(self):
        rows = ["first", "second"]
        vectors = {"A": [1, 0, 0], "B": [0, 1, 0], "C": [0, 0, 0],
                   "first": [0, 0, 1], "second": [0, 0, 2]}
        calls = []

        class FixedEncoder:
            name = "fixed-test"
            model_id = "fixed-test"
            prefix = "passage: "

            def encode_query(self, texts):
                raise AssertionError("residual-cluster must never call encode_query")

            def encode_passage(self, texts):
                calls.append(list(texts))
                return np.asarray([vectors[text] for text in texts], dtype=float)

        argv = ["third_concept_cli.py", "--A", "A", "--B", "B", "--C", "C", "--search-mode", "ray"]
        with io.TextIOWrapper(io.BytesIO(), encoding="utf-8") as output:
            with patch.object(sys, "argv", argv), patch.object(cli, "DEFAULT_CORPUS", rows), \
                    patch.object(cli, "make_encoder", return_value=FixedEncoder()), redirect_stdout(output):
                cli.main()
            output.flush()
            result = json.loads(output.buffer.getvalue().decode("utf-8"))

        self.assertEqual(calls, [["A", "B", "C", "first", "second"]])
        self.assertEqual(result["params"]["encoding_path"], "encode_passage")
        self.assertEqual(result["status"], "hits_found")
        self.assertEqual([hit["line"] for hit in result["rays"][0]["hits"]], rows)
        self.assertEqual(result["warnings"], [])


if __name__ == "__main__":
    unittest.main()
