"""User-facing regression tests for the isolated LLM knowledge pool draw."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"


class LlmPoolCliTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.pool_path = Path(self.temp.name) / "pool.json"
        self.pool = [f"concept-{index}" for index in range(40)]
        self.write_pool(self.pool)

    def write_pool(self, pool):
        self.pool_path.write_text(json.dumps(pool), encoding="utf-8")

    def run_cli(self, script, *arguments):
        return subprocess.run(
            [sys.executable, str(SCRIPTS / script), *arguments],
            cwd=ROOT,
            capture_output=True,
            check=False,
        )

    def success_json(self, script, *arguments):
        completed = self.run_cli(script, *arguments)
        self.assertEqual(completed.returncode, 0, completed.stderr.decode("utf-8", errors="replace"))
        return json.loads(completed.stdout.decode("utf-8"))

    def test_each_retrieval_cli_rejects_missing_material(self):
        commands = (
            ("run.py", "--context", "rain, clock"),
            ("third_concept_cli.py", "--A", "rain", "--B", "clock"),
            ("compare_third_concept.py", "--A", "rain", "--B", "clock"),
        )
        for script, *arguments in commands:
            with self.subTest(script=script):
                completed = self.run_cli(script, *arguments)
                self.assertEqual(completed.returncode, 2)
                self.assertIn("corpus", completed.stderr.decode("utf-8", errors="replace").lower())

    def test_seeded_draw_is_context_independent_and_has_its_own_schema(self):
        selected = []
        for context in ("rain, clock", "algebra, fermentation, migration"):
            result = self.success_json(
                "run.py", "--llm-pool", str(self.pool_path), "--context", context,
                "--limit", "6", "--seed", "41", "--llm-pool-model", "  host-model-x  ",
            )
            self.assertEqual(result["corpus_source"], "llm_knowledge")
            self.assertEqual(result["ranking_metric"], "seeded_uniform_draw")
            self.assertEqual(result["seed"], 41)
            self.assertEqual(result["encoder"], "none")
            self.assertEqual(result["knowledge_basis"], "host_llm_general_knowledge")
            self.assertEqual(result["generation_model"], "host-model-x")
            self.assertEqual([row["rank_value"] for row in result["selected"]], [1, 2, 3, 4, 5, 6])
            self.assertEqual([row["index"] for row in result["selected"]], [5, 7, 27, 9, 37, 4])
            self.assertEqual(
                [row["concept"] for row in result["selected"]],
                ["concept-5", "concept-7", "concept-27", "concept-9", "concept-37", "concept-4"],
            )
            selected.append(result["selected"])
        self.assertEqual(selected[0], selected[1])

    def test_empty_context_is_ignored_by_llm_pool_mode(self):
        for script in ("run.py", "third_concept_cli.py"):
            with self.subTest(script=script):
                result = self.success_json(
                    script, "--context", "", "--llm-pool", str(self.pool_path), "--seed", "17",
                )
                self.assertEqual(result["seed"], 17)

    def test_values_starting_with_equals_do_not_trigger_option_guards(self):
        corpus = ROOT / "tests" / "fixtures" / "synthetic_corpus.json"
        completed = self.run_cli(
            "third_concept_cli.py", "--A", "rain", "--B", "clock", "--corpus", str(corpus),
            "--exclude", "=x", "--limit", "1",
        )
        self.assertEqual(completed.returncode, 0, completed.stderr.decode("utf-8", errors="replace"))

    def test_compare_markdown_escapes_pool_concepts_for_table_cells(self):
        self.write_pool([
            "A|B testing", "line\nbreak", "carriage\rreturn", "crlf\r\nbreak",
            *[f"field-{index}" for index in range(11)],
        ])
        completed = self.run_cli(
            "compare_third_concept.py", "--llm-pool", str(self.pool_path), "--limit", "15",
            "--seed", "1", "--format", "markdown",
        )
        self.assertEqual(completed.returncode, 0, completed.stderr.decode("utf-8", errors="replace"))
        markdown = completed.stdout.decode("utf-8")
        self.assertIn("A\\|B testing", markdown)
        self.assertIn("line break", markdown)
        self.assertIn("carriage return", markdown)
        self.assertIn("crlf break", markdown)

    def test_pool_rejects_unpaired_surrogates_with_exit_two(self):
        self.pool_path.write_text(
            r'["\ud800x",' + ",".join(json.dumps(value) for value in self.pool[1:]) + "]",
            encoding="utf-8",
        )
        completed = self.run_cli("run.py", "--llm-pool", str(self.pool_path), "--limit", "40", "--seed", "1")
        self.assertEqual(completed.returncode, 2)
        self.assertIn("valid UTF-8", completed.stderr.decode("utf-8", errors="replace"))

    def test_empty_conversation_reports_conversation_path_error(self):
        completed = self.run_cli(
            "run.py", "--context", "rain,clock", "--conversation", "",
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("conversation path is required", completed.stderr.decode("utf-8", errors="replace"))

    def test_seed_omission_uses_a_fresh_seed_for_each_run(self):
        commands = (
            ("run.py", ()),
            ("third_concept_cli.py", ()),
            ("compare_third_concept.py", ()),
        )
        for script, args in commands:
            with self.subTest(script=script):
                results = [
                    self.success_json(script, *args, "--llm-pool", str(self.pool_path), "--limit", "10")
                    for _ in range(2)
                ]
                self.assertIsInstance(results[0]["seed"], int)
                self.assertIsInstance(results[1]["seed"], int)
                self.assertNotEqual(results[0]["seed"], results[1]["seed"])
                self.assertLess(results[0]["seed"], 2**53)
                self.assertLess(results[1]["seed"], 2**53)
                replay = self.success_json(
                    script, *args, "--llm-pool", str(self.pool_path), "--limit", "10",
                    "--seed", str(results[0]["seed"]),
                )
                self.assertEqual(replay["selected"], results[0]["selected"])

    def test_llm_pool_does_not_require_context_or_load_an_encoder(self):
        result = self.success_json("run.py", "--llm-pool", str(self.pool_path), "--seed", "8")
        self.assertEqual(result["encoder"], "none")
        self.assertEqual(len(result["selected"]), 5)

    def test_llm_pool_rejects_geometric_encoder_options(self):
        completed = self.run_cli(
            "run.py", "--llm-pool", str(self.pool_path), "--encoder", "semantic",
            "--model", "a-model-that-must-not-be-loaded",
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("cannot be combined with geometric or encoder options", completed.stderr.decode("utf-8", errors="replace"))

    def test_pool_validation_rejects_bad_schema_empty_duplicates_and_too_few_unique_items(self):
        invalid_pools = (
            ({"concepts": self.pool}, "JSON array"),
            ([*self.pool[:-1], "  "], "non-empty string"),
            ([*self.pool[:-1], 17], "non-empty string"),
            ([*self.pool[:9], self.pool[0]], "at least 10 unique concepts"),
        )
        for invalid, message in invalid_pools:
            with self.subTest(invalid=invalid[-1] if isinstance(invalid, list) else invalid):
                self.write_pool(invalid)
                completed = self.run_cli("run.py", "--llm-pool", str(self.pool_path))
                self.assertEqual(completed.returncode, 2)
                self.assertIn(message, completed.stderr.decode("utf-8", errors="replace"))

    def test_pool_draw_is_supported_by_third_concept_and_comparison_clis(self):
        for script, args in (
            ("third_concept_cli.py", ("--context", "one", "--context", "two")),
            ("compare_third_concept.py", ()),
        ):
            with self.subTest(script=script):
                result = self.success_json(
                    script, *args, "--llm-pool", str(self.pool_path), "--seed", "12", "--limit", "4",
                )
                self.assertEqual(result["corpus_source"], "llm_knowledge")
                self.assertEqual(result["ranking_metric"], "seeded_uniform_draw")
                self.assertEqual(result["seed"], 12)
                self.assertEqual(len(result["selected"]), 4)
                self.assertEqual([row["index"] for row in result["selected"]], [9, 32, 11, 5])

    def test_pool_file_accepts_utf8_bom(self):
        self.pool_path.write_bytes(b"\xef\xbb\xbf" + json.dumps(self.pool).encode("utf-8"))
        result = self.success_json("run.py", "--llm-pool", str(self.pool_path), "--seed", "3")
        self.assertEqual(result["total"], 40)
        self.assertEqual(len(result["selected"]), 5)

    def test_third_concept_context_does_not_change_the_pool_draw(self):
        first = self.success_json(
            "third_concept_cli.py", "--context", "rain", "--context", "clock",
            "--llm-pool", str(self.pool_path), "--seed", "19",
        )
        second = self.success_json(
            "third_concept_cli.py", "--context", "algebra", "--context", "fermentation",
            "--llm-pool", str(self.pool_path), "--seed", "19",
        )
        self.assertEqual(first["selected"], second["selected"])

    def test_demo_corpus_is_opt_in_and_labelled_in_each_geometry_cli(self):
        commands = (
            ("run.py", ("--context", "rain,clock")),
            ("third_concept_cli.py", ("--A", "rain", "--B", "clock")),
            ("compare_third_concept.py", ("--A", "rain", "--B", "clock", "--null-samples", "2")),
        )
        for script, args in commands:
            with self.subTest(script=script):
                result = self.success_json(script, *args, "--demo-corpus", "--limit", "1")
                self.assertEqual(result["corpus_source"], "demo")

    def test_llm_pool_cannot_silently_replace_an_explicit_corpus(self):
        for script, args in (
            ("run.py", ("--context", "rain,clock")),
            ("third_concept_cli.py", ("--A", "rain", "--B", "clock")),
            ("compare_third_concept.py", ("--A", "rain", "--B", "clock")),
        ):
            with self.subTest(script=script):
                completed = self.run_cli(
                    script, *args, "--corpus", str(self.pool_path), "--llm-pool", str(self.pool_path),
                )
                self.assertEqual(completed.returncode, 2)
                error = completed.stderr.decode("utf-8", errors="replace").lower()
                self.assertTrue("specify only one" in error or "cannot be combined" in error, error)

    def test_duplicate_concepts_are_removed_before_draw(self):
        self.write_pool([*self.pool, "  CONCEPT-0  ", "concept-1", "ＣＯＮＣＥＰＴ-2"])
        result = self.success_json(
            "run.py", "--llm-pool", str(self.pool_path), "--seed", "4", "--limit", "10",
        )
        self.assertEqual(result["total"], 40)
        self.assertEqual(len({row["concept"].casefold() for row in result["selected"]}), 10)

    def test_requested_limit_cannot_exceed_unique_pool(self):
        completed = self.run_cli(
            "run.py", "--llm-pool", str(self.pool_path), "--limit", "41", "--seed", "3",
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("cannot exceed", completed.stderr.decode("utf-8", errors="replace"))

    def test_compare_markdown_preserves_llm_pool_provenance_and_seed(self):
        result = self.run_cli(
            "compare_third_concept.py", "--llm-pool", str(self.pool_path), "--limit", "3",
            "--seed", "23", "--llm-pool-model", "host-model-y", "--format", "markdown",
        )
        self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8", errors="replace"))
        markdown = result.stdout.decode("utf-8")
        for expected in (
            "llm_knowledge", "host_llm_general_knowledge", "seeded_uniform_draw",
            "encoder: `none`", "seed: `23`", "generation_model: `host-model-y`",
            "| rank_value | index | concept | ranking_metric |",
        ):
            self.assertIn(expected, markdown)
        without_model = self.run_cli(
            "compare_third_concept.py", "--llm-pool", str(self.pool_path), "--limit", "1",
            "--seed", "23", "--format", "markdown",
        )
        self.assertEqual(without_model.returncode, 0, without_model.stderr.decode("utf-8", errors="replace"))
        self.assertIn("generation_model: `null`", without_model.stdout.decode("utf-8"))

    def test_demo_source_is_visible_in_markdown_and_generated_prompt_outputs(self):
        markdown = self.run_cli(
            "compare_third_concept.py", "--A", "rain", "--B", "clock", "--demo-corpus",
            "--null-samples", "2", "--limit", "1", "--format", "markdown",
        )
        self.assertEqual(markdown.returncode, 0, markdown.stderr.decode("utf-8", errors="replace"))
        self.assertIn("Corpus source: `demo`", markdown.stdout.decode("utf-8"))

        prompt = self.run_cli(
            "third_concept_cli.py", "--A", "rain", "--B", "clock", "--demo-corpus",
            "--emit-prompt", "--with-math", "--limit", "1",
        )
        self.assertEqual(prompt.returncode, 0, prompt.stderr.decode("utf-8", errors="replace"))
        self.assertIn("corpus_source: demo", prompt.stdout.decode("utf-8"))

        explicit_prompt = self.run_cli(
            "third_concept_cli.py", "--A", "rain", "--B", "clock", "--corpus", str(self.pool_path),
            "--emit-prompt", "--with-math", "--limit", "1",
        )
        self.assertEqual(explicit_prompt.returncode, 0, explicit_prompt.stderr.decode("utf-8", errors="replace"))
        self.assertNotIn("corpus_source:", explicit_prompt.stdout.decode("utf-8"))

    def test_blind_comparison_key_records_the_demo_source(self):
        sheet_dir = Path(self.temp.name) / "sheets"
        completed = self.run_cli(
            "compare_third_concept.py", "--A", "rain", "--B", "clock", "--demo-corpus",
            "--null-samples", "2", "--limit", "1", "--sheet-dir", str(sheet_dir),
        )
        self.assertEqual(completed.returncode, 0, completed.stderr.decode("utf-8", errors="replace"))
        key = json.loads((sheet_dir / "blind_key.json").read_text(encoding="utf-8"))
        self.assertEqual(key["corpus_source"], "demo")
        sheet = (sheet_dir / "blind_sheet.md").read_text(encoding="utf-8")
        self.assertIn("Corpus source: `demo`", sheet)

    def test_llm_pool_rejects_abbreviated_seed_option(self):
        for script, arguments in (
            ("run.py", ()),
            ("third_concept_cli.py", ()),
            ("compare_third_concept.py", ()),
        ):
            with self.subTest(script=script):
                completed = self.run_cli(
                    script, *arguments, "--llm-pool", str(self.pool_path), "--see", "12",
                )
                self.assertEqual(completed.returncode, 2)
                self.assertIn("unrecognized arguments", completed.stderr.decode("utf-8", errors="replace"))

    def test_llm_pool_accepts_complete_options_with_equals_values(self):
        for script in ("run.py", "third_concept_cli.py", "compare_third_concept.py"):
            with self.subTest(script=script):
                result = self.success_json(
                    script,
                    f"--llm-pool={self.pool_path}",
                    "--limit=4",
                    "--seed=12",
                )
                self.assertEqual(result["seed"], 12)
                self.assertEqual(len(result["selected"]), 4)

    def test_llm_pool_guards_abbreviated_geometric_options(self):
        commands = (
            ("run.py", ("--enc", "semantic")),
            ("third_concept_cli.py", ("--search", "random-orthogonal")),
            ("compare_third_concept.py", ("--enc", "semantic")),
        )
        for script, options in commands:
            with self.subTest(script=script):
                completed = self.run_cli(script, "--llm-pool", str(self.pool_path), *options)
                self.assertEqual(completed.returncode, 2)
                self.assertIn("unrecognized arguments", completed.stderr.decode("utf-8", errors="replace"))

    def test_geometry_clis_reject_option_abbreviations(self):
        corpus = ROOT / "tests" / "fixtures" / "synthetic_corpus.json"
        commands = (
            ("run.py", ("--cont", "rain,clock", "--corpus", str(corpus), "--enc", "hash", "--lim", "1")),
            ("third_concept_cli.py", ("--A", "rain", "--B", "clock", "--corpus", str(corpus), "--enc", "hash", "--lim", "1")),
            ("compare_third_concept.py", ("--A", "rain", "--B", "clock", "--corpus", str(corpus), "--enc", "hash", "--lim", "1", "--null-samples", "2")),
        )
        for script, options in commands:
            with self.subTest(script=script):
                completed = self.run_cli(script, *options)
                self.assertEqual(completed.returncode, 2)
                self.assertTrue(
                    "unrecognized arguments" in completed.stderr.decode("utf-8", errors="replace")
                    or "ambiguous option" in completed.stderr.decode("utf-8", errors="replace")
                )

    def test_minimum_pool_and_matching_limit_boundaries_are_valid(self):
        minimum = [f"concept-{index}" for index in range(10)]
        self.write_pool(minimum)
        result = self.success_json(
            "run.py", "--llm-pool", str(self.pool_path), "--limit", "10", "--seed", "1",
        )
        self.assertEqual(result["total"], 10)
        self.assertEqual(len(result["selected"]), 10)

    def test_empty_source_values_do_not_bypass_source_exclusivity(self):
        commands = (
            ("run.py", ("--corpus", "", "--llm-pool", str(self.pool_path))),
            ("third_concept_cli.py", ("--A", "rain", "--B", "clock", "--demo-corpus", "--llm-pool", "")),
            ("compare_third_concept.py", ("--A", "rain", "--B", "clock", "--corpus", str(self.pool_path), "--llm-pool", "")),
        )
        for script, options in commands:
            with self.subTest(script=script):
                completed = self.run_cli(script, *options)
                self.assertEqual(completed.returncode, 2)
                error = completed.stderr.decode("utf-8", errors="replace")
                self.assertTrue("specify only one" in error or "cannot be combined" in error, error)

    def test_internal_whitespace_duplicates_are_removed(self):
        concepts = [f"field {index}" for index in range(10)]
        self.write_pool([*concepts, "field   1"])
        result = self.success_json(
            "run.py", "--llm-pool", str(self.pool_path), "--limit", "10", "--seed", "2",
        )
        self.assertEqual(result["total"], 10)

    def test_llm_pool_metadata_cannot_be_ignored_by_geometry_modes(self):
        commands = (
            ("run.py", ("--context", "rain,clock", "--demo-corpus", "--seed", "5")),
            ("third_concept_cli.py", ("--A", "rain", "--B", "clock", "--demo-corpus")),
            ("compare_third_concept.py", ("--A", "rain", "--B", "clock", "--demo-corpus")),
        )
        for script, arguments in commands:
            with self.subTest(script=script):
                completed = self.run_cli(
                    script, *arguments, "--llm-pool-model", "host-model-z",
                )
                self.assertEqual(completed.returncode, 2)
                self.assertIn("requires --llm-pool", completed.stderr.decode("utf-8", errors="replace"))

    def test_third_concept_pool_rejects_A_and_B_inputs(self):
        completed = self.run_cli(
            "third_concept_cli.py", "--llm-pool", str(self.pool_path), "--A", "rain", "--B", "clock",
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("context-free draw", completed.stderr.decode("utf-8", errors="replace"))

    def test_llm_pool_model_must_be_non_empty(self):
        completed = self.run_cli(
            "run.py", "--llm-pool", str(self.pool_path), "--llm-pool-model", " ",
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("must be non-empty", completed.stderr.decode("utf-8", errors="replace"))

    def test_corpus_and_demo_corpus_cannot_be_combined_in_other_clis(self):
        for script, args in (
            ("third_concept_cli.py", ("--A", "rain", "--B", "clock")),
            ("compare_third_concept.py", ("--A", "rain", "--B", "clock")),
        ):
            with self.subTest(script=script):
                completed = self.run_cli(
                    script, *args, "--corpus", str(self.pool_path), "--demo-corpus",
                )
                self.assertEqual(completed.returncode, 2)
                self.assertIn("specify only one", completed.stderr.decode("utf-8", errors="replace"))

    def test_run_rejects_geometry_mode_with_llm_pool(self):
        completed = self.run_cli(
            "run.py", "--llm-pool", str(self.pool_path), "--mode", "similar",
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("separate draw mode", completed.stderr.decode("utf-8", errors="replace"))


if __name__ == "__main__":
    unittest.main()
