"""Checks the portable CrossRAG skill contract without host-specific tools."""
from __future__ import annotations

import re
import unittest
from pathlib import Path


SKILL_DIR = Path(__file__).resolve().parent.parent


def find_repository_root():
    for candidate in SKILL_DIR.parents:
        if (candidate / "README.md").is_file() and (candidate / "skill" / "crossrag" / "SKILL.md").is_file():
            return candidate
    return None


REPOSITORY_ROOT = find_repository_root()


def read_frontmatter():
    text = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        raise AssertionError("SKILL.md must start with YAML frontmatter")
    _, frontmatter, _ = text.split("---", 2)
    values = {}
    for line in frontmatter.splitlines():
        key, separator, value = line.partition(":")
        if separator:
            values[key.strip()] = value.strip().strip("\"'")
    return text, values


class SkillPortabilityTest(unittest.TestCase):
    def test_manifest_matches_portable_skill_requirements(self):
        _, metadata = read_frontmatter()

        self.assertEqual(metadata.get("name"), SKILL_DIR.name)
        self.assertRegex(metadata["name"], r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
        self.assertTrue(metadata.get("description"))
        self.assertLessEqual(len(metadata["description"]), 1024)
        self.assertEqual(metadata.get("license"), "MIT")
        self.assertIn("Python 3.10+", metadata.get("compatibility", ""))

    def test_standalone_bundle_contains_all_runtime_files_and_license(self):
        required = (
            "SKILL.md",
            "LICENSE",
            "requirements.txt",
            "agents/openai.yaml",
            "scripts/run.py",
            "scripts/engine.py",
            "scripts/llm_pool.py",
            "scripts/third_concept.py",
            "scripts/third_concept_cli.py",
            "scripts/serendipity_ray.py",
            "scripts/residual_ray.py",
            "scripts/compare_third_concept.py",
            "references/methodology.md",
            "references/modes.md",
            "references/example-corpus.json",
        )
        for relative_path in required:
            with self.subTest(path=relative_path):
                self.assertTrue((SKILL_DIR / relative_path).is_file())

        if REPOSITORY_ROOT is None:
            self.assertIn(b"MIT License", (SKILL_DIR / "LICENSE").read_bytes())
        else:
            self.assertEqual(
                (SKILL_DIR / "LICENSE").read_bytes(),
                (REPOSITORY_ROOT / "LICENSE").read_bytes(),
            )

    def test_skill_relative_markdown_links_and_bundled_paths_resolve(self):
        text, _ = read_frontmatter()
        body = text.split("---", 2)[2]
        targets = re.findall(r"\[[^\]]+\]\(([^)]+)\)", body)
        targets.extend(re.findall(r"`((?:scripts|references|agents)/[^\s`]+)`", body))
        self.assertTrue(targets, "expected local Markdown links or bundled paths to validate")
        for target in targets:
            if "://" in target or target.startswith("#"):
                continue
            local_path = target.split("#", 1)[0]
            with self.subTest(target=target):
                self.assertFalse(local_path.startswith("../"))
                self.assertTrue((SKILL_DIR / local_path).exists())

    def test_readme_covers_all_five_hosts_and_correct_shared_roots(self):
        if REPOSITORY_ROOT is None:
            self.skipTest("repository README is not part of a user-level skill install")
        readme_path = REPOSITORY_ROOT / "README.md"
        readme = readme_path.read_text(encoding="utf-8")
        for relative_path in (
            "LICENSE",
            "scripts/third_concept.py",
            "scripts/third_concept_cli.py",
            "scripts/compare_third_concept.py",
            "tests/",
        ):
            with self.subTest(path=relative_path):
                self.assertIn(relative_path, readme)
        for host in ("Codex", "Claude Code", "Gemini CLI", "Grok Build", "Pi"):
            with self.subTest(host=host):
                self.assertIn(host, readme)
        self.assertIn("~/.agents/skills/crossrag", readme)
        self.assertIn("~/.claude/skills/crossrag", readme)
        self.assertIn("__pycache__", readme)
        self.assertIn(".pytest_cache", readme)
        self.assertIn("探索先に含まれません", readme)
        gemini_row = next(line for line in readme.splitlines() if line.startswith("| Gemini CLI |"))
        self.assertIn("個別skillのslash起動は公式CLI資料に記載なし", gemini_row)
        self.assertIn("共有パスへのコピーと同時には使わない", readme)


if __name__ == "__main__":
    unittest.main()
