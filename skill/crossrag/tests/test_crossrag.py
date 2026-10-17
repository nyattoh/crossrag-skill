"""Regression checks for the CrossRAG CLI. Standard library only."""
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent / 'scripts'
sys.path.insert(0, str(SCRIPTS))

from engine import DEFAULT_CORPUS, HashEncoder, run  # noqa: E402


class WedgeDegenerateTest(unittest.TestCase):
    def test_default_hash_wedge_is_all_zero_and_reported(self):
        # Context pair shares no characters with any corpus pair, so every wedge is 0.
        result = run('犬、カレー、夕暮れ、帰宅', DEFAULT_CORPUS, 'wedge', 0.55, 0.8, 5, HashEncoder())
        self.assertTrue(all(abs(row['wedge']) < 1e-9 for row in result['selected']))
        self.assertTrue(any('wedge' in w for w in result['warnings']))

    def test_warning_covers_all_candidates_not_only_selected(self):
        result = run('犬、カレー、夕暮れ、帰宅', DEFAULT_CORPUS, 'wedge', 0.55, 0.8, 1, HashEncoder())
        self.assertEqual(len(result['selected']), 1)
        self.assertTrue(any('wedge' in w for w in result['warnings']))

    def test_no_warning_when_only_an_unselected_candidate_has_wedge(self):
        # Top pick has wedge 0, but a non-selected candidate is nonzero: must not warn.
        corpus = DEFAULT_CORPUS + [('カレー', '犬', 'y')]
        result = run('犬、カレー、夕暮れ、帰宅', corpus, 'wedge', 0.55, 0.8, 1, HashEncoder())
        self.assertEqual(result['selected'][0]['wedge'], 0.0)
        self.assertEqual(result['warnings'], [])

    def test_warning_names_the_encoder(self):
        result = run('犬、カレー、夕暮れ、帰宅', DEFAULT_CORPUS, 'wedge', 0.55, 0.8, 5, HashEncoder())
        self.assertIn(HashEncoder.name, result['warnings'][0])

    def test_no_warning_when_wedge_varies(self):
        corpus = [('犬', 'カレー', 'a'), ('カレー', '犬', 'b'), ('猫', '雨', 'c')]
        result = run('犬、カレー', corpus, 'wedge', 0.55, 0.8, 3, HashEncoder())
        self.assertEqual(result['warnings'], [])


class StdoutEncodingTest(unittest.TestCase):
    def test_json_is_utf8_even_when_console_default_is_not(self):
        env = dict(os.environ, PYTHONIOENCODING='cp932')
        env.pop('PYTHONUTF8', None)
        out = subprocess.run(
            [sys.executable, str(SCRIPTS / 'run.py'), '--context', '犬、カレー', '--limit', '1'],
            capture_output=True, env=env, check=True,
        ).stdout
        data = json.loads(out.decode('utf-8'))
        self.assertEqual(data['context'], ['犬', 'カレー'])


if __name__ == '__main__':
    unittest.main()
