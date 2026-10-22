"""Regression and contract checks for the CrossRAG single-metric engine and CLI."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent.parent / 'scripts'
sys.path.insert(0, str(SCRIPTS))

from engine import DEFAULT_CORPUS, HashEncoder, run  # noqa: E402
from run import load_conversation_corpus  # noqa: E402


class SingleMetricEngineTest(unittest.TestCase):
    def test_corpus_none_without_demo_flag_raises_value_error(self):
        with self.assertRaises(ValueError):
            run('犬、カレー', corpus=None)
        with self.assertRaises(ValueError):
            run('犬、カレー', corpus=None, use_demo_corpus=False)

    def test_use_demo_corpus_true_uses_default_corpus_and_records_source(self):
        result = run('犬、カレー', corpus=None, mode='orthogonal', limit=5, encoder=HashEncoder(), use_demo_corpus=True)
        self.assertEqual(result['corpus_source'], 'demo')
        self.assertEqual(result['total'], len(DEFAULT_CORPUS))
        self.assertEqual(result['encoder'], HashEncoder.name)
        self.assertEqual(result['note'], '順位は指定された単一指標のみで決定。複合スコアは使用しない')

    def test_explicit_corpus_records_explicit_source(self):
        result = run('犬、カレー', corpus=DEFAULT_CORPUS[:4], mode='similar', limit=3, encoder=HashEncoder())
        self.assertEqual(result['corpus_source'], 'explicit')
        self.assertEqual(result['total'], 4)

    def test_conversation_corpus_source_is_preserved(self):
        result = run(
            '犬、カレー',
            corpus=DEFAULT_CORPUS[:4],
            mode='orthogonal',
            limit=3,
            encoder=HashEncoder(),
            corpus_source='conversation',
        )
        self.assertEqual(result['corpus_source'], 'conversation')

    def test_unknown_modes_including_random_are_rejected(self):
        for bad_mode in ('random', 'invalid', ''):
            with self.subTest(mode=bad_mode), self.assertRaises(ValueError):
                run('犬、カレー', DEFAULT_CORPUS, mode=bad_mode)

    def test_each_mode_ranks_by_its_single_metric_and_omits_score_key(self):
        expected_metrics = {
            'similar': ('cosine', 'cosine'),
            'orthogonal': ('orthogonal_residual', 'orthogonal'),
            'wedge': ('gram_bivector_cos', 'wedge'),
        }
        corpus = DEFAULT_CORPUS + [('カレー', '犬', '逆向きのペアを含むテスト行。')]
        for mode, (metric_name, row_field) in expected_metrics.items():
            with self.subTest(mode=mode):
                result = run('犬、カレー、夕暮れ', corpus, mode=mode, limit=10, encoder=HashEncoder())
                self.assertEqual(result['mode'], mode)
                self.assertEqual(result['ranking_metric'], metric_name)
                self.assertEqual(result['note'], '順位は指定された単一指標のみで決定。複合スコアは使用しない')
                self.assertNotIn('score', result)
                selected = result['selected']
                self.assertEqual(len(selected), 10)
                for row in selected:
                    self.assertNotIn('score', row)
                    self.assertEqual(row['ranking_metric'], metric_name)
                    self.assertEqual(row['rank_value'], row[row_field])
                # Verify descending order by rank_value (tie-broken by ascending index)
                pairs = [(-row['rank_value'], row['index']) for row in selected]
                self.assertEqual(pairs, sorted(pairs))

    def test_ranking_sorts_by_unrounded_metric_before_rounding_output(self):
        class SubRoundingEncoder:
            name = 'sub-rounding'

            def encode(self, texts):
                # Context anchors average to [1, 0, 0].
                # Candidate 0 and Candidate 1 both round to 0.50000 at 5 decimals,
                # but Candidate 1 has a slightly higher unrounded cosine (0.500004 > 0.500001).
                c0 = 0.500001
                c1 = 0.500004
                mapping = {
                    'A': [1.0, 0.0, 0.0],
                    'B': [1.0, 0.0, 0.0],
                    'a0とb0。l0': [c0, np.sqrt(1.0 - c0 * c0), 0.0],
                    'a1とb1。l1': [c1, np.sqrt(1.0 - c1 * c1), 0.0],
                    'a0': [1.0, 0.0, 0.0],
                    'b0': [0.0, 1.0, 0.0],
                    'a1': [1.0, 0.0, 0.0],
                    'b1': [0.0, 1.0, 0.0],
                }
                return np.asarray([mapping[t] for t in texts], dtype=np.float64)

        corpus = [('a0', 'b0', 'l0'), ('a1', 'b1', 'l1')]
        result = run('A、B', corpus, mode='similar', limit=2, encoder=SubRoundingEncoder())
        self.assertEqual([row['index'] for row in result['selected']], [1, 0])
        self.assertEqual(result['selected'][0]['rank_value'], 0.5)
        self.assertEqual(result['selected'][1]['rank_value'], 0.5)


class WedgeDegenerateTest(unittest.TestCase):
    def test_default_hash_wedge_is_all_zero_and_reported_without_band_wording(self):
        # Context pair shares no characters with any corpus pair, so every wedge is 0.
        result = run('犬、カレー、夕暮れ、帰宅', DEFAULT_CORPUS, 'wedge', 5, HashEncoder())
        self.assertTrue(all(abs(row['wedge']) < 1e-9 for row in result['selected']))
        self.assertTrue(any('wedge' in w for w in result['warnings']))
        for warning in result['warnings']:
            self.assertNotIn('帯域', warning)

    def test_warning_covers_all_candidates_not_only_selected(self):
        result = run('犬、カレー、夕暮れ、帰宅', DEFAULT_CORPUS, 'wedge', 1, HashEncoder())
        self.assertEqual(len(result['selected']), 1)
        self.assertTrue(any('wedge' in w for w in result['warnings']))

    def test_no_warning_when_only_an_unselected_candidate_has_wedge(self):
        # A non-selected candidate is negative (-1.0) while top picks are 0.0: must not warn.
        corpus = DEFAULT_CORPUS + [('カレー', '犬', 'y')]
        result = run('犬、カレー、夕暮れ、帰宅', corpus, 'wedge', 1, HashEncoder())
        self.assertEqual(result['selected'][0]['wedge'], 0.0)
        self.assertEqual(result['warnings'], [])

    def test_warning_names_the_encoder(self):
        result = run('犬、カレー、夕暮れ、帰宅', DEFAULT_CORPUS, 'wedge', 5, HashEncoder())
        self.assertIn(HashEncoder.name, result['warnings'][0])

    def test_no_warning_when_wedge_varies(self):
        corpus = [('犬', 'カレー', 'a'), ('カレー', '犬', 'b'), ('猫', '雨', 'c')]
        result = run('犬、カレー', corpus, 'wedge', 3, HashEncoder())
        self.assertEqual(result['warnings'], [])
        self.assertGreater(result['selected'][0]['rank_value'], 0.0)


class RunCliContractTest(unittest.TestCase):
    def run_cli(self, *args, env=None):
        return subprocess.run(
            [sys.executable, str(SCRIPTS / 'run.py'), *args],
            capture_output=True,
            env=env,
            check=False,
        )

    def test_requires_corpus_or_conversation_or_demo_corpus(self):
        completed = self.run_cli('--context', '犬、カレー')
        self.assertEqual(completed.returncode, 2)
        self.assertEqual(completed.stdout, b'')

    def test_demo_corpus_flag_enables_demo_and_sets_corpus_source_demo(self):
        completed = self.run_cli('--context', '犬、カレー', '--demo-corpus', '--mode', 'orthogonal', '--limit', '3')
        self.assertEqual(completed.returncode, 0, completed.stderr.decode('utf-8', errors='replace'))
        data = json.loads(completed.stdout.decode('utf-8'))
        self.assertEqual(data['corpus_source'], 'demo')
        self.assertEqual(data['mode'], 'orthogonal')
        self.assertEqual(data['ranking_metric'], 'orthogonal_residual')
        self.assertEqual(data['encoder'], HashEncoder.name)
        self.assertTrue(all('score' not in row for row in data['selected']))

    def test_explicit_corpus_sets_corpus_source_explicit(self):
        corpus_path = SCRIPTS.parent / 'references' / 'example-corpus.json'
        completed = self.run_cli('--context', '犬、カレー', '--corpus', str(corpus_path), '--mode', 'similar')
        self.assertEqual(completed.returncode, 0, completed.stderr.decode('utf-8', errors='replace'))
        data = json.loads(completed.stdout.decode('utf-8'))
        self.assertEqual(data['corpus_source'], 'explicit')
        self.assertEqual(data['ranking_metric'], 'cosine')

    def test_conversation_json_and_text_inputs_set_corpus_source_conversation(self):
        with tempfile.TemporaryDirectory() as tmp:
            conv_json = Path(tmp) / 'conv.json'
            conv_json.write_text(
                json.dumps(
                    {
                        'messages': [
                            {'role': 'user', 'content': '雨の日の窓辺、時計の針が静かに進む。'},
                            {'role': 'assistant', 'content': '電車の約束、夕暮れの駅で待っている。'},
                            {'role': 'user', 'content': '台所の記憶、スープの湯気が立ちのぼる。'},
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding='utf-8',
            )
            completed_json = self.run_cli(
                '--context', '犬、カレー、夕暮れ', '--conversation', str(conv_json), '--mode', 'orthogonal', '--limit', '2'
            )
            self.assertEqual(completed_json.returncode, 0, completed_json.stderr.decode('utf-8', errors='replace'))
            data_json = json.loads(completed_json.stdout.decode('utf-8'))
            self.assertEqual(data_json['corpus_source'], 'conversation')
            self.assertEqual(data_json['total'], 3)
            self.assertEqual(data_json['encoder'], HashEncoder.name)
            self.assertEqual(len(data_json['selected']), 2)
            for row in data_json['selected']:
                self.assertNotIn('score', row)
                self.assertEqual(row['rank_value'], row['orthogonal'])

            conv_txt = Path(tmp) / 'conv.txt'
            conv_txt.write_text(
                'user: 猫の地図、街の裏側を歩く。\nassistant: 傘と月、夜道で空を見上げる。\n',
                encoding='utf-8',
            )
            completed_txt = self.run_cli(
                '--context', '犬、カレー', '--conversation', str(conv_txt), '--mode', 'similar'
            )
            self.assertEqual(completed_txt.returncode, 0, completed_txt.stderr.decode('utf-8', errors='replace'))
            data_txt = json.loads(completed_txt.stdout.decode('utf-8'))
            self.assertEqual(data_txt['corpus_source'], 'conversation')
            self.assertEqual(data_txt['total'], 2)
            self.assertEqual(data_txt['ranking_metric'], 'cosine')

    def test_conversation_malformed_json_is_rejected_not_parsed_as_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad_pretty = Path(tmp) / 'broken.txt'
            bad_pretty.write_text(
                '{\n  "messages": [\n    {"role": "user", "content": "雨の窓辺、静かな時計"}\n    {"role": "assistant", "content": "カンマ抜けの壊れたJSON"}\n  ]\n}\n',
                encoding='utf-8',
            )
            with self.assertRaises(ValueError):
                load_conversation_corpus(bad_pretty)
            completed = self.run_cli('--context', '犬、カレー', '--conversation', str(bad_pretty))
            self.assertEqual(completed.returncode, 2)

    def test_rejects_removed_options_and_random_mode(self):
        for bad_args in (
            ('--context', '犬、カレー', '--demo-corpus', '--mode', 'random'),
            ('--context', '犬、カレー', '--demo-corpus', '--novelty', '0.5'),
            ('--context', '犬、カレー', '--demo-corpus', '--coherence', '0.8'),
        ):
            with self.subTest(args=bad_args):
                completed = self.run_cli(*bad_args)
                self.assertEqual(completed.returncode, 2)

    def test_json_is_utf8_even_when_console_default_is_not(self):
        env = dict(os.environ, PYTHONIOENCODING='cp932')
        env.pop('PYTHONUTF8', None)
        completed = self.run_cli('--context', '犬、カレー', '--demo-corpus', '--limit', '1', env=env)
        self.assertEqual(completed.returncode, 0, completed.stderr.decode('utf-8', errors='replace'))
        data = json.loads(completed.stdout.decode('utf-8'))
        self.assertEqual(data['context'], ['犬', 'カレー'])


if __name__ == '__main__':
    unittest.main()
