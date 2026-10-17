#!/usr/bin/env python3
"""CrossRAG skill CLI. Retrieval is numerical; writing is done by the host LLM."""
import argparse
import json
import sys
from pathlib import Path
from engine import DEFAULT_CORPUS, HashEncoder, SentenceEncoder, run


def load_corpus(path):
    if not path:
        return DEFAULT_CORPUS
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(data, list):
        raise ValueError('corpus must be a JSON array')
    rows = []
    for row in data:
        if isinstance(row, dict):
            row = [row.get('a'), row.get('b'), row.get('line')]
        if not isinstance(row, (list, tuple)) or len(row) != 3 or not all(isinstance(x, str) and x.strip() for x in row):
            raise ValueError('each corpus item must contain non-empty a, b, line strings')
        rows.append(tuple(row))
    return rows


def main():
    # Windows consoles default to a legacy code page; the JSON contract is UTF-8.
    sys.stdout.reconfigure(encoding='utf-8')
    p = argparse.ArgumentParser(description='CrossRAG creative/discovery retrieval')
    p.add_argument('--context', required=True, help='Two or more concepts separated by commas')
    p.add_argument('--mode', choices=['creative', 'discovery'], default='creative')
    p.add_argument('--method', choices=['similar', 'orthogonal', 'wedge', 'random'], default='wedge')
    p.add_argument('--encoder', choices=['hash', 'semantic'], default='hash')
    p.add_argument('--model', default='intfloat/multilingual-e5-small')
    p.add_argument('--corpus', help='JSON list of [a,b,line] or {a,b,line}')
    p.add_argument('--novelty', type=float, default=0.55)
    p.add_argument('--coherence', type=float, default=0.8)
    p.add_argument('--limit', type=int, default=5)
    args = p.parse_args()
    try:
        encoder = HashEncoder() if args.encoder == 'hash' else SentenceEncoder(args.model)
        data = run(args.context, load_corpus(args.corpus), args.method, args.novelty, args.coherence, args.limit, encoder)
    except (ValueError, RuntimeError, ImportError, OSError) as exc:
        p.error(str(exc))
    data['task_mode'] = args.mode
    data['evidence_type'] = 'retrieved creative material, not verified factual evidence'
    data['instruction'] = ('Write an original short narrative using 1-3 retrieved motifs; favour one surprising but coherent relational twist. Do not copy retrieved sentences verbatim.' if args.mode == 'creative' else 'Present 3 possible cross-domain connections as hypotheses. For each, explain the bridge, uncertainties and a minimum validation test. Never assert that retrieved text proves a claim.')
    print(json.dumps(data, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
