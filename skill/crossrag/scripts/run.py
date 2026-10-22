#!/usr/bin/env python3
"""CrossRAG skill CLI. Retrieval is numerical; raw mathematical results are returned."""
import argparse
import json
import re
import sys
from pathlib import Path
from engine import DEFAULT_CORPUS, HashEncoder, SentenceEncoder, run

_UTTERANCE_KEYS = ('content', 'text', 'message', 'line', 'utterance')
_TOKEN_SPLIT_RE = re.compile(r'[、,，。．.!！?？:：;；/／「」『』（）()\[\]\s]+')
_JSON_DOC_START_RE = re.compile(r'^(?:\{\s*(?:"|\}|\r|\n)|\[\s*(?:\{|\[|"|\]|\r|\n))')


def load_corpus(path):
    if not path:
        raise ValueError('corpus path is required unless --demo-corpus or --conversation is specified')
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


def _extract_pair_from_text(text):
    cleaned = text.strip()
    clauses = [c.strip() for c in re.split(r'[、,，。．.!！?？/／\s]+', cleaned) if c.strip()]
    if len(clauses) >= 2:
        return clauses[0], clauses[1], cleaned
    midpoint = max(1, len(cleaned) // 2)
    left = cleaned[:midpoint].strip() or cleaned
    right = cleaned[midpoint:].strip() or cleaned
    return left, right, cleaned


def _row_from_conversation_item(item):
    if isinstance(item, (list, tuple)):
        if len(item) == 3 and all(isinstance(x, str) and x.strip() for x in item):
            return tuple(x.strip() for x in item)
        raise ValueError('conversation array item must be [a, b, line] with non-empty strings')
    if isinstance(item, dict):
        if all(isinstance(item.get(k), str) and item.get(k).strip() for k in ('a', 'b', 'line')):
            return (item['a'].strip(), item['b'].strip(), item['line'].strip())
        for key in _UTTERANCE_KEYS:
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                a = item.get('a')
                b = item.get('b')
                if isinstance(a, str) and a.strip() and isinstance(b, str) and b.strip():
                    return (a.strip(), b.strip(), value.strip())
                return _extract_pair_from_text(value)
        return None
    if isinstance(item, str):
        return _extract_pair_from_text(item) if item.strip() else None
    return None


def _looks_like_json_source(path, stripped):
    suffix = Path(path).suffix.lower()
    if suffix in {'.json', '.jsonl', '.ndjson'}:
        return True
    return bool(_JSON_DOC_START_RE.match(stripped))


def load_conversation_corpus(path):
    if not path:
        raise ValueError('conversation path is required')
    raw_text = Path(path).read_text(encoding='utf-8')
    stripped = raw_text.strip()
    if not stripped:
        raise ValueError('conversation file is empty')
    rows = []
    if _looks_like_json_source(path, stripped):
        try:
            parsed_json = json.loads(stripped)
        except json.JSONDecodeError as exc:
            non_empty_lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
            try:
                parsed_json = [json.loads(line) for line in non_empty_lines]
            except json.JSONDecodeError:
                raise ValueError(f'invalid conversation JSON: {exc}') from exc
            if not all(isinstance(item, (dict, list)) for item in parsed_json):
                raise ValueError(f'invalid conversation JSONL: {exc}') from exc
        if isinstance(parsed_json, dict):
            for key in ('messages', 'conversation', 'turns', 'utterances', 'items', 'corpus'):
                if isinstance(parsed_json.get(key), list):
                    parsed_json = parsed_json[key]
                    break
            else:
                raise ValueError('conversation JSON object must contain a messages/conversation/turns/utterances/items/corpus array')
        if not isinstance(parsed_json, list):
            raise ValueError('conversation JSON must be an array or an object containing a message array')
        for item in parsed_json:
            row = _row_from_conversation_item(item)
            if row is not None:
                rows.append(row)
    else:
        for raw_line in raw_text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if _JSON_DOC_START_RE.match(line):
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f'invalid JSON line in conversation file: {exc}') from exc
                row = _row_from_conversation_item(obj)
                if row is not None:
                    rows.append(row)
                continue
            line = re.sub(
                r'^(?:(?:\[[^\]\r\n]+\]|\{[^{}\r\n]+\})\s*[:：]?\s*)*(?:(?:user|assistant|system|human|ai|speaker|[\w\-]+)\s*[:：]\s*)?',
                '',
                line,
                flags=re.IGNORECASE,
            )
            if line.strip():
                rows.append(_extract_pair_from_text(line))
    if len(rows) < 2:
        raise ValueError('会話データから少なくとも2件の発話候補が必要です')
    return rows


def main():
    # Windows consoles default to a legacy code page; the JSON contract is UTF-8.
    sys.stdout.reconfigure(encoding='utf-8')
    p = argparse.ArgumentParser(description='CrossRAG single-metric geometric retrieval')
    p.add_argument('--context', required=True, help='Two or more concepts separated by commas')
    p.add_argument('--mode', choices=['similar', 'orthogonal', 'wedge'], default=None, help='Single ranking metric mode')
    p.add_argument('--method', choices=['similar', 'orthogonal', 'wedge'], default=None, help='Alias for --mode')
    p.add_argument('--encoder', choices=['hash', 'semantic'], default='hash')
    p.add_argument('--model', default='intfloat/multilingual-e5-small')
    p.add_argument('--corpus', help='JSON list of [a,b,line] or {a,b,line}')
    p.add_argument('--conversation', help='Conversation log (JSON, JSONL, or plain text) to extract candidate utterances from')
    p.add_argument('--demo-corpus', action='store_true', help='Explicitly use the bundled 22-item demo corpus')
    p.add_argument('--limit', type=int, default=5)
    args = p.parse_args()
    if args.mode and args.method and args.mode != args.method:
        p.error('--mode and --method must match when both are supplied')
    selected_mode = args.mode or args.method or 'orthogonal'
    sources = [bool(args.corpus), bool(args.conversation), bool(args.demo_corpus)]
    if sum(sources) == 0:
        p.error('one of --conversation, --corpus, or --demo-corpus is required (implicit demo corpus is disabled)')
    if sum(sources) > 1:
        p.error('specify only one of --conversation, --corpus, or --demo-corpus')
    try:
        encoder = HashEncoder() if args.encoder == 'hash' else SentenceEncoder(args.model)
        if args.demo_corpus:
            data = run(args.context, None, selected_mode, args.limit, encoder, use_demo_corpus=True, corpus_source='demo')
        elif args.conversation:
            corpus = load_conversation_corpus(args.conversation)
            data = run(args.context, corpus, selected_mode, args.limit, encoder, use_demo_corpus=False, corpus_source='conversation')
        else:
            corpus = load_corpus(args.corpus)
            data = run(args.context, corpus, selected_mode, args.limit, encoder, use_demo_corpus=False, corpus_source='explicit')
    except (ValueError, RuntimeError, ImportError, OSError, json.JSONDecodeError) as exc:
        p.error(str(exc))
    print(json.dumps(data, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
