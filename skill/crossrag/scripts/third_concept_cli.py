#!/usr/bin/env python3
"""Command-line entry point for third-concept exploration."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from engine import DEFAULT_CORPUS
from run import load_corpus
from third_concept import build_hypothesis_prompt, explore, load_hypotheses, make_encoder


def positive_limit(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("limit must be at least 1")
    return parsed


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Explore third-concept candidates from A and B")
    parser.add_argument("--A", required=True)
    parser.add_argument("--B", required=True)
    parser.add_argument("--corpus")
    parser.add_argument("--encoder", choices=("hash", "semantic"), default="hash")
    parser.add_argument("--model", default="intfloat/multilingual-e5-small")
    parser.add_argument("--limit", type=positive_limit, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--vocab")
    parser.add_argument("--hypotheses")
    parser.add_argument("--no-center", action="store_true")
    parser.add_argument("--emit-prompt", action="store_true")
    parser.add_argument("--with-math", action="store_true")
    args = parser.parse_args()
    try:
        if args.emit_prompt and not args.with_math:
            print(build_hypothesis_prompt(args.A, args.B))
            return
        corpus = load_corpus(args.corpus) if args.corpus else DEFAULT_CORPUS
        encoder = make_encoder(args.encoder, args.model)
        vocab = None
        if args.vocab:
            vocab = json.loads(Path(args.vocab).read_text(encoding="utf-8"))
            if not isinstance(vocab, list):
                raise ValueError("vocab must be a JSON string array")
        hypotheses = load_hypotheses(args.hypotheses, args.A, args.B) if args.hypotheses else None
        if args.emit_prompt:
            result = explore(args.A, args.B, corpus, encoder, args.limit, args.seed, vocab, None, not args.no_center, args.corpus)
            math_candidates = [item for item in result["candidates"] if item["origin"] == "math"]
            print(build_hypothesis_prompt(args.A, args.B, math_candidates))
            return
        result = explore(args.A, args.B, corpus, encoder, args.limit, args.seed, vocab, hypotheses, not args.no_center, args.corpus)
    except (ValueError, RuntimeError, ImportError, OSError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
