#!/usr/bin/env python3
"""Command-line entry point for third-concept exploration."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from engine import DEFAULT_CORPUS
from residual_ray import explore_residual_ray
from run import load_corpus
from third_concept import build_hypothesis_prompt, explore, explore_ray, is_plain_row, load_hypotheses, make_encoder


def positive_limit(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("limit must be at least 1")
    return parsed


def positive_null_samples(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("null samples must be at least 1")
    return parsed


def positive_set_size(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("ray minimum set size must be at least 1")
    return parsed


def positive_cluster_size(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("ray minimum cluster size must be at least 1")
    return parsed


def read_corpus(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    # Any plain-text row routes the corpus to explore(), which validates every row; pair corpora keep the legacy loader.
    if isinstance(data, list) and any(is_plain_row(row) for row in data):
        return data
    return load_corpus(path)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Rank third-concept candidates or explore corpus sets along a normal ray")
    parser.add_argument("--A", required=True)
    parser.add_argument("--B", required=True)
    parser.add_argument("--C", help="source article text for residual-cluster ray search")
    parser.add_argument("--corpus")
    parser.add_argument("--encoder", choices=("hash", "semantic"), default="hash")
    parser.add_argument("--model", default="intfloat/multilingual-e5-small")
    parser.add_argument("--prefix-policy", choices=("auto", "e5", "prefix-free"), default="auto")
    parser.add_argument("--limit", type=positive_limit, default=5, help="candidate limit or maximum number of rays (axial sets with pca-cross)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--exclude", action="append", default=[])
    parser.add_argument("--method", choices=("third-concept", "third-concept-ratio"), default="third-concept")
    parser.add_argument("--p-min", type=float)
    parser.add_argument("--null-samples", type=positive_null_samples, default=1000)
    parser.add_argument("--vocab")
    parser.add_argument("--hypotheses")
    parser.add_argument("--no-center", action="store_true")
    parser.add_argument("--search-mode", choices=("third-concept", "ray"), default="third-concept")
    parser.add_argument("--ray-method", choices=("residual-cluster", "pca-cross"), default="residual-cluster")
    parser.add_argument("--ray-radius", type=float, default=0.25)
    parser.add_argument("--ray-start-distance", type=float, default=0.25)
    parser.add_argument("--ray-cluster-cosine", type=float)
    parser.add_argument("--ray-min-cluster-size", type=positive_cluster_size)
    parser.add_argument("--ray-cluster-gap", type=float)
    parser.add_argument("--ray-min-set-size", type=positive_set_size)
    parser.add_argument("--emit-prompt", action="store_true")
    parser.add_argument("--with-math", action="store_true")
    args = parser.parse_args()
    try:
        if args.C is not None and args.search_mode != "ray":
            parser.error("--C is only supported with --search-mode ray")
        if args.search_mode == "ray" and (
            args.emit_prompt or args.hypotheses or args.with_math or args.method != "third-concept"
            or args.seed != 0 or args.null_samples != 1000 or args.p_min is not None
        ):
            parser.error("ray search does not support --emit-prompt, --hypotheses, --with-math, ranking --method, seed, null samples or p-min")
        if args.search_mode == "ray":
            if args.ray_method == "residual-cluster":
                if args.C is None:
                    parser.error("residual-cluster ray search requires --C")
                unsupported = (
                    ("--no-center", args.no_center),
                    ("--vocab", args.vocab is not None),
                    ("--ray-cluster-gap", args.ray_cluster_gap is not None),
                    ("--ray-min-set-size", args.ray_min_set_size is not None),
                )
            else:
                unsupported = (
                    ("--C", args.C is not None),
                    ("--ray-cluster-cosine", args.ray_cluster_cosine is not None),
                    ("--ray-min-cluster-size", args.ray_min_cluster_size is not None),
                )
            for flag, supplied in unsupported:
                if supplied:
                    parser.error(f"{args.ray_method} ray search does not support {flag}")
        if args.emit_prompt and not args.with_math:
            print(build_hypothesis_prompt(args.A, args.B))
            return
        corpus = read_corpus(args.corpus) if args.corpus else DEFAULT_CORPUS
        encoder = make_encoder(args.encoder, args.model, prefix_policy=args.prefix_policy)
        vocab = None
        if args.vocab:
            vocab = json.loads(Path(args.vocab).read_text(encoding="utf-8"))
            if not isinstance(vocab, list):
                raise ValueError("vocab must be a JSON string array")
        if args.search_mode == "ray":
            if args.ray_method == "residual-cluster":
                result = explore_residual_ray(
                    args.A, args.B, args.C, corpus, encoder,
                    corpus_path=args.corpus, exclude=args.exclude, radius=args.ray_radius,
                    start_distance=args.ray_start_distance,
                    cluster_cosine=0.7 if args.ray_cluster_cosine is None else args.ray_cluster_cosine,
                    min_cluster_size=2 if args.ray_min_cluster_size is None else args.ray_min_cluster_size,
                    max_rays=args.limit,
                )
            else:
                result = explore_ray(
                    args.A, args.B, corpus, encoder, vocab=vocab, center=not args.no_center,
                    corpus_path=args.corpus, exclude=args.exclude, radius=args.ray_radius,
                    start_distance=args.ray_start_distance,
                    cluster_gap=0.25 if args.ray_cluster_gap is None else args.ray_cluster_gap,
                    min_set_size=2 if args.ray_min_set_size is None else args.ray_min_set_size,
                    max_sets=args.limit,
                )
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return
        hypotheses = load_hypotheses(args.hypotheses, args.A, args.B) if args.hypotheses else None
        if args.emit_prompt:
            result = explore(
                args.A, args.B, corpus, encoder, args.limit, args.seed, vocab, None, not args.no_center, args.corpus,
                args.exclude, args.method, args.p_min, args.null_samples,
            )
            math_candidates = [item for item in result["candidates"] if item["origin"] == "math"]
            print(build_hypothesis_prompt(args.A, args.B, math_candidates))
            return
        result = explore(
            args.A, args.B, corpus, encoder, args.limit, args.seed, vocab, hypotheses, not args.no_center, args.corpus,
            args.exclude, args.method, args.p_min, args.null_samples,
        )
    except (ValueError, RuntimeError, ImportError, OSError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
