#!/usr/bin/env python3
"""Command-line entry point for third-concept exploration."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from engine import DEFAULT_CORPUS
from llm_pool import draw_llm_pool, load_llm_pool
from residual_ray import explore_residual_ray
from run import explicitly_supplied, load_corpus
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


def percentage(value):
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("serendipity must be an integer between 0 and 100") from exc
    if not 0 <= parsed <= 100:
        raise argparse.ArgumentTypeError("serendipity must be an integer between 0 and 100")
    return parsed


def build_host_prompt(contexts, selected_clusters, tone, tone_instruction, guardrails):
    directive = "この世界が成立する奇妙な法則を一つ作れ"
    prompt = (
        f"{directive}\n\n"
        f"contexts: {json.dumps(contexts, ensure_ascii=False)}\n"
        f"selected_clusters: {json.dumps(selected_clusters, ensure_ascii=False)}\n"
        f"tone: {tone}\n"
        f"tone_instruction: {tone_instruction}\n"
        f"guardrails: {json.dumps(guardrails, ensure_ascii=False)}"
    )
    return {
        "contexts": contexts,
        "selected_clusters": selected_clusters,
        "tone": tone,
        "tone_instruction": tone_instruction,
        "guardrails": guardrails,
        "prompt": prompt,
        "required_directive": directive,
    }


def read_corpus(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    # Any plain-text row routes the corpus to explore(), which validates every row; pair corpora keep the legacy loader.
    if isinstance(data, list) and any(is_plain_row(row) for row in data):
        return data
    return load_corpus(path)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Rank third-concept candidates or explore corpus sets along a normal ray")
    parser.add_argument("--A")
    parser.add_argument("--B")
    parser.add_argument("--C", help="source article text for residual-cluster ray search")
    parser.add_argument("--corpus")
    parser.add_argument("--demo-corpus", action="store_true", help="Explicitly use the bundled demo corpus")
    parser.add_argument("--llm-pool", help="JSON array of concepts generated from host LLM knowledge without conversation context")
    parser.add_argument("--llm-pool-model", help="Optional metadata naming the host LLM that generated the pool")
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
    parser.add_argument("--search-mode", choices=("third-concept", "ray", "random-orthogonal"), default="third-concept")
    parser.add_argument("--ray-method", choices=("residual-cluster", "pca-cross"), default="residual-cluster")
    parser.add_argument("--ray-radius", type=float, default=0.25)
    parser.add_argument("--ray-start-distance", type=float, default=0.25)
    parser.add_argument("--ray-cluster-cosine", type=float)
    parser.add_argument("--ray-min-cluster-size", type=positive_cluster_size)
    parser.add_argument("--ray-cluster-gap", type=float)
    parser.add_argument("--ray-min-set-size", type=positive_set_size)
    parser.add_argument("--emit-prompt", action="store_true")
    parser.add_argument("--with-math", action="store_true")
    parser.add_argument("--context", action="append", default=[])
    parser.add_argument("--ray-count", type=positive_limit, default=1)
    parser.add_argument("--serendipity", type=percentage, default=60)
    parser.add_argument("--law")
    argv = sys.argv[1:]
    args = parser.parse_args()
    try:
        if args.llm_pool is not None:
            if args.corpus is not None or args.demo_corpus:
                parser.error("--llm-pool cannot be combined with --corpus or --demo-corpus")
            if args.A is not None or args.B is not None:
                parser.error("--llm-pool is a context-free draw and cannot be combined with --A or --B")
            unsupported = (
                "--C", "--encoder", "--model", "--prefix-policy", "--exclude", "--method", "--p-min",
                "--null-samples", "--vocab", "--hypotheses", "--no-center", "--search-mode",
                "--ray-method", "--ray-radius", "--ray-start-distance", "--ray-cluster-cosine",
                "--ray-min-cluster-size", "--ray-cluster-gap", "--ray-min-set-size", "--emit-prompt",
                "--with-math", "--ray-count", "--serendipity", "--law",
            )
            if explicitly_supplied(argv, *unsupported):
                parser.error("--llm-pool cannot be combined with geometric, hypothesis, encoder, or --law options")
            pool_seed = args.seed if explicitly_supplied(argv, "--seed") else None
            result = draw_llm_pool(load_llm_pool(args.llm_pool), args.limit, pool_seed, args.llm_pool_model)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return
        if args.llm_pool_model is not None:
            parser.error("--llm-pool-model requires --llm-pool")
        has_new_options = explicitly_supplied(argv, "--context", "--ray-count", "--serendipity", "--law")
        if args.search_mode == "random-orthogonal":
            if args.A is not None or args.B is not None or args.C is not None:
                parser.error("random-orthogonal search accepts --context only and does not support --A, --B or --C")
            if len(args.context) < 2:
                parser.error("random-orthogonal search requires at least two --context values")
            if explicitly_supplied(
                argv, "--limit", "--method", "--p-min", "--null-samples", "--vocab",
                "--hypotheses", "--no-center", "--ray-method", "--ray-radius", "--ray-start-distance",
                "--ray-min-cluster-size", "--ray-cluster-gap", "--ray-min-set-size",
                "--emit-prompt", "--with-math",
            ):
                parser.error("random-orthogonal search does not support legacy ranking, hypothesis, maths, centre, vocab or ray-method options")
            if args.law is not None and not args.law.strip():
                parser.error("--law must be non-empty")
        else:
            if args.A is None:
                parser.error("the following arguments are required: --A")
            if args.B is None:
                parser.error("the following arguments are required: --B")
            if has_new_options:
                parser.error("--context, --ray-count, --serendipity and --law are only supported with --search-mode random-orthogonal")
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
        if args.search_mode == "random-orthogonal":
            if args.corpus is not None and args.demo_corpus:
                parser.error("specify only one of --corpus or --demo-corpus")
            if args.corpus is None and not args.demo_corpus:
                parser.error("one of --corpus or --demo-corpus is required; implicit demo corpus is disabled")
            corpus = read_corpus(args.corpus) if args.corpus is not None else DEFAULT_CORPUS
            from serendipity_ray import explore_random_orthogonal_ray

            encoder = make_encoder(args.encoder, args.model, prefix_policy=args.prefix_policy)
            result = explore_random_orthogonal_ray(
                args.context, corpus, encoder, seed=args.seed, ray_count=args.ray_count,
                level=args.serendipity, exclude=args.exclude, corpus_path=args.corpus,
                cosine_threshold=args.ray_cluster_cosine,
            )
            result.setdefault("params", {})["serendipity"] = args.serendipity
            result["corpus_source"] = "explicit" if args.corpus is not None else "demo"
            result["input"] = {"contexts": args.context}
            selected_clusters = result.get("combined_clusters", [])
            tone = result["params"].get("tone_band", "playful")
            tone_instruction = {
                "grounded": "身近でありそうな要素から、控えめで具体的な奇妙さを作る。",
                "playful": "選ばれたクラスターを手がかりに、簡潔で具体的な奇妙な法則として表現する。",
                "absurd": "大胆で不条理な飛躍を使い、創作であることが伝わる奇妙な法則にする。",
            }.get(tone, "選ばれたクラスターを手がかりに、簡潔で具体的な奇妙な法則として表現する。")
            result["params"]["tone"] = tone
            result["params"]["tone_instruction"] = tone_instruction
            result["workflow_status"] = "finalized" if args.law is not None else "awaiting_law"
            result["finalized"] = args.law is not None
            result["awaiting_host_law"] = args.law is None
            result["law"] = args.law
            result["host_prompt"] = None if args.law is not None else build_host_prompt(
                args.context, selected_clusters, tone, tone_instruction,
                ["法則は創作として提示し、現実の事実や科学的発見と主張しない。", "入力文脈や候補の真偽を捏造しない。"],
            )
            source_indices = sorted({
                hit["corpus_index"]
                for ray in result.get("rays", [])
                for hit in ray.get("hits", [])
                if isinstance(hit, dict) and isinstance(hit.get("corpus_index"), int)
            })
            result["provenance"] = {"source_corpus_indices": source_indices}
            if args.law is not None:
                result["provenance"]["law_source"] = "host_llm"
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return
        if args.emit_prompt and not args.with_math:
            print(build_hypothesis_prompt(args.A, args.B))
            return
        if args.corpus is not None and args.demo_corpus:
            parser.error("specify only one of --corpus or --demo-corpus")
        if args.corpus is None and not args.demo_corpus:
            parser.error("one of --corpus or --demo-corpus is required; implicit demo corpus is disabled")
        corpus = read_corpus(args.corpus) if args.corpus is not None else DEFAULT_CORPUS
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
            result["corpus_source"] = "explicit" if args.corpus is not None else "demo"
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return
        hypotheses = load_hypotheses(args.hypotheses, args.A, args.B) if args.hypotheses else None
        if args.emit_prompt:
            result = explore(
                args.A, args.B, corpus, encoder, args.limit, args.seed, vocab, None, not args.no_center, args.corpus,
                args.exclude, args.method, args.p_min, args.null_samples,
            )
            math_candidates = [item for item in result["candidates"] if item["origin"] == "math"]
            prompt = build_hypothesis_prompt(args.A, args.B, math_candidates)
            if args.demo_corpus:
                print(f"corpus_source: demo\n\n{prompt}")
            else:
                print(prompt)
            return
        result = explore(
            args.A, args.B, corpus, encoder, args.limit, args.seed, vocab, hypotheses, not args.no_center, args.corpus,
            args.exclude, args.method, args.p_min, args.null_samples,
        )
        result["corpus_source"] = "explicit" if args.corpus is not None else "demo"
    except (ValueError, RuntimeError, ImportError, OSError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
