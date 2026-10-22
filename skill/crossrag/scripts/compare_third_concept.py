#!/usr/bin/env python3
"""Compare CrossRAG single-metric retrieval with third-concept rankings."""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from pathlib import Path

import numpy as np

from engine import DEFAULT_CORPUS, normalise, run
from llm_pool import draw_llm_pool, load_llm_pool
from run import explicitly_supplied
from third_concept import (
    _contains_input,
    _normalised_exclude_terms,
    _normalised_text,
    _rows,
    _validated_p_min,
    auto_p_min,
    corpus_id,
    exclusion_mask,
    geometry,
    make_encoder,
    null_geometry,
    rank_entries,
    shared_geometry_space,
    token_geometry,
)
from third_concept_cli import read_corpus


METHOD_NAMES = (
    "similar",
    "orthogonal",
    "wedge",
    "third-concept",
    "mean",
    "third-concept-no-wedge",
)
PLAIN_METHOD_NAMES = (
    "similar",
    "third-concept",
    "mean",
    "third-concept-no-wedge",
)
BASELINE_METHODS = ("similar", "orthogonal", "wedge")
RATIO_METHOD = "third-concept-ratio"
THIRD_CONCEPT_METHODS = ("third-concept", "mean", "third-concept-no-wedge")
METRIC_NAMES = (
    "s_A",
    "s_B",
    "plane_projection",
    "residual",
    "emergence_norm",
    "score_third",
)
RATIO_METRIC_NAMES = (
    "emergence_ratio",
    "emergence_ratio_norm",
    "score_ratio",
    "balance",
    "p_below_min_fraction",
    "between_AB_fraction",
)
NOTES = [
    'similar / orthogonal / wedge は engine.run() を通り、その encode() は全テキストに "query: " を適用する。',
    "third-concept は score_third を目的関数として選ぶため、score_third の平均で高くなることは循環的であり、各平均は記述値にすぎず、優越性の証拠ではない。",
    "third-concept がベースラインと異ならない結果も、そのまま報告する。",
    "Baselines are not exclusion-filtered; baseline の contains_input は入力概念との文字列一致・包含を示す。",
    "seed は評価シートのシャッフルと null 標本にだけ使われる。",
]
EVALUATION_NOTES = [
    "wedge ablation compares each third-concept score with balance alone and with third-concept-no-wedge over all eligible rows.",
    "null random uses seeded isotropic vectors; null shuffled uses seeded coordinate permutations of centred passage vectors.",
    "percentile uses the mid-rank definition: 100 * (count(null < x) + 0.5 * count(null == x)) / n.",
    "random/shuffled null percentiles and p_min are heuristic references; they do not model topic-matched controls or correct multiple comparisons.",
    "Results are reported even when they do not outperform baselines; objective-score means do not establish superiority.",
]
DISCLAIMER = "全指標は探索用ヒューリスティックであり、人手評価は未実施。方式間の品質差や優越性を実証しない。"


def calibrate_control_pairs(A, B, corpus, encoder, *, control_pairs=None, control_words=None, samples=20, seed=0, limit=5, input_kind="word", exclude=(), null_samples=1000, p_min=None):
    """Compare the requested pair with seeded, corpus/model-matched lexical controls.

    Calibration is descriptive; interpretation depends on how well the sampled
    lexical controls represent the comparison of interest.
    """
    sampling_method = "provided" if control_pairs is not None else "uniform-without-replacement-from-word-combinations"
    if type(samples) is not int or samples < 1:
        raise ValueError("samples must be an integer at least 1")
    if control_pairs is None:
        words = list(dict.fromkeys(control_words or []))
        if len(words) < 2:
            raise ValueError("provide control_pairs or at least two control_words")
        all_pairs = [(left, right) for i, left in enumerate(words) for right in words[i + 1:]
                     if {left.casefold(), right.casefold()} != {A.casefold(), B.casefold()}]
        if not all_pairs:
            raise ValueError("control_words do not produce any non-target pairs")
        rng = np.random.default_rng(seed)
        chosen = rng.permutation(len(all_pairs))[:min(samples, len(all_pairs))]
        control_pairs = [all_pairs[int(i)] for i in chosen]
    control_pairs = [tuple(pair) for pair in control_pairs]
    if not control_pairs or any(len(pair) != 2 or not all(isinstance(v, str) and v.strip() for v in pair) for pair in control_pairs):
        raise ValueError("control_pairs must contain non-empty two-item pairs")
    if any({a.casefold(), b.casefold()} == {A.casefold(), B.casefold()} for a, b in control_pairs):
        raise ValueError("control_pairs must not contain the target pair")

    candidate_texts = [row[2] if isinstance(row, (list, tuple)) else row.get("line", "") if isinstance(row, dict) else str(row) for row in corpus]
    token_encoder = getattr(encoder, "encode_passage_tokens", None)
    cached_token_vectors = token_encoder(candidate_texts) if token_encoder else [np.asarray([v]) for v in encoder.encode_passage(candidate_texts)]

    def summarize(left, right):
        result, space, metrics = _compare(left, right, corpus, encoder, limit=limit, seed=seed,
                                          input_kind=input_kind, exclude=exclude, p_min=p_min,
                                          null_samples=null_samples)
        eligible_indices = [i for i, excluded_row in enumerate(exclusion_mask(space, left, right, exclude)) if not excluded_row]
        valid_count = sum(bool(metrics[i]["between_AB"]) for i in eligible_indices)
        valid_rate = valid_count / len(eligible_indices) if eligible_indices else 0.0
        anchor_a = encoder.encode_query([left])[0]
        anchor_b = encoder.encode_query([right])[0]
        token_rows = token_geometry(anchor_a, anchor_b, [cached_token_vectors[i] for i in eligible_indices])
        token_scores = {i: entry["score"] for i, entry in zip(eligible_indices, token_rows)}
        token_order = sorted(token_scores, key=lambda index: (-token_scores[index], index))
        token_top = token_order[:limit]
        joint_supported = [
            entry["a_side_support"] > 0 and entry["b_side_support"] > 0
            and entry["between_direction_support"] > 0.5 and entry["plane_support"] > 0
            for entry in token_rows
        ]
        def mean_criterion(key):
            return float(np.mean([entry[key] for entry in token_rows])) if token_rows else 0.0
        return {
            "valid_rate": valid_rate,
            "valid_count": valid_count,
            "emergence_ratio_mean": float(np.mean([metrics[i]["emergence_ratio"] for i in eligible_indices])) if eligible_indices else 0.0,
            "top_k_emergence_ratio_mean": result["evaluation"]["ratio_means"][RATIO_METHOD]["emergence_ratio"],
            "top_k_valid_rate": result["evaluation"]["ratio_means"][RATIO_METHOD]["between_AB_fraction"],
            "top_k": [item["corpus_index"] for item in result["evaluation"]["methods"][RATIO_METHOD]["selected"]],
            "token_wedge_valid_rate": float(np.mean(joint_supported)) if joint_supported else 0.0,
            "token_joint_supported_count": int(sum(joint_supported)),
            "token_joint_supported_rate": float(np.mean(joint_supported)) if joint_supported else 0.0,
            "token_score_mean": mean_criterion("score"),
            "token_a_side_support_mean": mean_criterion("a_side_support"),
            "token_b_side_support_mean": mean_criterion("b_side_support"),
            "token_between_direction_support_mean": mean_criterion("between_direction_support"),
            "token_plane_support_mean": mean_criterion("plane_support"),
            "token_wedge_top_k": token_top,
            "token_wedge_top_k_mean": float(np.mean([token_scores[index] for index in token_top])) if token_top else 0.0,
            "eligible_count": len(eligible_indices),
        }

    target = summarize(A, B)
    controls = [{"pair": list(pair), **summarize(*pair)} for pair in control_pairs]
    values = [item["valid_rate"] for item in controls]
    target["pair_percentile"] = midrank_percentile(values, target["valid_rate"])
    target["rank_descending"] = 1 + sum(value > target["valid_rate"] for value in values)
    token_values = [item["token_wedge_valid_rate"] for item in controls]
    target["token_wedge_pair_percentile"] = midrank_percentile(token_values, target["token_wedge_valid_rate"])
    target["token_wedge_rank_descending"] = 1 + sum(value > target["token_wedge_valid_rate"] for value in token_values)
    target["control_count"] = len(controls)
    target_set = set(target["top_k"])
    for item in controls:
        other = set(item["top_k"])
        item["top_k_jaccard_vs_target"] = len(target_set & other) / len(target_set | other) if target_set | other else 1.0
        token_other = set(item["token_wedge_top_k"])
        token_target = set(target["token_wedge_top_k"])
        item["token_wedge_top_k_jaccard_vs_target"] = len(token_other & token_target) / len(token_other | token_target) if token_other | token_target else 1.0
    return {
        "token_method": "third-concept-token-joint-v2",
        "token_aggregation": "mean-vector-geometric-v2",
        "target": target,
        "controls": controls,
        "sampling": {"seed": int(seed), "method": sampling_method},
        "limitations": {
            "independence_assumption": "Descriptive lexical controls may share words; target/control lexical properties, corpus topic structure, and embedding geometry can differ. Pair sampling is not independent and may have orientation bias.",
            "interpretation": "Calibration values are descriptive comparisons over the supplied candidates and controls.",
        },
    }


def _normalise_input(text):
    return re.sub(r"[、,，/／\s]+", "_", text.strip()).strip("_")


def _public_metrics(metrics):
    return {name: round(float(metrics[name]), 6) for name in METRIC_NAMES}


def _mean_metrics(indices, common_metrics):
    if not indices:
        return {name: 0.0 for name in METRIC_NAMES}
    return {
        name: round(float(np.mean([common_metrics[index][name] for index in indices])), 6)
        for name in METRIC_NAMES
    }


def _ratio_means(indices, common_metrics):
    if not indices:
        return {name: 0.0 for name in RATIO_METRIC_NAMES}
    values = {
        "emergence_ratio": [common_metrics[index]["emergence_ratio"] for index in indices],
        "emergence_ratio_norm": [common_metrics[index]["emergence_ratio_norm"] for index in indices],
        "score_ratio": [common_metrics[index]["score_ratio"] for index in indices],
        "balance": [common_metrics[index]["balance"] for index in indices],
        "p_below_min_fraction": [common_metrics[index]["p_below_min"] for index in indices],
        "between_AB_fraction": [common_metrics[index]["between_AB"] for index in indices],
    }
    return {name: round(float(np.mean(value)), 6) for name, value in values.items()}


def _jaccard(left, right):
    union = left | right
    return round(len(left & right) / len(union), 6) if union else 1.0


def kendall_tau_b(x, y):
    """Return Kendall's tau-b using an O(n^2) pair count."""
    # ponytail: O(n^2) is fine for ~1e3 rows; use Knight's O(n log n) algorithm for many thousands.
    left = list(x)
    right = list(y)
    if len(left) != len(right):
        raise ValueError("x and y must have the same length")
    if len(left) < 2:
        return None
    concordant = discordant = ties_left = ties_right = 0
    for first in range(len(left) - 1):
        for second in range(first + 1, len(left)):
            dx = left[first] - left[second]
            dy = right[first] - right[second]
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                ties_left += 1
            elif dy == 0:
                ties_right += 1
            elif dx * dy > 0:
                concordant += 1
            else:
                discordant += 1
    denominator = math.sqrt(
        (concordant + discordant + ties_left)
        * (concordant + discordant + ties_right)
    )
    if denominator == 0.0:
        return None
    return (concordant - discordant) / denominator


def midrank_percentile(null_values, x):
    values = list(null_values)
    if not values:
        raise ValueError("null_values must not be empty")
    below = sum(value < x for value in values)
    equal = sum(value == x for value in values)
    return 100.0 * (below + 0.5 * equal) / len(values)


def _row_contains(row, A, B, terms):
    return any(
        _contains_input(value, A, B) or any(term in _normalised_text(value) for term in terms)
        for value in row
        if value is not None
    )


def _selected_entry(name, index, rows, common_metrics, A, B, terms, plain=False):
    if plain:
        return {
            "corpus_index": index,
            "line": rows[index][2],
            "metrics": _public_metrics(common_metrics[index]),
            "contains_input": _row_contains(rows[index], A, B, terms),
        }
    entry = {
        "corpus_index": index,
        "a": rows[index][0],
        "b": rows[index][1],
        "metrics": _public_metrics(common_metrics[index]),
    }
    if name in BASELINE_METHODS:
        entry["contains_input"] = _row_contains(rows[index], A, B, terms)
    return entry


def _plain_baselines(space, limit):
    direction = normalise(space["a_vector"] + space["b_vector"])
    scores = [float(np.dot(normalise(vector), direction)) for vector in space["passage_vectors"]]
    similar = sorted(range(len(scores)), key=lambda index: (-scores[index], index))[:limit]
    return {
        "similar": [{"index": index} for index in similar],
    }


def _canonical_corpus(rows):
    return [line if a is None else (a, b, line) for a, b, line in rows]


def _rank_indices(metrics, score, eligible, limit):
    return sorted(eligible, key=lambda index: (-metrics[index][score], index))[:limit]


def _round_optional(value):
    return None if value is None else round(float(value), 6)


def _null_evaluation(null, p_min, common_metrics, selected_indices, eligible, samples, seed):
    quantiles = {
        source: {
            "50": round(float(np.percentile(null[source]["p"], 50)), 6),
            "95": round(float(np.percentile(null[source]["p"], 95)), 6),
        }
        for source in ("random", "shuffled")
    }
    selected_percentiles = {}
    per_method = {}
    for method, indices in selected_indices.items():
        entries = []
        for index in indices:
            metric = common_metrics[index]
            entries.append(
                {
                    "corpus_index": index,
                    "emergence_ratio": round(float(metric["emergence_ratio"]), 6),
                    "p": round(float(metric["plane_projection"]), 6),
                    "emergence_ratio_percentile": {
                        source: round(midrank_percentile(null[source]["emergence_ratio"], metric["emergence_ratio"]), 6)
                        for source in ("random", "shuffled")
                    },
                    "p_percentile": {
                        source: round(midrank_percentile(null[source]["p"], metric["plane_projection"]), 6)
                        for source in ("random", "shuffled")
                    },
                }
            )
        selected_percentiles[method] = entries
        per_method[method] = {
            key: {
                source: round(float(np.mean([entry[key][source] for entry in entries])), 6)
                if entries else 0.0
                for source in ("random", "shuffled")
            }
            for key in ("emergence_ratio_percentile", "p_percentile")
        }
    eligible_metrics = [common_metrics[index] for index in eligible]
    emergence = [metric["emergence_ratio"] for metric in eligible_metrics]
    return {
        "samples": samples,
        "seed": seed,
        "p_quantiles": quantiles,
        "p_min": round(float(p_min), 6),
        "per_method": per_method,
        "selected_percentiles": selected_percentiles,
        "eligible_summary": {
            "emergence_ratio_max": round(float(max(emergence)), 6) if emergence else 0.0,
            "emergence_ratio_median": round(float(np.median(emergence)), 6) if emergence else 0.0,
            "p_below_min_fraction": round(float(np.mean([metric["p_below_min"] for metric in eligible_metrics])), 6) if eligible_metrics else 0.0,
            "between_AB_fraction": round(float(np.mean([metric["between_AB"] for metric in eligible_metrics])), 6) if eligible_metrics else 0.0,
        },
    }


def _compare(
    A,
    B,
    corpus,
    encoder,
    limit=5,
    seed=0,
    corpus_path=None,
    *,
    input_kind="word",
    exclude=(),
    p_min=None,
    null_samples=1000,
    extra_query_texts=(),
    excluded=None,
    space=None,
):
    """Run single-metric and third-concept methods in one shared representation."""
    rows = _rows(corpus)
    if not rows:
        raise ValueError("corpus must not be empty")
    requested_limit = int(limit)
    if not 1 <= requested_limit <= 20:
        raise ValueError("limit must be between 1 and 20 because engine.run() clips to that range")
    if input_kind not in {"word", "sentence"}:
        raise ValueError("input_kind must be 'word' or 'sentence'")
    if type(null_samples) is not int or null_samples < 1:
        raise ValueError("null_samples must be an integer at least 1")
    terms = _normalised_exclude_terms(exclude)
    used_a = _normalise_input(A)
    used_b = _normalise_input(B)
    if not used_a or not used_b:
        raise ValueError("A and B must contain non-whitespace text")
    geometry_a = A.strip() if input_kind == "sentence" else used_a
    geometry_b = B.strip() if input_kind == "sentence" else used_b
    plain = any(a is None or b is None for a, b, _ in rows)
    method_names = PLAIN_METHOD_NAMES if plain else METHOD_NAMES

    canonical_corpus = _canonical_corpus(rows)
    if space is None:
        space = shared_geometry_space(
            geometry_a, geometry_b, canonical_corpus, encoder,
            vocab=list(extra_query_texts),
        )
    if excluded is None:
        excluded_mask = exclusion_mask(space, A, B, exclude)
    else:
        excluded_mask = list(excluded)
        if len(excluded_mask) != len(rows):
            raise ValueError("excluded mask length must match corpus rows")
    null = null_geometry(
        space["a_vector"], space["b_vector"], space["passage_vectors"], space["mu"],
        n=null_samples, seed=int(seed),
    )
    if p_min is None:
        used_p_min = auto_p_min(null)
        p_min_source = "auto:shuffled-p95"
    else:
        used_p_min = _validated_p_min(p_min)
        p_min_source = "user"

    ranked = {}
    if plain:
        ranked.update(_plain_baselines(space, requested_limit))
    else:
        context = f"{used_a}、{used_b}"
        for name in BASELINE_METHODS:
            ranked[name] = run(context, rows, name, limit=requested_limit, encoder=encoder)["selected"]
    for name in THIRD_CONCEPT_METHODS:
        ranked[name] = rank_entries(
            geometry_a, geometry_b, rows, encoder, name, requested_limit, space=space,
            exclusion_inputs=(A, B), exclude=exclude, excluded=excluded_mask, p_min=used_p_min,
        )
    ranked_ratio = rank_entries(
        geometry_a, geometry_b, rows, encoder, RATIO_METHOD, requested_limit, space=space,
        exclusion_inputs=(A, B), exclude=exclude, excluded=excluded_mask, p_min=used_p_min,
    )
    common_metrics = geometry(
        space["a_vector"], space["b_vector"], space["passage_vectors"],
        mu=space["mu"], p_min=used_p_min,
    )

    methods = {}
    selected_indices = {}
    for name in method_names:
        indices = [int(item["index"]) for item in ranked[name]]
        selected_indices[name] = indices
        methods[name] = {
            "selected": [_selected_entry(name, index, rows, common_metrics, A, B, terms, plain) for index in indices],
            "means": _mean_metrics(indices, common_metrics),
        }
    ratio_indices = [int(item["index"]) for item in ranked_ratio]
    selected_indices[RATIO_METHOD] = ratio_indices
    ratio_method = {
        "selected": [_selected_entry(RATIO_METHOD, index, rows, common_metrics, A, B, terms, plain) for index in ratio_indices],
        "means": _mean_metrics(ratio_indices, common_metrics),
    }
    evaluation_names = (*method_names, RATIO_METHOD)
    selected_sets = {name: set(selected_indices[name]) for name in evaluation_names}
    evaluation_jaccard = {
        left: {right: _jaccard(selected_sets[left], selected_sets[right]) for right in evaluation_names}
        for left in evaluation_names
    }
    legacy_jaccard = {
        left: {right: evaluation_jaccard[left][right] for right in method_names}
        for left in method_names
    }
    eligible = [index for index, value in enumerate(excluded_mask) if not value]
    balance_top = _rank_indices(common_metrics, "balance", eligible, requested_limit)
    no_wedge_values = [common_metrics[index]["score_no_wedge"] for index in eligible]
    balance_values = [common_metrics[index]["balance"] for index in eligible]
    wedge_ablation = {}
    for name, score in (("third-concept", "score_third"), (RATIO_METHOD, "score_ratio")):
        score_values = [common_metrics[index][score] for index in eligible]
        score_top = _rank_indices(common_metrics, score, eligible, requested_limit)
        wedge_ablation[name] = {
            "kendall_tau_b_vs_balance": _round_optional(kendall_tau_b(score_values, balance_values)),
            "kendall_tau_b_vs_no_wedge": _round_optional(kendall_tau_b(score_values, no_wedge_values)),
            "topk_jaccard_vs_balance": _jaccard(set(score_top), set(balance_top)),
            "topk_identical_to_balance": score_top == balance_top,
            "eligible_count": len(eligible),
        }
    evaluation_notes = list(EVALUATION_NOTES)
    if plain:
        evaluation_notes.append("orthogonal and wedge baselines are omitted because engine.run requires pair-labelled corpus rows.")
    input_normalization = {
        "rule": r"[、,，/／\s]+ -> _",
        "A": {"original": A, "used": used_a},
        "B": {"original": B, "used": used_b},
    }
    if input_kind == "sentence" or plain:
        input_normalization["applies_to"] = "engine-baselines-only" if input_kind == "sentence" else "all-methods"
    result = {
        "input": {"A": A, "B": B},
        "input_normalization": input_normalization,
        "encoder": {"name": encoder.name, "model_id": encoder.model_id, "prefix": encoder.prefix},
        "corpus_id": corpus_id(rows, corpus_path),
        "corpus_size": len(rows),
        "params": {"limit": requested_limit, "seed": int(seed)},
        "methods": methods,
        "jaccard": legacy_jaccard,
        "notes": list(NOTES),
        "disclaimer": DISCLAIMER,
        "evaluation": {
            "params": {
                "input_kind": input_kind,
                "exclude": list(exclude),
                "p_min": round(float(used_p_min), 6),
                "p_min_source": p_min_source,
                "null_samples": null_samples,
                "null_seed": int(seed),
            },
            "baseline_source": "shared-space" if plain else "engine.run",
            "methods": {RATIO_METHOD: ratio_method},
            "ratio_means": {name: _ratio_means(selected_indices[name], common_metrics) for name in evaluation_names},
            "jaccard": evaluation_jaccard,
            "wedge_ablation": wedge_ablation,
            "null": _null_evaluation(null, used_p_min, common_metrics, selected_indices, eligible, null_samples, int(seed)),
            "excluded_count": len(rows) - len(eligible),
            "eligible_count": len(eligible),
            "notes": evaluation_notes,
        },
    }
    return result, space, common_metrics


def compare_methods(
    A,
    B,
    corpus,
    encoder,
    limit=5,
    seed=0,
    corpus_path=None,
    *,
    input_kind="word",
    exclude=(),
    p_min=None,
    null_samples=1000,
    extra_query_texts=(),
    excluded=None,
):
    """Run single-metric and third-concept methods in one shared representation."""
    result, _, _ = _compare(
        A, B, corpus, encoder, limit, seed, corpus_path,
        input_kind=input_kind, exclude=exclude, p_min=p_min,
        null_samples=null_samples, extra_query_texts=extra_query_texts,
        excluded=excluded,
    )
    return result


def _result_indices(result, method):
    source = result["evaluation"]["methods"][RATIO_METHOD] if method == RATIO_METHOD else result["methods"][method]
    return [item["corpus_index"] for item in source["selected"]]


def compare_pair(word_pair, sentence_pair, corpus, encoder, limit, seed, corpus_path, exclude, p_min, null_samples):
    word_a, word_b = word_pair
    sentence_a, sentence_b = sentence_pair
    rows = _rows(corpus)
    word_q = (_normalise_input(word_a), _normalise_input(word_b))
    sent_q = (sentence_a.strip(), sentence_b.strip())
    if not all((*word_q, *sent_q)):
        raise ValueError("A and B must contain non-whitespace text")
    canonical_corpus = _canonical_corpus(rows)
    word_space = shared_geometry_space(*word_q, canonical_corpus, encoder, vocab=list(sent_q))
    sentence_space = shared_geometry_space(*sent_q, canonical_corpus, encoder, vocab=list(word_q))
    word_mask = exclusion_mask(word_space, word_a, word_b, exclude)
    sentence_mask = exclusion_mask(sentence_space, sentence_a, sentence_b, exclude)
    common_excluded = [left or right for left, right in zip(word_mask, sentence_mask)]
    word, _, word_metrics = _compare(
        word_a, word_b, canonical_corpus, encoder, limit, seed, corpus_path, input_kind="word", exclude=exclude,
        p_min=p_min, null_samples=null_samples, excluded=common_excluded, space=word_space,
    )
    sentence, _, sentence_metrics = _compare(
        sentence_a, sentence_b, canonical_corpus, encoder, limit, seed, corpus_path, input_kind="sentence", exclude=exclude,
        p_min=p_min, null_samples=null_samples, excluded=common_excluded, space=sentence_space,
    )
    method_names = (*word["methods"], RATIO_METHOD)
    eligible = [index for index, value in enumerate(common_excluded) if not value]
    family_scores = {
        "third-concept": "score_third",
        RATIO_METHOD: "score_ratio",
        "mean": "s_mean",
        "third-concept-no-wedge": "score_no_wedge",
    }
    per_method = {}
    for method in method_names:
        entry = {
            "topk_jaccard": _jaccard(set(_result_indices(word, method)), set(_result_indices(sentence, method)))
        }
        if method in family_scores:
            score = family_scores[method]
            tau = kendall_tau_b(
                [word_metrics[index][score] for index in eligible],
                [sentence_metrics[index][score] for index in eligible],
            )
            entry["kendall_tau_b"] = _round_optional(tau)
        else:
            entry["kendall_tau_b"] = None
        per_method[method] = entry
    return {
        "mode": "word-vs-sentence",
        "word": word,
        "sentence": sentence,
        "paired": {
            "shared_conditions": {
                "encoder_model_id": encoder.model_id,
                "corpus_id": word["corpus_id"],
                "limit": int(limit),
                "seed": int(seed),
                "exclude": list(exclude),
                "null_samples": null_samples,
                "mu_query_texts": [*word_q, *sent_q],
                "p_min": {
                    "word": word["evaluation"]["params"]["p_min"],
                    "sentence": sentence["evaluation"]["params"]["p_min"],
                },
                "eligible_count": len(eligible),
            },
            "per_method": per_method,
        },
    }


def _calibration_markdown_lines(calibration):
    if not calibration:
        return []
    target = calibration["target"]
    lines = ["", "## Calibration", "",
             f"- Controls: {target['control_count']}",
             f"- Valid rate: {target['valid_rate']:.4f} ({target.get('valid_count', 'n/a')}/{target.get('eligible_count', 'n/a')})",
             f"- Pair percentile: {target['pair_percentile']:.1f}; descending rank: {target['rank_descending']}",
             f"- Token joint-supported rate: {target['token_joint_supported_rate']:.4f} ({target['token_joint_supported_count']}/{target['eligible_count']})",
             "",
             "| Pair | Score mean | A support | B support | Between support | Plane support | Joint-supported rate |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for label, item in [("target", target), *((" × ".join(control["pair"]), control) for control in calibration["controls"])]:
        lines.append(
            f"| {label} | {item['token_score_mean']:.4f} | {item['token_a_side_support_mean']:.4f} | "
            f"{item['token_b_side_support_mean']:.4f} | {item['token_between_direction_support_mean']:.4f} | "
            f"{item['token_plane_support_mean']:.4f} | {item['token_joint_supported_rate']:.4f} |"
        )
    lines.extend(["", f"- Interpretation: {calibration['limitations']['interpretation']}"])
    return lines


def render_markdown(result):
    if result.get("mode") == "word-vs-sentence":
        rendered = "\n\n".join(("# Word comparison\n\n" + render_markdown(result["word"]), "# Sentence comparison\n\n" + render_markdown(result["sentence"])))
        if result.get("corpus_source"):
            rendered = f"Corpus source: `{result['corpus_source']}`\n\n" + rendered
        return rendered + "\n".join(_calibration_markdown_lines(result.get("calibration")))
    method_names = tuple(result["methods"])
    lines = [
        "# Third-concept comparison",
        "",
        f"- Input: A = `{result['input']['A']}`, B = `{result['input']['B']}`",
        f"- Encoder: `{result['encoder']['name']}` (`{result['encoder']['model_id']}`)",
        f"- Corpus: `{result['corpus_id']}` ({result['corpus_size']} entries)",
        f"- Limit: {result['params']['limit']}; seed: {result['params']['seed']}",
        f"- p_min: {result['evaluation']['params']['p_min']:.6f}",
        "",
        "## Mean comparison metrics",
        "",
        "| Method | s_A | s_B | Plane projection | Residual | Emergence norm | Score third |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    if result.get("corpus_source"):
        lines.insert(4, f"- Corpus source: `{result['corpus_source']}`")
    for name in method_names:
        means = result["methods"][name]["means"]
        values = " | ".join(f"{means[metric]:.6f}" for metric in METRIC_NAMES)
        lines.append(f"| {name} | {values} |")
    lines.extend(["", "## Top entries", ""])
    for name in method_names:
        lines.extend([f"### {name}", ""])
        for number, item in enumerate(result["methods"][name]["selected"], 1):
            label = item["line"] if "line" in item else f"{item['a']} × {item['b']}"
            lines.append(f"{number}. {label}")
        lines.append("")
    lines.extend(["## Jaccard overlap", ""])
    lines.append("| Method | " + " | ".join(method_names) + " |")
    lines.append("|---|" + "---:|" * len(method_names))
    for left in method_names:
        values = " | ".join(f"{result['jaccard'][left][right]:.6f}" for right in method_names)
        lines.append(f"| {left} | {values} |")
    lines.extend(["", "## Evaluation summary", ""])
    for name, item in result["evaluation"]["wedge_ablation"].items():
        lines.append(f"- {name}: tau vs balance = {item['kendall_tau_b_vs_balance']}; tau vs no-wedge = {item['kendall_tau_b_vs_no_wedge']}")
    lines.extend(_calibration_markdown_lines(result.get("calibration")))
    lines.extend(["", "## Notes", ""])
    lines.extend(f"- {note}" for note in result["notes"])
    lines.extend(["", result["disclaimer"], ""])
    return "\n".join(lines)


def write_blind_materials(result, corpus, sheet_dir):
    """Write a method-blind human evaluation sheet and its private key."""
    rows = _rows(corpus)
    methods_by_index = {}
    for name in result["methods"]:
        for selected in result["methods"][name]["selected"]:
            methods_by_index.setdefault(selected["corpus_index"], []).append(name)
    indices = list(methods_by_index)
    random.Random(result["params"]["seed"]).shuffle(indices)
    output_dir = Path(sheet_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sheet_path = output_dir / "blind_sheet.md"
    key_path = output_dir / "blind_key.json"
    sheet_lines = [
        "# Blind human evaluation sheet", "", f"- Input A: `{result['input']['A']}`",
        f"- Input B: `{result['input']['B']}`",
    ]
    if result.get("corpus_source"):
        sheet_lines.append(f"- Corpus source: `{result['corpus_source']}`")
    sheet_lines.extend([
        "", "人手評価は未実施です。各観点を 1–5 で記入してください。", "",
        "| ID | A | B | Entry text | Unexpectedness (1–5) | Coherence (1–5) | Emergence (1–5) | Testability (1–5) | Comment |",
        "|---|---|---|---|---|---|---|---|---|",
    ])
    key_entries = {}
    width = max(2, len(str(len(indices))))
    for number, index in enumerate(indices, 1):
        blind_id = f"E{number:0{width}d}"
        a, b, line = rows[index]
        safe_values = [(value or "").replace("|", "\\|").replace("\n", " ") for value in (a, b, line)]
        sheet_lines.append(f"| {blind_id} | {safe_values[0]} | {safe_values[1]} | {safe_values[2]} |  |  |  |  |  |")
        key_entries[blind_id] = {"corpus_index": index, "methods": methods_by_index[index]}
    sheet_path.write_text("\n".join(sheet_lines) + "\n", encoding="utf-8")
    key = {"seed": result["params"]["seed"], "corpus_id": result["corpus_id"], "entries": key_entries}
    if "corpus_source" in result:
        key["corpus_source"] = result["corpus_source"]
    key_path.write_text(json.dumps(key, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"sheet": str(sheet_path), "key": str(key_path)}


def _strip_corpus_text(value):
    if isinstance(value, dict):
        return {
            key: _strip_corpus_text(item)
            for key, item in value.items()
            if not ("corpus_index" in value and key in {"a", "b", "line"})
        }
    if isinstance(value, list):
        return [_strip_corpus_text(item) for item in value]
    return value


def positive_null_samples(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("null samples must be at least 1")
    return parsed


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Compare CrossRAG retrieval methods", allow_abbrev=False)
    parser.add_argument("--A")
    parser.add_argument("--B")
    parser.add_argument("--A-sentence")
    parser.add_argument("--B-sentence")
    parser.add_argument("--corpus")
    parser.add_argument("--demo-corpus", action="store_true", help="Explicitly use the bundled demo corpus")
    parser.add_argument("--llm-pool", help="JSON array of concepts generated from host LLM knowledge without comparison context")
    parser.add_argument("--llm-pool-model", help="Optional metadata naming the host LLM that generated the pool")
    parser.add_argument("--encoder", choices=("hash", "semantic"), default="hash")
    parser.add_argument("--model", default="intfloat/multilingual-e5-small")
    parser.add_argument("--prefix-policy", choices=("auto", "e5", "prefix-free"), default="auto")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--input-kind", choices=("word", "sentence"), default="word")
    parser.add_argument("--exclude", action="append", default=[])
    parser.add_argument("--p-min", type=float)
    parser.add_argument("--null-samples", type=positive_null_samples, default=1000)
    parser.add_argument("--control-pair", action="append", default=[], help="control pair as LEFT|RIGHT; repeatable")
    parser.add_argument("--control-words", help="JSON string array used to sample control pairs")
    parser.add_argument("--control-samples", type=positive_null_samples, default=20)
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--format", choices=("json", "markdown"), default="json")
    parser.add_argument("--sheet-dir")
    argv = sys.argv[1:]
    args = parser.parse_args()
    if args.llm_pool is not None:
        if args.corpus is not None or args.demo_corpus:
            parser.error("--llm-pool cannot be combined with --corpus or --demo-corpus")
        unsupported = (
            "--A-sentence", "--B-sentence", "--encoder", "--model", "--prefix-policy", "--input-kind",
            "--exclude", "--p-min", "--null-samples", "--control-pair", "--control-words",
            "--control-samples", "--aggregate-only", "--sheet-dir",
        )
        if explicitly_supplied(argv, *unsupported) or args.A is not None or args.B is not None:
            parser.error("--llm-pool is a separate context-free draw and cannot be combined with comparison inputs or options")
        pool_seed = args.seed if explicitly_supplied(argv, "--seed") else None
        try:
            result = draw_llm_pool(load_llm_pool(args.llm_pool), args.limit, pool_seed, args.llm_pool_model)
        except (ValueError, RuntimeError, ImportError, OSError, json.JSONDecodeError) as exc:
            parser.error(str(exc))
        if args.format == "json":
            print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        else:
            lines = [
                "# LLM knowledge pool draw",
                "",
                f"- mode: `{result['mode']}`",
                f"- corpus_source: `{result['corpus_source']}`",
                f"- knowledge_basis: `{result['knowledge_basis']}`",
                f"- ranking_metric: `{result['ranking_metric']}`",
                f"- encoder: `{result['encoder']}`",
                f"- seed: `{result['seed']}`",
                f"- generation_model: `{result['generation_model'] if result['generation_model'] is not None else 'null'}`",
                f"- total: {result['total']}",
                "",
                "## Selected concepts",
                "",
                "| rank_value | index | concept | ranking_metric |",
                "| ---: | ---: | --- | --- |",
            ]
            for row in result["selected"]:
                concept = " ".join(row["concept"].splitlines()).replace("|", "\\|")
                lines.append(
                    f"| {row['rank_value']} | {row['index']} | {concept} | {row['ranking_metric']} |"
                )
            print("\n".join(lines))
        return
    if args.llm_pool_model is not None:
        parser.error("--llm-pool-model requires --llm-pool")
    if args.A is None or args.B is None:
        parser.error("the following arguments are required: --A and --B")
    if args.corpus is not None and args.demo_corpus:
        parser.error("specify only one of --corpus or --demo-corpus")
    if args.corpus is None and not args.demo_corpus:
        parser.error("one of --corpus or --demo-corpus is required; implicit demo corpus is disabled")
    if (args.A_sentence is None) != (args.B_sentence is None):
        parser.error("--A-sentence and --B-sentence must be supplied together")
    if args.A_sentence is not None and args.sheet_dir:
        parser.error("--sheet-dir cannot be used with --A-sentence/--B-sentence")
    if args.aggregate_only and args.format == "markdown":
        parser.error("--aggregate-only is available only with --format json")
    if args.aggregate_only and args.sheet_dir:
        parser.error("--aggregate-only cannot be used with --sheet-dir")
    try:
        rows = read_corpus(args.corpus) if args.corpus is not None else DEFAULT_CORPUS
        encoder = make_encoder(args.encoder, args.model, prefix_policy=args.prefix_policy)
        if args.A_sentence is not None:
            result = compare_pair(
                (args.A, args.B), (args.A_sentence, args.B_sentence), rows, encoder, args.limit,
                args.seed, args.corpus, args.exclude, args.p_min, args.null_samples,
            )
            result["corpus_source"] = "explicit" if args.corpus is not None else "demo"
        else:
            result = compare_methods(
                args.A, args.B, rows, encoder, limit=args.limit, seed=args.seed, corpus_path=args.corpus,
                input_kind=args.input_kind, exclude=args.exclude, p_min=args.p_min, null_samples=args.null_samples,
            )
            result["corpus_source"] = "explicit" if args.corpus is not None else "demo"
            if args.sheet_dir:
                write_blind_materials(result, rows, args.sheet_dir)
        if args.control_pair or args.control_words:
            pairs = []
            for value in args.control_pair:
                if value.count("|") != 1:
                    parser.error("--control-pair must use LEFT|RIGHT")
                pairs.append(tuple(part.strip() for part in value.split("|", 1)))
            words = json.loads(Path(args.control_words).read_text(encoding="utf-8")) if args.control_words else None
            if words is not None and not isinstance(words, list):
                raise ValueError("control-words file must be a JSON string array")
            result["calibration"] = calibrate_control_pairs(
                args.A, args.B, rows, encoder, control_pairs=pairs or None, control_words=words,
                samples=args.control_samples, seed=args.seed, limit=args.limit, input_kind=args.input_kind,
                exclude=args.exclude, null_samples=args.null_samples, p_min=args.p_min,
            )
        if args.aggregate_only:
            result = _strip_corpus_text(result)
            result["aggregate_only"] = True
    except (ValueError, RuntimeError, ImportError, OSError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    if args.format == "json":
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    else:
        print(render_markdown(result))


if __name__ == "__main__":
    main()
