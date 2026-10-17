#!/usr/bin/env python3
"""Compare legacy CrossRAG retrieval with third-concept rankings."""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np

from engine import DEFAULT_CORPUS, normalise, run
from third_concept import (
    _contains_input,
    _rows,
    auto_p_min,
    corpus_id,
    exclusion_mask,
    geometry,
    make_encoder,
    null_geometry,
    rank_entries,
    shared_geometry_space,
)
from third_concept_cli import read_corpus


METHOD_NAMES = (
    "similar",
    "orthogonal",
    "wedge",
    "random",
    "third-concept",
    "mean",
    "third-concept-no-wedge",
)
PLAIN_METHOD_NAMES = (
    "similar",
    "random",
    "third-concept",
    "mean",
    "third-concept-no-wedge",
)
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
    'similar / orthogonal / wedge / random は engine.run() を通り、その encode() は全テキストに "query: " を適用する。',
    "third-concept は score_third を目的関数として選ぶため、score_third の平均で高くなることは循環的であり、各平均は記述値にすぎず、優越性の証拠ではない。",
    "third-concept がベースラインと異ならない結果も、そのまま報告する。",
    "Baselines are not exclusion-filtered; baseline の contains_input は入力概念との文字列一致・包含を示す。",
    "seed は評価シートのシャッフルにだけ使われ、engine.run() の random baseline は context とコーパス対の hash で決まる。",
]
EVALUATION_NOTES = [
    "wedge ablation compares each third-concept score with balance alone and with third-concept-no-wedge over all eligible rows.",
    "null random uses seeded isotropic vectors; null shuffled uses seeded coordinate permutations of centred passage vectors.",
    "percentile uses the mid-rank definition: 100 * (count(null < x) + 0.5 * count(null == x)) / n.",
    "Results are reported even when they do not outperform baselines; objective-score means do not establish superiority.",
]
DISCLAIMER = "全指標は探索用ヒューリスティックであり、人手評価は未実施。方式間の品質差や優越性を実証しない。"


def _normalise_input(text):
    return re.sub(r"[、,，/／\s]+", "_", text.strip()).strip("_")


def _normalised_text(text):
    return "".join(unicodedata.normalize("NFKC", text).casefold().split())


def _exclude_terms(exclude):
    terms = []
    for term in exclude:
        if not isinstance(term, str) or not term.strip():
            raise ValueError("exclude terms must be non-empty strings")
        value = _normalised_text(term)
        if not value:
            raise ValueError("exclude terms must be non-empty strings")
        terms.append(value)
    return terms


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
    if name in METHOD_NAMES[:4]:
        entry["contains_input"] = _row_contains(rows[index], A, B, terms)
    return entry


def _validate_p_min(value):
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("p_min must be finite and between 0 and 1") from exc
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise ValueError("p_min must be finite and between 0 and 1")
    return parsed


def _plain_baselines(space, limit, seed):
    direction = normalise(space["a_vector"] + space["b_vector"])
    scores = [float(np.dot(normalise(vector), direction)) for vector in space["passage_vectors"]]
    similar = sorted(range(len(scores)), key=lambda index: (-scores[index], index))[:limit]
    random_indices = np.random.default_rng(seed).permutation(len(scores))[:limit].tolist()
    return {
        "similar": [{"index": index} for index in similar],
        "random": [{"index": int(index)} for index in random_indices],
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
    """Run legacy and third-concept methods in one shared representation."""
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
    terms = _exclude_terms(exclude)
    used_a = _normalise_input(A)
    used_b = _normalise_input(B)
    if not used_a or not used_b:
        raise ValueError("A and B must contain non-whitespace text")
    geometry_a = A.strip() if input_kind == "sentence" else used_a
    geometry_b = B.strip() if input_kind == "sentence" else used_b
    plain = any(a is None or b is None for a, b, _ in rows)
    method_names = PLAIN_METHOD_NAMES if plain else METHOD_NAMES

    canonical_corpus = _canonical_corpus(rows)
    space = shared_geometry_space(geometry_a, geometry_b, canonical_corpus, encoder, vocab=list(extra_query_texts))
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
        used_p_min = _validate_p_min(p_min)
        p_min_source = "user"

    ranked = {}
    if plain:
        ranked.update(_plain_baselines(space, requested_limit, int(seed)))
    else:
        context = f"{used_a}、{used_b}"
        for name in METHOD_NAMES[:4]:
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
    return {
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


def _result_indices(result, method):
    source = result["evaluation"]["methods"][RATIO_METHOD] if method == RATIO_METHOD else result["methods"][method]
    return [item["corpus_index"] for item in source["selected"]]


def compare_pair(word_pair, sentence_pair, corpus, encoder, limit, seed, corpus_path, exclude, p_min, null_samples):
    word_a, word_b = word_pair
    sentence_a, sentence_b = sentence_pair
    rows = _rows(corpus)
    word_used = (_normalise_input(word_a), _normalise_input(word_b))
    canonical_corpus = _canonical_corpus(rows)
    word_space = shared_geometry_space(*word_used, canonical_corpus, encoder, vocab=[sentence_a.strip(), sentence_b.strip()])
    sentence_space = shared_geometry_space(sentence_a.strip(), sentence_b.strip(), canonical_corpus, encoder, vocab=[*word_pair])
    word_mask = exclusion_mask(word_space, word_a, word_b, exclude)
    sentence_mask = exclusion_mask(sentence_space, sentence_a, sentence_b, exclude)
    common_excluded = [left or right for left, right in zip(word_mask, sentence_mask)]
    word = compare_methods(
        word_a, word_b, canonical_corpus, encoder, limit, seed, corpus_path, input_kind="word", exclude=exclude,
        p_min=p_min, null_samples=null_samples, extra_query_texts=sentence_pair, excluded=common_excluded,
    )
    sentence = compare_methods(
        sentence_a, sentence_b, canonical_corpus, encoder, limit, seed, corpus_path, input_kind="sentence", exclude=exclude,
        p_min=p_min, null_samples=null_samples, extra_query_texts=word_pair, excluded=common_excluded,
    )
    method_names = (*word["methods"], RATIO_METHOD)
    eligible = [index for index, value in enumerate(common_excluded) if not value]
    word_metrics = geometry(
        word_space["a_vector"], word_space["b_vector"], word_space["passage_vectors"],
        mu=word_space["mu"], p_min=word["evaluation"]["params"]["p_min"],
    )
    sentence_metrics = geometry(
        sentence_space["a_vector"], sentence_space["b_vector"], sentence_space["passage_vectors"],
        mu=sentence_space["mu"], p_min=sentence["evaluation"]["params"]["p_min"],
    )
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
                "mu_query_texts": [word_a, word_b, sentence_a, sentence_b],
                "p_min": {
                    "word": word["evaluation"]["params"]["p_min"],
                    "sentence": sentence["evaluation"]["params"]["p_min"],
                },
                "eligible_count": len(eligible),
            },
            "per_method": per_method,
        },
    }


def render_markdown(result):
    if result.get("mode") == "word-vs-sentence":
        return "\n\n".join(("# Word comparison\n\n" + render_markdown(result["word"]), "# Sentence comparison\n\n" + render_markdown(result["sentence"])))
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
        f"- Input B: `{result['input']['B']}`", "", "人手評価は未実施です。各観点を 1–5 で記入してください。", "",
        "| ID | A | B | Entry text | Unexpectedness (1–5) | Coherence (1–5) | Emergence (1–5) | Testability (1–5) | Comment |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    key_entries = {}
    width = max(2, len(str(len(indices))))
    for number, index in enumerate(indices, 1):
        blind_id = f"E{number:0{width}d}"
        a, b, line = rows[index]
        safe_values = [(value or "").replace("|", "\\|").replace("\n", " ") for value in (a, b, line)]
        sheet_lines.append(f"| {blind_id} | {safe_values[0]} | {safe_values[1]} | {safe_values[2]} |  |  |  |  |  |")
        key_entries[blind_id] = {"corpus_index": index, "methods": methods_by_index[index]}
    sheet_path.write_text("\n".join(sheet_lines) + "\n", encoding="utf-8")
    key_path.write_text(json.dumps({"seed": result["params"]["seed"], "corpus_id": result["corpus_id"], "entries": key_entries}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
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
    parser = argparse.ArgumentParser(description="Compare CrossRAG retrieval methods")
    parser.add_argument("--A", required=True)
    parser.add_argument("--B", required=True)
    parser.add_argument("--A-sentence")
    parser.add_argument("--B-sentence")
    parser.add_argument("--corpus")
    parser.add_argument("--encoder", choices=("hash", "semantic"), default="hash")
    parser.add_argument("--model", default="intfloat/multilingual-e5-small")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--input-kind", choices=("word", "sentence"), default="word")
    parser.add_argument("--exclude", action="append", default=[])
    parser.add_argument("--p-min", type=float)
    parser.add_argument("--null-samples", type=positive_null_samples, default=1000)
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--format", choices=("json", "markdown"), default="json")
    parser.add_argument("--sheet-dir")
    args = parser.parse_args()
    if (args.A_sentence is None) != (args.B_sentence is None):
        parser.error("--A-sentence and --B-sentence must be supplied together")
    if args.aggregate_only and args.format == "markdown":
        parser.error("--aggregate-only is available only with --format json")
    try:
        rows = read_corpus(args.corpus) if args.corpus else DEFAULT_CORPUS
        encoder = make_encoder(args.encoder, args.model)
        if args.A_sentence is not None:
            result = compare_pair(
                (args.A, args.B), (args.A_sentence, args.B_sentence), rows, encoder, args.limit,
                args.seed, args.corpus, args.exclude, args.p_min, args.null_samples,
            )
        else:
            result = compare_methods(
                args.A, args.B, rows, encoder, limit=args.limit, seed=args.seed, corpus_path=args.corpus,
                input_kind=args.input_kind, exclude=args.exclude, p_min=args.p_min, null_samples=args.null_samples,
            )
            if args.sheet_dir:
                write_blind_materials(result, rows, args.sheet_dir)
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
