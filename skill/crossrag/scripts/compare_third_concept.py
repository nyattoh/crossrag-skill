#!/usr/bin/env python3
"""Compare legacy CrossRAG retrieval with third-concept rankings."""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

import numpy as np

from engine import DEFAULT_CORPUS, run
from run import load_corpus
from third_concept import _contains_input, _rows, corpus_id, geometry, make_encoder, rank_entries, shared_geometry_space


METHOD_NAMES = (
    "similar",
    "orthogonal",
    "wedge",
    "random",
    "third-concept",
    "mean",
    "third-concept-no-wedge",
)
METRIC_NAMES = (
    "s_A",
    "s_B",
    "plane_projection",
    "residual",
    "emergence_norm",
    "score_third",
)
NOTES = [
    'similar / orthogonal / wedge / random は engine.run() を通り、その encode() は全テキストに "query: " を適用する。',
    "third-concept は score_third を目的関数として選ぶため、score_third の平均で高くなることは循環的であり、各平均は記述値にすぎず、優越性の証拠ではない。",
    "third-concept がベースラインと異ならない結果も、そのまま報告する。",
    "Baselines are not exclusion-filtered; baseline の contains_input は入力概念との文字列一致・包含を示す。",
    "seed は評価シートのシャッフルにだけ使われ、engine.run() の random baseline は context とコーパス対の hash で決まる。",
]
DISCLAIMER = "全指標は探索用ヒューリスティックであり、人手評価は未実施。方式間の品質差や優越性を実証しない。"


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


def _jaccard(left, right):
    union = left | right
    return round(len(left & right) / len(union), 6) if union else 1.0


def _selected_entry(name, index, rows, common_metrics, A, B):
    entry = {
        "corpus_index": index,
        "a": rows[index][0],
        "b": rows[index][1],
        "metrics": _public_metrics(common_metrics[index]),
    }
    if name in METHOD_NAMES[:4]:
        entry["contains_input"] = any(_contains_input(value, A, B) for value in rows[index])
    return entry


def compare_methods(A, B, corpus, encoder, limit=5, seed=0, corpus_path=None):
    """Run all seven methods and recompute their metrics in one representation."""
    rows = _rows(corpus)
    if not rows:
        raise ValueError("corpus must not be empty")
    requested_limit = int(limit)
    if not 1 <= requested_limit <= 20:
        raise ValueError("limit must be between 1 and 20 because engine.run() clips to that range")
    used_a = _normalise_input(A)
    used_b = _normalise_input(B)
    if not used_a or not used_b:
        raise ValueError("A and B must contain non-whitespace text")

    context = f"{used_a}、{used_b}"
    space = shared_geometry_space(used_a, used_b, rows, encoder)
    ranked = {}
    for name in METHOD_NAMES[:4]:
        result = run(context, rows, name, limit=requested_limit, encoder=encoder)
        ranked[name] = result["selected"]
    for name in METHOD_NAMES[4:]:
        ranked[name] = rank_entries(
            used_a,
            used_b,
            rows,
            encoder,
            name,
            requested_limit,
            space=space,
            exclusion_inputs=(A, B),
        )

    common_metrics = geometry(
        space["a_vector"], space["b_vector"], space["passage_vectors"], mu=space["mu"]
    )

    methods = {}
    selected_sets = {}
    for name in METHOD_NAMES:
        indices = [int(item["index"]) for item in ranked[name]]
        selected_sets[name] = set(indices)
        methods[name] = {
            "selected": [
                _selected_entry(name, index, rows, common_metrics, A, B)
                for index in indices
            ],
            "means": _mean_metrics(indices, common_metrics),
        }

    jaccard = {
        left: {right: _jaccard(selected_sets[left], selected_sets[right]) for right in METHOD_NAMES}
        for left in METHOD_NAMES
    }
    return {
        "input": {"A": A, "B": B},
        "input_normalization": {
            "rule": r"[、,，/／\s]+ -> _",
            "A": {"original": A, "used": used_a},
            "B": {"original": B, "used": used_b},
        },
        "encoder": {"name": encoder.name, "model_id": encoder.model_id, "prefix": encoder.prefix},
        "corpus_id": corpus_id(rows, corpus_path),
        "corpus_size": len(rows),
        "params": {"limit": requested_limit, "seed": int(seed)},
        "methods": methods,
        "jaccard": jaccard,
        "notes": list(NOTES),
        "disclaimer": DISCLAIMER,
    }


def render_markdown(result):
    lines = [
        "# Third-concept comparison",
        "",
        f"- Input: A = `{result['input']['A']}`, B = `{result['input']['B']}`",
        f"- Encoder: `{result['encoder']['name']}` (`{result['encoder']['model_id']}`)",
        f"- Corpus: `{result['corpus_id']}` ({result['corpus_size']} entries)",
        f"- Limit: {result['params']['limit']}; seed: {result['params']['seed']}",
        "",
        "## Mean comparison metrics",
        "",
        "| Method | s_A | s_B | Plane projection | Residual | Emergence norm | Score third |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in METHOD_NAMES:
        means = result["methods"][name]["means"]
        values = " | ".join(f"{means[metric]:.6f}" for metric in METRIC_NAMES)
        lines.append(f"| {name} | {values} |")

    lines.extend(["", "## Top entries", ""])
    for name in METHOD_NAMES:
        lines.extend([f"### {name}", ""])
        for number, item in enumerate(result["methods"][name]["selected"], 1):
            lines.append(f"{number}. {item['a']} × {item['b']}")
        lines.append("")

    lines.extend(["## Jaccard overlap", ""])
    lines.append("| Method | " + " | ".join(METHOD_NAMES) + " |")
    lines.append("|---|" + "---:|" * len(METHOD_NAMES))
    for left in METHOD_NAMES:
        values = " | ".join(f"{result['jaccard'][left][right]:.6f}" for right in METHOD_NAMES)
        lines.append(f"| {left} | {values} |")
    lines.extend(["", "## Notes", ""])
    lines.extend(f"- {note}" for note in result["notes"])
    lines.extend(["", result["disclaimer"], ""])
    return "\n".join(lines)


def write_blind_materials(result, corpus, sheet_dir):
    """Write a method-blind human evaluation sheet and its private key."""
    rows = _rows(corpus)
    methods_by_index = {}
    for name in METHOD_NAMES:
        for selected in result["methods"][name]["selected"]:
            methods_by_index.setdefault(selected["corpus_index"], []).append(name)
    indices = list(methods_by_index)
    random.Random(result["params"]["seed"]).shuffle(indices)

    output_dir = Path(sheet_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sheet_path = output_dir / "blind_sheet.md"
    key_path = output_dir / "blind_key.json"
    sheet_lines = [
        "# Blind human evaluation sheet",
        "",
        f"- Input A: `{result['input']['A']}`",
        f"- Input B: `{result['input']['B']}`",
        "",
        "人手評価は未実施です。各観点を 1–5 で記入してください。",
        "",
        "| ID | A | B | Entry text | Unexpectedness (1–5) | Coherence (1–5) | Emergence (1–5) | Testability (1–5) | Comment |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    key_entries = {}
    width = max(2, len(str(len(indices))))
    for number, index in enumerate(indices, 1):
        blind_id = f"E{number:0{width}d}"
        a, b, line = rows[index]
        safe_values = [value.replace("|", "\\|").replace("\n", " ") for value in (a, b, line)]
        sheet_lines.append(
            f"| {blind_id} | {safe_values[0]} | {safe_values[1]} | {safe_values[2]} |  |  |  |  |  |"
        )
        key_entries[blind_id] = {"corpus_index": index, "methods": methods_by_index[index]}
    sheet_path.write_text("\n".join(sheet_lines) + "\n", encoding="utf-8")
    key_path.write_text(
        json.dumps(
            {
                "seed": result["params"]["seed"],
                "corpus_id": result["corpus_id"],
                "entries": key_entries,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {"sheet": str(sheet_path), "key": str(key_path)}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Compare CrossRAG retrieval methods")
    parser.add_argument("--A", required=True)
    parser.add_argument("--B", required=True)
    parser.add_argument("--corpus")
    parser.add_argument("--encoder", choices=("hash", "semantic"), default="hash")
    parser.add_argument("--model", default="intfloat/multilingual-e5-small")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--format", choices=("json", "markdown"), default="json")
    parser.add_argument("--sheet-dir")
    args = parser.parse_args()
    try:
        rows = load_corpus(args.corpus) if args.corpus else DEFAULT_CORPUS
        encoder = make_encoder(args.encoder, args.model)
        result = compare_methods(
            args.A,
            args.B,
            rows,
            encoder,
            limit=args.limit,
            seed=args.seed,
            corpus_path=args.corpus,
        )
        if args.sheet_dir:
            write_blind_materials(result, rows, args.sheet_dir)
    except (ValueError, RuntimeError, ImportError, OSError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    if args.format == "json":
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(render_markdown(result))


if __name__ == "__main__":
    main()
