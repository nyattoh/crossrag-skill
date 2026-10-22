"""Validation and seeded selection for host-generated knowledge pools."""
from __future__ import annotations

import json
import secrets
import unicodedata
from pathlib import Path

import numpy as np

MIN_UNIQUE_CONCEPTS = 10


def load_llm_pool(path):
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(data, list):
        raise ValueError("LLM pool must be a JSON array of strings")

    concepts = []
    seen = set()
    for concept in data:
        if not isinstance(concept, str) or not concept.strip():
            raise ValueError("each LLM pool item must be a non-empty string")
        concept = concept.strip()
        try:
            concept.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ValueError("each LLM pool item must contain valid UTF-8 Unicode") from exc
        key = " ".join(unicodedata.normalize("NFKC", concept).casefold().split())
        if key not in seen:
            seen.add(key)
            concepts.append(concept)

    if len(concepts) < MIN_UNIQUE_CONCEPTS:
        raise ValueError(f"LLM pool must contain at least {MIN_UNIQUE_CONCEPTS} unique concepts after deduplication")
    return concepts


def draw_llm_pool(concepts, limit=5, seed=None, generation_model=None):
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("LLM pool limit must be at least 1")
    if limit > len(concepts):
        raise ValueError("LLM pool limit cannot exceed the number of unique concepts")
    if generation_model is not None and not generation_model.strip():
        raise ValueError("--llm-pool-model must be non-empty")
    if generation_model is not None:
        generation_model = generation_model.strip()
    if seed is None:
        seed = secrets.randbits(53)

    selected = []
    for rank, index in enumerate(np.random.default_rng(seed).permutation(len(concepts))[:limit], start=1):
        selected.append({
            "index": int(index),
            "concept": concepts[int(index)],
            "rank_value": rank,
            "ranking_metric": "seeded_uniform_draw",
        })
    return {
        "mode": "llm-pool",
        "corpus_source": "llm_knowledge",
        "ranking_metric": "seeded_uniform_draw",
        "encoder": "none",
        "knowledge_basis": "host_llm_general_knowledge",
        "seed": seed,
        "generation_model": generation_model,
        "selected": selected,
        "total": len(concepts),
    }
