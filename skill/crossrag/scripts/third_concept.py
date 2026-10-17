"""Exploratory third-concept ranking over explicit candidate pools."""
from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from pathlib import Path

import numpy as np

from engine import HashEncoder, SentenceEncoder, normalise


SCORE_KEYS = (
    "s_A",
    "s_B",
    "plane_projection",
    "residual",
    "emergence_norm",
    "balance",
    "unexpectedness_h",
    "coherence_h",
    "score_third",
)
SCORE_BY_METHOD = {
    "third-concept": "score_third",
    "third-concept-ratio": "score_ratio",
    "third-concept-no-wedge": "score_no_wedge",
    "mean": "s_mean",
}


class ConceptEncoder:
    """Apply model-appropriate prefixes while exposing an engine-compatible API."""

    def __init__(self, model, name, model_id, prefix):
        self.model = model
        self.name = name
        self.model_id = model_id
        self.prefix = prefix

    def _prepare(self, texts, kind):
        texts = list(texts)
        if self.prefix is None:
            return texts
        wanted = self.prefix[kind]
        known = tuple(self.prefix.values())
        return [text if text.startswith(known) else wanted + text for text in texts]

    def _encode(self, texts, kind):
        prepared = self._prepare(texts, kind)
        if isinstance(self.model, HashEncoder):
            return np.asarray(self.model.encode(prepared), dtype=np.float64)
        return np.asarray(
            self.model.encode(
                prepared,
                normalize_embeddings=True,
                show_progress_bar=False,
            ),
            dtype=np.float64,
        )

    def encode_query(self, texts):
        return self._encode(texts, "query")

    def encode_passage(self, texts):
        return self._encode(texts, "passage")

    def encode(self, texts):
        return self.encode_query(texts)

    def encode_passage_tokens(self, texts):
        """Return contextual token vectors when the backend exposes them; otherwise one full-text vector per row."""
        prepared = self._prepare(texts, "passage")
        if isinstance(self.model, HashEncoder):
            return [np.asarray([row], dtype=np.float64) for row in self.model.encode(prepared)]
        try:
            values = self.model.encode(prepared, output_value="token_embeddings", show_progress_bar=False)
            return [np.asarray(value, dtype=np.float64) for value in values]
        except (TypeError, ValueError, AttributeError):
            encoded = self.encode_passage(texts)
            return [np.asarray([row], dtype=np.float64) for row in encoded]


def make_encoder(kind, model_id="intfloat/multilingual-e5-small", model=None, prefix_policy="auto"):
    if kind == "hash":
        backend = model if model is not None else HashEncoder()
        return ConceptEncoder(backend, backend.name, "hash-ngram", None)
    if kind != "semantic":
        raise ValueError("encoder kind must be 'hash' or 'semantic'")
    if prefix_policy not in {"auto", "e5", "prefix-free"}:
        raise ValueError("prefix_policy must be 'auto', 'e5', or 'prefix-free'")
    backend = model
    if backend is None:
        backend = SentenceEncoder(model_id).model
    # auto is deliberately closed-world: known e5 IDs get role prefixes; unknown IDs are prefix-free.
    prefix = {"query": "query: ", "passage": "passage: "} if prefix_policy == "e5" or (prefix_policy == "auto" and "e5" in model_id.casefold()) else None
    return ConceptEncoder(backend, model_id + "（意味埋め込み）", model_id, prefix)


def token_geometry(a_tokens, b_tokens, candidate_tokens, *, mu=None):
    """Score joint A+B direction support from contextual token vectors.

    All vectors share one model coordinate space; the default common centre is
    the unweighted mean of the unit A anchor, unit B anchor, and each candidate
    phrase's mean unit-token vector. Callers may supply an explicit shared ``mu``.
    The centre is subtracted before direction measurements. Per candidate, token
    directions are averaged (never max-pooled), then unit-normalised.

    A/B side support is positive cosine (reported separately). Positive-cone
    coordinates are obtained by solving the 2-anchor Gram system. Between support
    is plane support times a balance term: 1 for an equal positive A+B mix, 0.5
    on either positive cone boundary, and 0 outside the positive cone. Plane
    support is squared projected-norm fraction. Final score is the fourth root
    of the product of plane support, between support, and the two smoothed side
    supports ``(1 + positive cosine) / 2``. Thus a single-side vector scores
    below a balanced joint vector, while orthogonal-plane noise and zero vectors
    score zero. This is a geometric diagnostic, not semantic/jointness detection.
    """
    def rows(value):
        array = np.asarray(value, dtype=np.float64)
        return np.atleast_2d(array)

    def unit(vector):
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm > 1e-15 else np.zeros_like(vector, dtype=np.float64)

    a_raw, b_raw = rows(a_tokens), rows(b_tokens)
    if a_raw.shape[1] != b_raw.shape[1]:
        raise ValueError("A and B token vectors must have the same dimension")
    a_raw, b_raw = np.mean(a_raw, axis=0), np.mean(b_raw, axis=0)
    try:
        candidate_array = np.asarray(candidate_tokens, dtype=np.float64)
    except (ValueError, TypeError):
        candidate_array = None
    if candidate_array is None:
        candidate_groups = [rows(group) for group in candidate_tokens]
    elif candidate_array.ndim == 1:
        candidate_groups = [candidate_array[None, :]]
    elif candidate_array.ndim == 2:
        candidate_groups = [candidate_array[index:index + 1] for index in range(len(candidate_array))]
    elif candidate_array.ndim == 3:
        candidate_groups = [candidate_array[index] for index in range(len(candidate_array))]
    else:
        raise ValueError("candidate_tokens must be a token-vector matrix or a batch of token-vector matrices")
    if any(group.shape[1] != len(a_raw) for group in candidate_groups):
        raise ValueError("candidate token vectors must match the A/B vector dimension")

    pooled = []
    for group in candidate_groups:
        token_units = [unit(token) for token in group]
        nonzero = [token for token in token_units if np.linalg.norm(token) > 0]
        pooled.append(unit(np.mean(nonzero, axis=0)) if nonzero else np.zeros_like(a_raw))
    a_unit, b_unit = unit(a_raw), unit(b_raw)
    if mu is None:
        centre_rows = [a_unit, b_unit, *pooled]
        centre = np.mean(centre_rows, axis=0) if centre_rows else np.zeros_like(a_raw)
    else:
        centre = np.asarray(mu, dtype=np.float64)
        if centre.shape != a_raw.shape:
            raise ValueError("mu must match the A/B vector dimension")
    a, b = unit(a_unit - centre), unit(b_unit - centre)
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return [{"score": 0.0, "wedge_norm": 0.0, "a_side_support": 0.0,
                 "b_side_support": 0.0, "between_direction_support": 0.0,
                 "plane_support": 0.0, "aggregation": "mean-vector-geometric-v2"}
                for _ in candidate_groups]

    gram = np.array([[np.dot(a, a), np.dot(a, b)], [np.dot(b, a), np.dot(b, b)]])
    if float(np.linalg.det(gram)) <= 1e-10:
        return [{"score": 0.0, "wedge_norm": 0.0, "a_side_support": 0.0,
                 "b_side_support": 0.0, "between_direction_support": 0.0,
                 "plane_support": 0.0, "aggregation": "mean-vector-geometric-v2"}
                for _ in candidate_groups]
    gram_inverse = np.linalg.pinv(gram, rcond=1e-12)
    bisector = unit(a + b)
    output = []
    for c in pooled:
        c = unit(c - centre)
        if np.linalg.norm(c) == 0:
            a_support = b_support = between = plane = score = 0.0
        else:
            a_cos = float(np.clip(np.dot(c, a), -1.0, 1.0))
            b_cos = float(np.clip(np.dot(c, b), -1.0, 1.0))
            a_support, b_support = max(0.0, a_cos), max(0.0, b_cos)
            rhs = np.array([np.dot(a, c), np.dot(b, c)])
            coefficients = gram_inverse @ rhs
            projected = coefficients[0] * a + coefficients[1] * b
            plane = float(np.clip(np.dot(projected, projected) / np.dot(c, c), 0.0, 1.0))
            positive = np.maximum(coefficients, 0.0)
            positive_sum = float(np.sum(positive))
            positive_cone = bool(np.all(coefficients >= -1e-12)) and positive_sum > 1e-12
            balance = 2 * float(np.min(positive)) / positive_sum if positive_cone else 0.0
            between = plane * (0.5 + 0.5 * balance) if positive_cone else 0.0
            factors = ((1 + a_support) / 2, (1 + b_support) / 2, between, plane)
            score = float(np.prod(factors) ** 0.25)
        output.append({
            "score": float(np.clip(score, 0.0, 1.0)),
            "wedge_norm": float(np.clip(score, 0.0, 1.0)),
            "a_side_support": float(a_support),
            "b_side_support": float(b_support),
            "between_direction_support": float(np.clip(between, 0.0, 1.0)),
            "plane_support": float(np.clip(plane, 0.0, 1.0)),
            "aggregation": "mean-vector-geometric-v2",
        })
    return output


def _centred_unit(vector, mu):
    vector = np.asarray(vector, dtype=np.float64)
    return normalise(vector - mu) if mu is not None else normalise(vector)


def _validated_p_min(p_min):
    try:
        value = float(p_min)
    except (TypeError, ValueError) as exc:
        raise ValueError("p_min must be finite and between 0 and 1") from exc
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("p_min must be finite and between 0 and 1")
    return value


def geometry(a_vec, b_vec, c_vecs, mu=None, p_min=0.0):
    """Return the design's geometric heuristic values for each candidate vector."""
    p_min = _validated_p_min(p_min)
    mu_array = None if mu is None else np.asarray(mu, dtype=np.float64)
    a = _centred_unit(a_vec, mu_array)
    b = _centred_unit(b_vec, mu_array)
    candidates = np.atleast_2d(np.asarray(c_vecs, dtype=np.float64))
    rho = float(np.clip(np.dot(a, b), -1.0, 1.0))
    e2_raw = b - rho * a
    e2_norm = float(np.linalg.norm(e2_raw))
    degenerate = e2_norm < 1e-6 or (1.0 - rho) < 1e-6
    e2 = np.zeros_like(a) if degenerate else e2_raw / e2_norm
    mean_direction = normalise(a + b)
    upper_bound = max(0.0, (1.0 - rho) / 2.0)
    warning = "A と B の張る平面が退化しているため emergence_norm を 0 とした。"
    results = []
    for raw in candidates:
        c = _centred_unit(raw, mu_array)
        s_a = float(np.clip(np.dot(c, a), -1.0, 1.0))
        s_b = float(np.clip(np.dot(c, b), -1.0, 1.0))
        p_sq = float(np.clip(np.dot(c, a) ** 2 + (0.0 if degenerate else np.dot(c, e2) ** 2), 0.0, 1.0))
        p = math.sqrt(p_sq)
        residual = math.sqrt(max(0.0, 1.0 - p_sq))
        emergence = 0.0
        if not degenerate and s_a > 0.0 and s_b > 0.0:
            emergence = max(0.0, p_sq - max(s_a * s_a, s_b * s_b))
        emergence_norm = float(np.clip(emergence / upper_bound, 0.0, 1.0)) if upper_bound > 1e-12 else 0.0
        between_ab = False
        if not degenerate:
            denominator = 1.0 - rho * rho
            alpha = (s_a - rho * s_b) / denominator
            beta = (s_b - rho * s_a) / denominator
            between_ab = alpha >= -1e-12 and beta >= -1e-12 and p > 1e-12
        emergence_ratio = 0.0
        if between_ab and s_a > 0.0 and s_b > 0.0:
            emergence_ratio = float(np.clip(1.0 - max(s_a * s_a, s_b * s_b) / p_sq, 0.0, 1.0))
        emergence_ratio_norm = float(np.clip(emergence_ratio / upper_bound, 0.0, 1.0)) if upper_bound > 1e-12 else 0.0
        p_below_min = p < p_min
        s_mean = float(np.clip(np.dot(c, mean_direction), -1.0, 1.0)) if np.linalg.norm(mean_direction) else 0.0
        # The in-plane projection must point towards A+B; a candidate opposed to both is not "half explained" by them.
        balance = float(np.clip(2.0 * p * residual, 0.0, 1.0)) if s_mean > 0.0 else 0.0
        unexpectedness = float(np.clip(1.0 - max(s_a, s_b, 0.0), 0.0, 1.0))
        score_no_wedge = 2.0 * s_mean * math.sqrt(max(0.0, 1.0 - s_mean * s_mean))
        results.append(
            {
                "s_A": s_a,
                "s_B": s_b,
                "plane_projection": p,
                "residual": residual,
                "emergence_norm": emergence_norm,
                "between_AB": bool(between_ab),
                "emergence_ratio": emergence_ratio,
                "emergence_ratio_norm": emergence_ratio_norm,
                "p_below_min": bool(p_below_min),
                "balance": balance,
                "unexpectedness_h": unexpectedness,
                "coherence_h": p,
                "score_third": 0.5 * emergence_norm + 0.5 * balance,
                "score_ratio": 0.5 * (0.0 if p_below_min else emergence_ratio_norm) + 0.5 * balance,
                "s_mean": s_mean,
                "score_no_wedge": score_no_wedge,
                "warnings": [warning] if degenerate else [],
            }
        )
    return results


def null_geometry(a_vec, b_vec, passage_vectors, mu, n=1000, seed=0):
    if type(n) is not int or n < 1:
        raise ValueError("n must be an integer at least 1")
    passages = np.asarray(passage_vectors, dtype=np.float64)
    if passages.size == 0:
        raise ValueError("passage_vectors must not be empty")
    passages = np.atleast_2d(passages)
    mu_array = None if mu is None else np.asarray(mu, dtype=np.float64)
    a = _centred_unit(a_vec, mu_array)
    b = _centred_unit(b_vec, mu_array)
    centred_passages = np.asarray([_centred_unit(vector, mu_array) for vector in passages])
    dimension = a.shape[0]

    random_rng = np.random.default_rng(seed)
    random_vectors = np.asarray([normalise(vector) for vector in random_rng.standard_normal((n, dimension))])
    shuffled_rng = np.random.default_rng(seed + 1)
    shuffled_vectors = np.asarray(
        [centred_passages[index % len(centred_passages)][shuffled_rng.permutation(dimension)] for index in range(n)]
    )

    result = {}
    for name, vectors in (("random", random_vectors), ("shuffled", shuffled_vectors)):
        rows = geometry(a, b, vectors, mu=None)
        result[name] = {
            "p": np.asarray([row["plane_projection"] for row in rows]),
            "emergence_ratio": np.asarray([row["emergence_ratio"] for row in rows]),
        }
    return result


def auto_p_min(null):
    return float(np.percentile(null["shuffled"]["p"], 95))


def is_plain_row(row):
    """A plain-text row is a string or a dict whose only key is "line"; it carries no concept pair."""
    return isinstance(row, str) or (isinstance(row, dict) and set(row) == {"line"})


def _rows(corpus):
    rows = []
    for row in corpus:
        if is_plain_row(row):
            line = row if isinstance(row, str) else row.get("line")
            if not isinstance(line, str) or not line.strip():
                raise ValueError("plain-text corpus items must be non-empty strings")
            rows.append((None, None, line))
            continue
        if isinstance(row, dict):
            row = (row.get("a"), row.get("b"), row.get("line"))
        if not isinstance(row, (list, tuple)) or len(row) != 3 or not all(isinstance(value, str) and value.strip() for value in row):
            raise ValueError("each corpus item must contain non-empty a, b, line strings")
        rows.append(tuple(row))
    return rows


def _row_values(row):
    return [value for value in row if value is not None]


def _vocabulary_pool(rows, vocab=None):
    pool = []
    for a, b, _ in rows:
        for value in (a, b):
            if value is not None and value not in pool:
                pool.append(value)
    for value in vocab or []:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("vocab entries must be non-empty strings")
        if value not in pool:
            pool.append(value)
    return pool


def shared_geometry_space(A, B, corpus, encoder, vocab=None, center=True):
    """Encode one exploration space and compute its single shared centring mean."""
    rows = _rows(corpus)
    pool = _vocabulary_pool(rows, vocab)
    query_texts = []
    for value in (A, B, *pool):
        if value not in query_texts:
            query_texts.append(value)
    query_vectors = encoder.encode_query(query_texts)
    query_by_text = dict(zip(query_texts, query_vectors))
    passage_texts = [line if a is None else f"{a}と{b}。{line}" for a, b, line in rows]
    passage_vectors = encoder.encode_passage(passage_texts)
    all_vectors = np.vstack([query_vectors, passage_vectors])
    return {
        "rows": rows,
        "pool": pool,
        "pool_vectors": np.asarray([query_by_text[value] for value in pool]),
        "passage_texts": passage_texts,
        "passage_vectors": passage_vectors,
        "a_vector": query_by_text[A],
        "b_vector": query_by_text[B],
        "mu": np.mean(all_vectors, axis=0) if center else None,
    }


def _rank_vectors(a_vector, b_vector, candidate_vectors, method, limit, mu, p_min=0.0):
    metrics = geometry(a_vector, b_vector, candidate_vectors, mu=mu, p_min=p_min)
    score_name = SCORE_BY_METHOD[method]
    order = sorted(range(len(metrics)), key=lambda index: (-metrics[index][score_name], index))
    return [(index, metrics[index]) for index in order[: max(0, int(limit))]]


def rank_entries(
    A, B, corpus, encoder, method, limit, center=True, space=None, exclusion_inputs=None,
    exclude=(), excluded=None, p_min=0.0,
):
    if method not in SCORE_BY_METHOD:
        raise ValueError("unknown third-concept ranking method")
    if int(limit) < 1:
        raise ValueError("limit must be at least 1")
    space = space or shared_geometry_space(A, B, corpus, encoder, center=center)
    rows = space["rows"]
    exclusion_a, exclusion_b = exclusion_inputs or (A, B)
    if excluded is None:
        excluded = exclusion_mask(space, exclusion_a, exclusion_b, exclude)
    else:
        excluded = list(excluded)
        if len(excluded) != len(rows):
            raise ValueError("excluded mask length must match corpus rows")
    ranked = []
    all_ranked = _rank_vectors(
        space["a_vector"], space["b_vector"], space["passage_vectors"], method, len(rows), space["mu"], p_min
    )
    for index, scores in all_ranked:
        a, b, line = rows[index]
        if excluded[index]:
            continue
        ranked.append({"index": index, "a": a, "b": b, "line": line, "score": scores[SCORE_BY_METHOD[method]], "scores": scores})
        if len(ranked) >= int(limit):
            break
    return ranked


def _normalised_text(text):
    return "".join(unicodedata.normalize("NFKC", text).casefold().split())


def _contains_input(candidate, A, B):
    candidate_norm = _normalised_text(candidate)
    for source in (A, B):
        source_norm = _normalised_text(source)
        if source_norm and (candidate_norm in source_norm or source_norm in candidate_norm):
            return True
    return False


def _normalised_exclude_terms(exclude):
    terms = []
    for term in exclude:
        if not isinstance(term, str) or not term.strip():
            raise ValueError("exclude terms must be non-empty strings")
        normalised = _normalised_text(term)
        if not normalised:
            raise ValueError("exclude terms must be non-empty strings")
        terms.append(normalised)
    return terms


def _contains_exclude(candidate, terms):
    candidate_norm = _normalised_text(candidate)
    return any(term in candidate_norm for term in terms)


def _raw_cosine_exclusion_mask_from_vectors(vectors, a_vector, b_vector):
    unit_anchors = [normalise(a_vector), normalise(b_vector)]
    return [any(float(np.dot(normalise(vector), anchor)) >= 0.98 for anchor in unit_anchors) for vector in vectors]


def _raw_cosine_exclusion_mask(candidates, A, B, encoder, kind="query"):
    if not candidates:
        return []
    anchors = encoder.encode_query([A, B])
    vectors = encoder.encode_passage(candidates) if kind == "passage" else encoder.encode_query(candidates)
    return _raw_cosine_exclusion_mask_from_vectors(vectors, anchors[0], anchors[1])


def exclusion_mask(space, A, B, exclude=()):
    terms = _normalised_exclude_terms(exclude)
    cosine_excluded = _raw_cosine_exclusion_mask_from_vectors(
        space["passage_vectors"], space["a_vector"], space["b_vector"]
    )
    return [
        cosine_excluded[index]
        or any(_contains_input(value, A, B) or _contains_exclude(value, terms) for value in _row_values(row))
        for index, row in enumerate(space["rows"])
    ]


def _public_scores(scores):
    return {key: round(float(scores[key]), 6) for key in SCORE_KEYS}


def _public_ratio_scores(scores):
    return {
        "emergence_ratio": round(float(scores["emergence_ratio"]), 6),
        "emergence_ratio_norm": round(float(scores["emergence_ratio_norm"]), 6),
        "score_ratio": round(float(scores["score_ratio"]), 6),
        "between_AB": bool(scores["between_AB"]),
        "p_below_min": bool(scores["p_below_min"]),
    }


def corpus_id(rows, path=None):
    if path is not None:
        content = Path(path).read_bytes()
    else:
        content = json.dumps(list(rows), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(content).hexdigest()[:16]


def load_hypotheses(path, A, B):
    source = Path(path)
    raw = source.read_bytes()
    data = json.loads(raw.decode("utf-8"))
    if data.get("input") != {"A": A, "B": B}:
        raise ValueError("hypotheses input does not match A and B")
    if not isinstance(data.get("generator"), dict) or not isinstance(data.get("hypotheses"), list):
        raise ValueError("invalid hypotheses file")
    required = {"concept", "kind", "explanation", "relation_to_A_alone", "relation_to_B_alone", "why_requires_AB", "uncertainty", "falsification_or_check", "llm_self_assessment"}
    for item in data["hypotheses"]:
        if not isinstance(item, dict) or not required <= item.keys() or item.get("kind") not in {"llm_hypothesis", "idea"} or not all(isinstance(item.get(key), str) and item[key].strip() for key in required - {"kind", "llm_self_assessment"}):
            raise ValueError("invalid hypothesis item")
        assessment = item.get("llm_self_assessment")
        if assessment is not None:
            expected = {"unexpectedness", "coherence", "emergence", "testability"}
            if set(assessment) != expected or any(type(value) is not int or not 1 <= value <= 5 for value in assessment.values()):
                raise ValueError("invalid llm_self_assessment")
    serialised = json.dumps(data, ensure_ascii=False)
    if re.search(r"(?i)\b(?:verified|proven|confirmed)\b|証明済|検証済", serialised):
        raise ValueError("hypotheses file contains disallowed claim wording")
    data["_path"] = str(source)
    data["_sha256"] = hashlib.sha256(raw).hexdigest()
    return data


def _candidate(origin, number, concept, scores, generation_method, relation_a, relation_b, why, uncertainty, check, source, assessment=None, explanation=None):
    result = {
        "id": f"{origin}-{number}",
        "concept": concept,
        "origin": origin,
        "generation_method": generation_method,
        "relation_to_A_alone": relation_a,
        "relation_to_B_alone": relation_b,
        "why_requires_AB": why,
        "uncertainty": uncertainty,
        "falsification_or_check": check,
        "scores": {
            "heuristic_math": _public_scores(scores),
            "heuristic_ratio": _public_ratio_scores(scores),
            "llm_self_assessment": assessment,
        },
        "source": source,
    }
    if explanation is not None:
        result["explanation"] = explanation
    return result


def _why_requires_ab(scores):
    if scores["emergence_norm"] == 0.0:
        return "幾何上は A∧B への依存を示さない（emergence_norm=0.000）。意味的必然性は未検証"
    p_sq = scores["plane_projection"] ** 2
    single_sq = max(scores["s_A"] ** 2, scores["s_B"] ** 2)
    return f"p²={p_sq:.3f} が単独最大 {single_sq:.3f} を上回る（emergence_norm={scores['emergence_norm']:.3f}）。意味的必然性は未検証"


def explore(
    A, B, corpus, encoder, limit=5, seed=0, vocab=None, hypotheses=None, center=True, corpus_path=None,
    exclude=(), method="third-concept", p_min=None, null_samples=1000,
):
    if int(limit) < 1:
        raise ValueError("limit must be at least 1")
    if method not in {"third-concept", "third-concept-ratio"}:
        raise ValueError("method must be 'third-concept' or 'third-concept-ratio'")
    if type(null_samples) is not int or null_samples < 1:
        raise ValueError("null_samples must be an integer at least 1")
    exclude = list(exclude)
    exclude_terms = _normalised_exclude_terms(exclude)
    space = shared_geometry_space(A, B, corpus, encoder, vocab=vocab, center=center)
    rows = space["rows"]
    if p_min is None:
        null = null_geometry(
            space["a_vector"], space["b_vector"], space["passage_vectors"], space["mu"],
            n=null_samples, seed=seed,
        )
        p_min = auto_p_min(null)
        p_min_source = "auto:shuffled-p95"
    else:
        p_min = _validated_p_min(p_min)
        p_min_source = "user"
    warnings = []
    excluded = 0
    candidates = []

    ranked_passages = _rank_vectors(
        space["a_vector"], space["b_vector"], space["passage_vectors"], method, len(rows), space["mu"], p_min
    )
    passage_excluded = exclusion_mask(space, A, B, exclude)
    retrieved_number = 0
    for index, scores in ranked_passages:
        a, b, line = rows[index]
        if passage_excluded[index]:
            excluded += 1
            continue
        if retrieved_number < int(limit):
            retrieved_number += 1
            candidates.append(
                _candidate(
                    "retrieved",
                    retrieved_number,
                    line,
                    scores,
                    "third-concept score over corpus passages",
                    f"s_A={scores['s_A']:.3f}（中心化コサイン）",
                    f"s_B={scores['s_B']:.3f}（中心化コサイン）",
                    _why_requires_ab(scores),
                    "コーパス内の記述であること以外は未確認。幾何スコアは探索用代理指標。",
                    "出典記述と A・B の関係を独立に確認する。",
                    {"corpus_index": index, "a": a, "b": b, "line": line},
                )
            )
            warnings.extend(scores["warnings"])

    pool = space["pool"]
    pool_cosine_excluded = _raw_cosine_exclusion_mask_from_vectors(
        space["pool_vectors"], space["a_vector"], space["b_vector"]
    )
    kept_pool = []
    kept_pool_vectors = []
    for value, vector, cosine_excluded in zip(pool, space["pool_vectors"], pool_cosine_excluded):
        if _contains_input(value, A, B) or _contains_exclude(value, exclude_terms) or cosine_excluded:
            excluded += 1
        else:
            kept_pool.append(value)
            kept_pool_vectors.append(vector)
    if kept_pool:
        ranked_math = _rank_vectors(
            space["a_vector"], space["b_vector"], np.asarray(kept_pool_vectors), method,
            min(int(limit), len(kept_pool)), space["mu"], p_min
        )
        for number, (index, scores) in enumerate(ranked_math, 1):
            concept = kept_pool[index]
            candidates.append(
                _candidate(
                    "math",
                    number,
                    concept,
                    scores,
                    "third-concept score over vocabulary pool",
                    f"s_A={scores['s_A']:.3f}（中心化コサイン）",
                    f"s_B={scores['s_B']:.3f}（中心化コサイン）",
                    _why_requires_ab(scores),
                    "候補語彙プール内の位置だけを示す。意味的妥当性は未検証。",
                    "A・B の組合せに依存する説明が再現できるか人手で確認する。",
                    {},
                )
            )
            warnings.extend(scores["warnings"])

    hypotheses_data = hypotheses
    if hypotheses_data is not None and not isinstance(hypotheses_data, dict):
        hypotheses_data = load_hypotheses(hypotheses_data, A, B)
    if hypotheses_data is not None:
        if hypotheses_data.get("input") != {"A": A, "B": B}:
            raise ValueError("hypotheses input does not match A and B")
        accepted = []
        hypothesis_items = hypotheses_data["hypotheses"]
        hypothesis_cosine_excluded = _raw_cosine_exclusion_mask([item["concept"] for item in hypothesis_items], A, B, encoder)
        for item, cosine_excluded in zip(hypothesis_items, hypothesis_cosine_excluded):
            concept = item["concept"]
            if _contains_input(concept, A, B) or _contains_exclude(concept, exclude_terms) or cosine_excluded:
                excluded += 1
                warnings.append(f"入力概念と重なる仮説候補を除外した: {concept}")
            else:
                accepted.append(item)
        if accepted:
            texts = [item["concept"] for item in accepted]
            hypothesis_vectors = encoder.encode_query(texts)
            ranked = _rank_vectors(
                space["a_vector"], space["b_vector"], hypothesis_vectors, method, len(texts), space["mu"], p_min
            )
            metrics_by_index = {index: scores for index, scores in ranked}
            ordered = ranked if method == "third-concept-ratio" else [
                (index, metrics_by_index[index]) for index in range(len(accepted))
            ]
            counters = {"llm_hypothesis": 0, "idea": 0}
            for index, scores in ordered:
                item = accepted[index]
                origin = item["kind"]
                counters[origin] += 1
                candidates.append(
                    _candidate(
                        origin,
                        counters[origin],
                        item["concept"],
                        scores,
                        "external hypothesis file",
                        item["relation_to_A_alone"],
                        item["relation_to_B_alone"],
                        item["why_requires_AB"],
                        item["uncertainty"],
                        item["falsification_or_check"],
                        {"hypotheses_file": hypotheses_data.get("_path"), "generator": hypotheses_data["generator"]},
                        item.get("llm_self_assessment"),
                        item.get("explanation"),
                    )
                )
                warnings.extend(scores["warnings"])

    warnings = list(dict.fromkeys(warnings))
    hypotheses_sha = hypotheses_data.get("_sha256") if hypotheses_data else None
    return {
        "mode": "third-concept",
        "input": {"A": A, "B": B},
        "encoder": {
            "name": encoder.name,
            "model_id": encoder.model_id,
            "prefix": encoder.prefix,
            "applied_to": None if encoder.prefix is None else {"query": ["A", "B", "concept candidates"], "passage": ["corpus entries"]},
        },
        "corpus_id": corpus_id(rows, corpus_path),
        "corpus_size": len(rows),
        "params": {
            "limit": limit,
            "seed": seed,
            "center": bool(center),
            "weights": {"emergence_norm": 0.5, "balance": 0.5},
            "excluded_inputs": excluded,
            "vocab_size": len(vocab or []),
            "hypotheses_sha256": hypotheses_sha,
            "method": method,
            "p_min": p_min,
            "p_min_source": p_min_source,
            "exclude": exclude,
            "null_samples": null_samples,
        },
        "origin_legend": {
            "retrieved": "指定コーパス内の記述。内容の真偽は未確認。",
            "math": "候補語彙の埋め込み空間上の位置。意味的妥当性は未検証。",
            "llm_hypothesis": "外部 LLM が説明付きで作った未検証の仮説。自己評価は独立評価ではない。",
            "idea": "外部 LLM が作った説明の連鎖が弱い着想。自己評価は独立評価ではない。",
        },
        "candidates": candidates,
        "warnings": warnings,
        "disclaimer": "すべての候補は未検証。retrieved はコーパス内の存在のみ、math は埋め込み空間上の位置のみ、llm_hypothesis / idea は LLM の推論であり、いずれも事実・発見として扱わない。",
    }


def build_hypothesis_prompt(A, B, math_candidates=None):
    prompt = f"""あなたは発想支援のための仮説生成役です。概念A「{A}」と概念B「{B}」の組み合わせから生じうる、予想外の第三概念Cの候補を5〜8個挙げてください。
条件:
- AまたはB単独からは容易に連想できず、AとBの組み合わせに依存するものを優先する。
- 一般論・既存研究の要約・並列説明・助言の列挙は避け、具体的な現象・製品・仕組み・実験方法の形にする。
- 各候補は検証されていない仮説として書く。確立した事実のように書かない。
- 各候補について concept, kind(llm_hypothesis|idea), explanation, relation_to_A_alone, relation_to_B_alone, why_requires_AB, uncertainty, falsification_or_check, llm_self_assessment(unexpectedness, coherence, emergence, testability を1〜5) を与える。
- 出力は JSON のみ: {{"hypotheses": [...]}}"""
    if math_candidates:
        concepts = [item.get("concept", "") if isinstance(item, dict) else str(item) for item in math_candidates]
        prompt += "\n参考: 埋め込み空間上の候補語（意味的妥当性は未検証）: " + "、".join(filter(None, concepts))
    return prompt
