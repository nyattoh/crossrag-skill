"""Seeded random rays orthogonal to a context subspace."""
from __future__ import annotations

import math

import numpy as np

from residual_ray import _remove_encoder_role_prefixes
from third_concept import (
    _contains_exclude,
    _normalised_exclude_terms,
    _rows,
    corpus_id,
)


def _finite_array(values, ndim, name):
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite {ndim}-dimensional array") from error
    if array.ndim != ndim or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite {ndim}-dimensional array")
    return array


def _finite_number(value, name):
    if isinstance(value, (str, bytes, bool, np.bool_)) or not np.isscalar(value):
        raise ValueError(f"{name} must be finite")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be finite") from error
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _bounded_number(value, name, lower=0.0, upper=100.0):
    number = _finite_number(value, name)
    if not lower <= number <= upper:
        raise ValueError(f"{name} must be between {lower:g} and {upper:g}")
    return number


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def fit_context_subspace(context_vectors):
    """Fit the affine context directions, using the first vector as origin."""
    vectors = _finite_array(context_vectors, 2, "context vectors")
    if vectors.shape[0] < 2 or vectors.shape[1] < 1:
        raise ValueError("context vectors must contain at least two non-empty vectors")
    origin = vectors[0]
    offsets = vectors[1:] - origin
    if not np.all(np.isfinite(offsets)):
        raise ValueError("context differences must be finite")
    _, singular_values, right_vectors = np.linalg.svd(offsets, full_matrices=False)
    if not np.all(np.isfinite(singular_values)):
        raise ValueError("context vectors must have representable scale")
    sigma_max = float(singular_values[0]) if singular_values.size else 0.0
    threshold = max(1e-12, 1e-6 * sigma_max)
    rank = int(np.count_nonzero(singular_values > threshold))
    dimension = int(vectors.shape[1])
    result = {
        "status": "unavailable",
        "origin": origin.tolist(),
        "dimension": dimension,
        "rank": rank,
        "rank_threshold": threshold,
        "singular_values": singular_values.tolist(),
        "basis": right_vectors[:rank].tolist(),
    }
    if rank == 0:
        result["reason"] = "context_subspace_rank_zero"
    elif rank >= dimension:
        result["reason"] = "context_subspace_spans_embedding"
    else:
        result["status"] = "available"
    return result


def ray_geometry(vectors, origin, direction, radius=0.25):
    """Measure exact axial, radial, and forward-ray distances for each vector."""
    points = _finite_array(vectors, 2, "vectors")
    origin = _finite_array(origin, 1, "origin")
    direction = _finite_array(direction, 1, "direction")
    radius = _bounded_number(radius, "radius")
    if points.shape[1] != origin.size or direction.shape != origin.shape:
        raise ValueError("vectors, origin, and direction must have matching dimensions")
    norm = float(np.linalg.norm(direction))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError("direction must have non-zero finite norm")
    unit = direction / norm
    offsets = points - origin
    axial = offsets @ unit
    radial_vectors = offsets - axial[:, None] * unit
    radial = np.linalg.norm(radial_vectors, axis=1)
    forward = np.maximum(axial, 0.0)
    ray_distance = np.linalg.norm(offsets - forward[:, None] * unit, axis=1)
    if not all(np.all(np.isfinite(values)) for values in (axial, radial, ray_distance)):
        raise ValueError("ray distances must be finite")
    return [
        {
            "axial_distance": float(t),
            "radial_distance": float(r),
            "ray_distance": float(d),
            "behind_origin": bool(t < 0),
            "in_tube": bool(t >= 0 and r <= radius),
        }
        for t, r, d in zip(axial, radial, ray_distance)
    ]


def _passage_text(row):
    a, b, line = row
    return f"{a}と{b}。{line}" if a is not None else line


def _cluster_candidates(residuals, cosine_threshold, min_norm=1e-12):
    """Make stable index-order leader clusters; zero residuals stay singleton."""
    norms = np.linalg.norm(residuals, axis=1)
    if not np.all(np.isfinite(norms)):
        raise ValueError("candidate residual norms must be finite")
    usable = norms > min_norm
    directions = np.zeros_like(residuals)
    directions[usable] = residuals[usable] / norms[usable, None]
    assigned = np.zeros(len(residuals), dtype=bool)
    clusters = []
    for index in range(len(residuals)):
        if assigned[index]:
            continue
        if not usable[index]:
            members = [index]
        else:
            members = [
                candidate for candidate in range(index, len(residuals))
                if not assigned[candidate] and usable[candidate]
                and float(np.dot(directions[index], directions[candidate])) >= cosine_threshold - 1e-12
            ]
            if index not in members:
                members.insert(0, index)
        assigned[members] = True
        clusters.append({
            "cluster_id": f"cluster-{len(clusters):04d}",
            "seed_index": index,
            "member_indices": members,
            "direction": directions[index].tolist() if usable[index] else None,
        })
    return clusters


def explore_random_orthogonal_ray(
    context_texts, corpus, encoder, *, seed=0, ray_count=1, level=60,
    exclude=(), radius=None, cosine_threshold=None, corpus_path=None,
):
    """Explore the corpus with seeded random rays normal to the context span.

    This is geometric retrieval only. Candidate membership is never semantically
    filtered; only explicitly excluded text is removed.
    """
    if not isinstance(context_texts, (list, tuple)) or len(context_texts) < 2 or any(
        not isinstance(text, str) or not text.strip() for text in context_texts
    ):
        raise ValueError("contexts must contain at least two non-empty strings")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    seed = int(seed)
    ray_count = _positive_integer(ray_count, "ray_count")
    if isinstance(level, (bool, np.bool_)) or not isinstance(level, (int, np.integer)):
        raise ValueError("level must be an integer between 0 and 100")
    if not 0 <= level <= 100:
        raise ValueError("level must be an integer between 0 and 100")
    level = int(level)
    strength = level / 100.0
    base_radius = 0.25
    effective_radius = (
        _bounded_number(radius, "radius") if radius is not None
        else base_radius * (0.5 + strength)
    )
    mapped_cosine = 0.85 - 0.35 * strength
    effective_cosine = (
        _bounded_number(cosine_threshold, "cosine_threshold") if cosine_threshold is not None
        else mapped_cosine
    )
    if not 0 < effective_cosine <= 1:
        raise ValueError("cosine_threshold must be in (0, 1]")

    rows = _rows(corpus)
    excludes = _normalised_exclude_terms(exclude)
    eligible = [
        (index, row) for index, row in enumerate(rows)
        if not any(_contains_exclude(value, excludes) for value in row if value is not None)
    ]
    input_texts = [*context_texts, *(_passage_text(row) for _, row in eligible)]
    encoded_texts = [_remove_encoder_role_prefixes(text, encoder) for text in input_texts]
    vectors = _finite_array(encoder.encode_passage(encoded_texts), 2, "encoded vectors")
    if vectors.shape[0] != len(input_texts):
        raise ValueError("encoder must return one vector per input text")
    if vectors.shape[1] < 1:
        raise ValueError("encoded vectors must have a positive dimension")
    context_vectors = vectors[:len(context_texts)]
    subspace = fit_context_subspace(context_vectors)
    common = {
        "mode": "random-orthogonal",
        "context_subspace": subspace,
        "corpus_id": corpus_id(rows, corpus_path),
        "corpus_size": len(rows),
        "encoder": {
            "name": getattr(encoder, "name", None),
            "model_id": getattr(encoder, "model_id", None),
            "prefix": getattr(encoder, "prefix", None),
        },
        "params": {
            "seed": seed,
            "ray_count": ray_count,
            "level": level,
            "serendipity": level,
            "tone_band": "grounded" if level <= 33 else "playful" if level <= 66 else "absurd",
            "base_radius": base_radius,
            "effective_radius": effective_radius,
            "radius": effective_radius,
            "mapped_cosine_threshold": mapped_cosine,
            "cosine_threshold": effective_cosine,
            "cosine_threshold_source": (
                "explicit_override" if cosine_threshold is not None else "serendipity_mapping"
            ),
            "orthogonality_locked": True,
            "encoding_path": "encode_passage",
        },
        "rays": [],
        "combined_clusters": [],
    }
    if subspace["status"] != "available":
        common.update(search_status="unavailable", reason=subspace["reason"])
        return common

    q = np.asarray(subspace["basis"], dtype=np.float64)
    origin = np.asarray(subspace["origin"], dtype=np.float64)
    candidate_vectors = vectors[len(context_texts):]
    offsets = candidate_vectors - origin
    residuals = offsets - (offsets @ q.T) @ q
    if not np.all(np.isfinite(residuals)):
        raise ValueError("candidate residuals must be finite")
    candidate_clusters = _cluster_candidates(residuals, effective_cosine)
    cluster_for_row = {
        eligible[member][0]: cluster
        for cluster in candidate_clusters for member in cluster["member_indices"]
    }

    rng = np.random.default_rng(seed)
    combined = {}
    combined_hits = {}
    rays = []
    for ray_index in range(ray_count):
        direction = None
        for _ in range(8):
            draw = rng.standard_normal(subspace["dimension"])
            projected = draw - q.T @ (q @ draw)
            projected = projected - q.T @ (q @ projected)
            norm = float(np.linalg.norm(projected))
            if math.isfinite(norm) and norm > 1e-12:
                direction = projected / norm
                break
        if direction is None:
            common.update(search_status="unavailable", reason="random_projection_degenerate")
            return common
        if float(np.max(np.abs(q @ direction))) > 1e-12:
            common.update(search_status="unavailable", reason="orthogonality_tolerance_failed")
            return common
        geometry = ray_geometry(candidate_vectors, origin, direction, effective_radius)
        hits = []
        for (corpus_index, row), measure in zip(eligible, geometry):
            hit = {
                "corpus_index": corpus_index,
                "a": row[0],
                "b": row[1],
                "line": row[2],
                **measure,
            }
            if measure["in_tube"]:
                hit["cluster_id"] = cluster_for_row[corpus_index]["cluster_id"]
                hits.append(hit)
        fallback_used = False
        fallback_reason = None
        if not hits and eligible:
            nearest_index, nearest_row, nearest_measure = min(
                ((corpus_index, row, measure) for (corpus_index, row), measure in zip(eligible, geometry)),
                key=lambda item: (item[2]["ray_distance"], item[0]),
            )
            fallback_used = True
            fallback_reason = "no_tube_hits"
            hits = [{
                "corpus_index": nearest_index,
                "a": nearest_row[0],
                "b": nearest_row[1],
                "line": nearest_row[2],
                **nearest_measure,
                "cluster_id": f"fallback-ray-{ray_index:04d}-row-{nearest_index}",
            }]
        hits_by_cluster = {}
        for hit in hits:
            cluster_id = hit["cluster_id"]
            hits_by_cluster.setdefault(cluster_id, []).append(hit)
            entry = combined.setdefault(cluster_id, {
                "cluster_id": cluster_id,
                "ray_index": ray_index,
                "ray_indices": [],
                "hits": [],
                "fallback_used": fallback_used,
                "fallback_reason": fallback_reason,
            })
            cluster_hits = combined_hits.setdefault(cluster_id, {})
            combined_hit = cluster_hits.get(hit["corpus_index"])
            if combined_hit is None:
                combined_hit = {
                    **{key: hit[key] for key in ("corpus_index", "a", "b", "line")},
                    "cluster_id": cluster_id,
                    "ray_associations": [],
                }
                cluster_hits[hit["corpus_index"]] = combined_hit
                entry["hits"].append(combined_hit)
            if ray_index not in entry["ray_indices"]:
                entry["ray_indices"].append(ray_index)
            combined_hit["ray_associations"].append({
                "ray_index": ray_index,
                **{key: hit[key] for key in (
                    "axial_distance", "radial_distance", "ray_distance",
                    "behind_origin", "in_tube",
                )},
                "fallback_used": fallback_used,
                "fallback_reason": fallback_reason,
            })
        selected_clusters = []
        for cluster_id, members in hits_by_cluster.items():
            selected_clusters.append({
                "cluster_id": cluster_id,
                "member_indices": [hit["corpus_index"] for hit in members],
                "hits": members,
                "fallback": fallback_used,
            })
        rays.append({
            "ray_index": ray_index,
            "origin": origin.tolist(),
            "direction": direction.tolist(),
            "selected_clusters": selected_clusters,
            "hits": hits,
            "fallback_used": fallback_used,
            "fallback_reason": fallback_reason,
        })
    common["rays"] = rays
    common["combined_clusters"] = list(combined.values())
    if eligible:
        common["search_status"] = "available"
    else:
        common.update(search_status="unavailable", reason="empty_or_excluded_corpus")
    return common
