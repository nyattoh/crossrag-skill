"""Corpus-only geometric rays from clusters of affine-plane residuals."""
from __future__ import annotations

import math

import numpy as np

from third_concept import _contains_exclude, _normalised_exclude_terms, _rows, corpus_id


def _finite_array(values, ndim, name):
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != ndim or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite {ndim}-dimensional array")
    return array


def _finite_number(value, message):
    if isinstance(value, (str, bytes, bool, np.bool_)) or not np.isscalar(value):
        raise ValueError(message)
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(message) from error
    if not math.isfinite(number):
        raise ValueError(message)
    return number


def _cluster_parameters(cosine_threshold, min_residual_norm):
    message = "cluster cosine must be finite and in (0, 1]"
    cosine_threshold = _finite_number(cosine_threshold, message)
    if not 0 < cosine_threshold <= 1:
        raise ValueError(message)
    message = "minimum residual norm must be finite and positive"
    min_residual_norm = _finite_number(min_residual_norm, message)
    if min_residual_norm <= 0:
        raise ValueError(message)
    return cosine_threshold, min_residual_norm


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be an integer of at least one")
    return int(value)


def _remove_encoder_role_prefixes(text, encoder):
    prefixes = getattr(encoder, "prefix", None)
    if not isinstance(prefixes, dict):
        return text
    known = sorted(
        {prefix for prefix in prefixes.values() if isinstance(prefix, str) and prefix},
        key=len,
        reverse=True,
    )
    while True:
        matched = next((prefix for prefix in known if text.startswith(prefix)), None)
        if matched is None:
            return text
        text = text[len(matched):]


def fit_affine_plane(a_vec, b_vec, c_vec):
    """Fit C + span(A-C, B-C) in raw encoder coordinates."""
    a = _finite_array(a_vec, 1, "anchor A")
    b = _finite_array(b_vec, 1, "anchor B")
    c = _finite_array(c_vec, 1, "anchor C")
    if a.shape != b.shape or a.shape != c.shape:
        raise ValueError("anchor vectors must have matching dimensions")
    generators = np.asarray([a - c, b - c])
    if not np.all(np.isfinite(generators)):
        raise ValueError("plane generators must be finite")
    _, singular_values, basis = np.linalg.svd(generators, full_matrices=False)
    if not np.all(np.isfinite(singular_values)):
        raise ValueError("plane anchors must be finite and of representable scale")
    plane = {
        "status": "unavailable",
        "origin": c.tolist(),
        "singular_values": singular_values.tolist(),
        "dimension": int(c.size),
    }
    if c.size < 3:
        plane["reason"] = "embedding_dimension_below_three"
    elif singular_values[0] <= 1e-12:
        plane["reason"] = "source_anchors_coincide"
    elif singular_values[1] <= 1e-6 * singular_values[0]:
        plane["reason"] = "plane_rank_deficient"
    else:
        plane["status"] = "available"
        plane["basis"] = basis.tolist()
    return plane


def affine_residuals(plane, vectors):
    """Return the components of X-C orthogonal to the available affine plane."""
    if plane.get("status") != "available":
        raise ValueError("affine plane is unavailable")
    origin = _finite_array(plane["origin"], 1, "plane origin")
    basis = _finite_array(plane["basis"], 2, "plane basis")
    vectors = _finite_array(vectors, 2, "candidate vectors")
    dimension = plane["dimension"]
    if origin.shape != (dimension,) or basis.shape != (2, dimension) or vectors.shape[1] != dimension:
        raise ValueError("candidate vectors and affine plane must have matching dimensions")
    offsets = vectors - origin
    residuals = offsets - (offsets @ basis.T) @ basis
    if not np.all(np.isfinite(residuals)):
        raise ValueError("affine residuals must be finite")
    return residuals


def cluster_residual_directions(residuals, *, cosine_threshold, min_residual_norm):
    """Assign in index order by cosine to each seed, never to an updated mean.

    Membership is order-dependent: the first unassigned usable row is the next
    seed, and only that seed's direction decides which unassigned rows join.
    """
    cosine_threshold, min_residual_norm = _cluster_parameters(cosine_threshold, min_residual_norm)
    residuals = _finite_array(residuals, 2, "residuals")
    norms = np.linalg.norm(residuals, axis=1)
    if not np.all(np.isfinite(norms)):
        raise ValueError("residual norms must be finite")
    usable = norms > min_residual_norm
    directions = np.zeros_like(residuals)
    directions[usable] = residuals[usable] / norms[usable, None]
    unassigned = usable.copy()
    clusters = []
    for seed in range(len(residuals)):
        if not unassigned[seed]:
            continue
        # Allow round-off in normalised dot products, including identical vectors at 1.0.
        members = np.flatnonzero(unassigned & ((directions @ directions[seed]) >= cosine_threshold - 1e-12)).tolist()
        if seed not in members:
            members.insert(0, seed)
        unassigned[members] = False
        clusters.append({"seed_index": seed, "member_indices": members})
    return {"clusters": clusters, "unusable_indices": np.flatnonzero(~usable).tolist()}


def cluster_direction(residuals):
    """Summarise raw residuals without replacing their magnitudes by unit weights."""
    residuals = _finite_array(residuals, 2, "residuals")
    member_count = len(residuals)
    resultant = np.sum(residuals, axis=0)
    resultant_norm = float(np.linalg.norm(resultant))
    norm_sum = float(np.sum(np.linalg.norm(residuals, axis=1)))
    if not math.isfinite(resultant_norm) or not math.isfinite(norm_sum):
        raise ValueError("cluster resultant and residual norms must be finite")
    mean = resultant / member_count if member_count else np.zeros(residuals.shape[1])
    available = norm_sum > 0 and resultant_norm > 1e-12 * norm_sum
    return {
        "status": "available" if available else "zero_resultant",
        "direction": (mean / np.linalg.norm(mean)).tolist() if available else None,
        "mean_residual": mean.tolist(),
        "resultant_norm": resultant_norm,
        "coherence": float(np.clip(resultant_norm / norm_sum, 0, 1)) if norm_sum else 0.0,
        "member_count": member_count,
    }


def explore_residual_ray(
    A, B, C, corpus, encoder, *, corpus_path=None, exclude=(),
    radius=0.25, start_distance=0.25, cluster_cosine=0.7,
    min_cluster_size=2, max_rays=5, min_residual_norm=1e-9,
):
    """Cluster corpus residuals, then retrieve every geometric hit for each ray.

    All inputs share encode_passage coordinates. Only explicit exclude terms
    remove rows; a near-zero residual only prevents clustering, not retrieval.
    """
    for name, value in (("A", A), ("B", B), ("C", C)):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string")
    message = "ray radius must be finite and positive"
    radius = _finite_number(radius, message)
    if radius <= 0:
        raise ValueError(message)
    message = "ray start distance must be finite and non-negative"
    start_distance = _finite_number(start_distance, message)
    if start_distance < 0:
        raise ValueError(message)
    cluster_cosine, min_residual_norm = _cluster_parameters(cluster_cosine, min_residual_norm)
    min_cluster_size = _positive_integer(min_cluster_size, "minimum cluster size")
    if max_rays is not None:
        max_rays = _positive_integer(max_rays, "maximum rays")
    exclude = list(exclude)
    exclude_terms = _normalised_exclude_terms(exclude)
    rows = _rows(corpus)
    passage_texts = [line if a is None else f"{a}と{b}。{line}" for a, b, line in rows]
    encoding_texts = [_remove_encoder_role_prefixes(text, encoder) for text in [A, B, C, *passage_texts]]
    vectors = _finite_array(encoder.encode_passage(encoding_texts), 2, "encoded vectors")
    if vectors.shape[0] != 3 + len(rows):
        raise ValueError("encoder must return one vector per input text")
    plane = fit_affine_plane(*vectors[:3])
    candidate_indices = [
        index for index, row in enumerate(rows)
        if not any(_contains_exclude(value, exclude_terms) for value in row if value is not None)
    ]
    clustering = {
        "candidate_count": len(candidate_indices),
        "excluded_count": len(rows) - len(candidate_indices),
        "usable_residual_count": 0,
        "unusable_corpus_indices": [],
        "cluster_count": 0,
        "dropped_small_cluster_count": 0,
        "zero_resultant_cluster_count": 0,
        "truncated_ray_count": 0,
    }
    result = {
        "mode": "normal-ray",
        "method": "affine_residual_cluster",
        "status": "plane_unavailable",
        "input": {"A": A, "B": B, "C": C},
        "encoder": {"name": encoder.name, "model_id": encoder.model_id, "prefix": encoder.prefix},
        "corpus_id": corpus_id(rows, corpus_path),
        "corpus_size": len(rows),
        "params": {
            "coordinates": "raw_encoder_embedding",
            "encoding_path": "encode_passage",
            "origin": "C",
            "plane": "C + span(A-C, B-C)",
            "radius": radius,
            "start_distance": start_distance,
            "cluster_cosine": cluster_cosine,
            "min_cluster_size": min_cluster_size,
            "min_residual_norm": min_residual_norm,
            "max_rays": max_rays,
            "exclude": exclude,
        },
        "plane": plane,
        "clustering": clustering,
        "rays": [],
        "warnings": [],
        "disclaimer": "Geometric residual-direction clusters over the supplied corpus only; no semantic relevance, novelty or usefulness is claimed; empty results are valid.",
    }
    if plane["status"] != "available":
        return result
    passage_vectors = vectors[3:][candidate_indices]
    residuals = affine_residuals(plane, passage_vectors)
    clustered = cluster_residual_directions(
        residuals, cosine_threshold=cluster_cosine, min_residual_norm=min_residual_norm,
    )
    clustering["unusable_corpus_indices"] = [candidate_indices[index] for index in clustered["unusable_indices"]]
    clustering["usable_residual_count"] = len(candidate_indices) - len(clustered["unusable_indices"])
    clustering["cluster_count"] = len(clustered["clusters"])
    rays = []
    for cluster in clustered["clusters"]:
        members = cluster["member_indices"]
        if len(members) < min_cluster_size:
            clustering["dropped_small_cluster_count"] += 1
            continue
        summary = cluster_direction(residuals[members])
        if summary["status"] != "available":
            clustering["zero_resultant_cluster_count"] += 1
            continue
        rays.append({
            "seed_corpus_index": candidate_indices[cluster["seed_index"]],
            "member_corpus_indices": [candidate_indices[index] for index in members],
            **{key: value for key, value in summary.items() if key != "status"},
        })
    rays.sort(key=lambda ray: (-ray["member_count"], ray["seed_corpus_index"]))
    if max_rays is not None:
        clustering["truncated_ray_count"] = max(0, len(rays) - max_rays)
        rays = rays[:max_rays]
    offsets = passage_vectors - np.asarray(plane["origin"])
    residual_norms = np.linalg.norm(residuals, axis=1)
    for ray_id, ray in enumerate(rays):
        direction = np.asarray(ray["direction"])
        axial = offsets @ direction
        axial[residual_norms <= min_residual_norm] = 0.0
        radial = np.linalg.norm(offsets - axial[:, None] * direction, axis=1)
        if not np.all(np.isfinite(axial)) or not np.all(np.isfinite(radial)):
            raise ValueError("ray distances must be finite")
        member_indices = set(ray["member_corpus_indices"])
        hits = []
        for index in np.flatnonzero((axial >= start_distance) & (radial <= radius)):
            corpus_index = candidate_indices[index]
            a, b, line = rows[corpus_index]
            hits.append({
                "corpus_index": corpus_index,
                "a": a,
                "b": b,
                "line": line,
                "axial_distance": float(axial[index]),
                "radial_distance": float(radial[index]),
                "in_cluster": corpus_index in member_indices,
            })
        hits.sort(key=lambda hit: (hit["radial_distance"], hit["corpus_index"]))
        ray.update({"ray_id": ray_id, "hit_count": len(hits), "hits": hits})
    result["rays"] = rays
    result["status"] = "no_residual_clusters" if not rays else "hits_found" if any(ray["hits"] for ray in rays) else "no_hits"
    return result
