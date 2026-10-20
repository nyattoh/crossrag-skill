"""Synthetic contracts for seeded random-orthogonal serendipity retrieval."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from serendipity_ray import (  # noqa: E402
    explore_random_orthogonal_ray,
    fit_context_subspace,
    ray_geometry,
)


class FixedEncoder:
    name = "fixed"
    model_id = "fixed-serendipity-test"
    prefix = {"query": "query: ", "passage": "passage: "}

    def __init__(self, vectors):
        self.vectors = {text: np.asarray(vector, dtype=float) for text, vector in vectors.items()}
        self.calls = []

    def encode_query(self, texts):
        raise AssertionError("random-orthogonal retrieval must only use encode_passage")

    def encode_passage(self, texts):
        self.calls.append(list(texts))
        return np.asarray([self.vectors[text] for text in texts], dtype=float)


class ContextSubspaceTest(unittest.TestCase):
    def test_first_context_is_origin_and_svd_rank_uses_scaled_threshold(self):
        vectors = np.asarray([[8, -3, 2, 1], [9, -3, 2, 1], [8, -3, 2, 1 + 1e-7]])
        subspace = fit_context_subspace(vectors)

        self.assertEqual(subspace["origin"], vectors[0].tolist())
        self.assertEqual(subspace["dimension"], 4)
        self.assertEqual(subspace["rank"], 1)
        self.assertEqual(subspace["status"], "available")
        self.assertEqual(subspace["rank_threshold"], max(1e-12, 1e-6 * subspace["singular_values"][0]))
        self.assertEqual(np.asarray(subspace["basis"]).shape, (1, 4))

    def test_context_subspace_rejects_rank_zero_rank_d_and_invalid_vectors(self):
        duplicate_contexts = np.ones((2, 4))
        for vectors in (duplicate_contexts,):
            with self.subTest(shape=vectors.shape):
                result = fit_context_subspace(vectors)
                self.assertEqual(result["status"], "unavailable")
        full_rank_2d = fit_context_subspace(np.asarray([[0, 0], [1, 0], [0, 1]]))
        self.assertEqual(full_rank_2d["rank"], 2)
        self.assertEqual(full_rank_2d["dimension"], 2)
        self.assertEqual(full_rank_2d["status"], "unavailable")
        rank_deficient_contexts = fit_context_subspace(np.asarray([[0, 0], [1, 0], [2, 0]]))
        self.assertEqual(rank_deficient_contexts["rank"], 1)
        self.assertEqual(rank_deficient_contexts["dimension"], 2)
        self.assertEqual(rank_deficient_contexts["status"], "available")
        for vectors in (np.ones(4), np.asarray([[0, 0], [1, np.nan]]), np.empty((0, 4))):
            with self.subTest(vectors=vectors), self.assertRaises(ValueError):
                fit_context_subspace(vectors)

    def test_seeded_direction_is_normal_reprojected_and_orthogonal(self):
        contexts = np.asarray([[3, -2, 7, 4], [4, -2, 7, 4], [5, -2, 7, 4]])
        first = explore_random_orthogonal_ray(
            ["first context", "second context", "third context"], ["candidate"],
            FixedEncoder({"first context": contexts[0], "second context": contexts[1],
                          "third context": contexts[2], "candidate": [3, -2, 8, 4]}),
            seed=51,
        )
        second = explore_random_orthogonal_ray(
            ["first context", "second context", "third context"], ["candidate"],
            FixedEncoder({"first context": contexts[0], "second context": contexts[1],
                          "third context": contexts[2], "candidate": [3, -2, 8, 4]}),
            seed=51,
        )

        direction = np.asarray(first["rays"][0]["direction"])
        basis = np.asarray(first["context_subspace"]["basis"])
        self.assertEqual(first["rays"], second["rays"])
        self.assertAlmostEqual(np.linalg.norm(direction), 1.0, places=12)
        self.assertLessEqual(float(np.max(np.abs(basis @ direction))), 1e-12)

    def test_different_seeds_yield_different_directions(self):
        contexts = {"one": [0, 0, 0, 0], "two": [1, 0, 0, 0], "three": [2, 0, 0, 0]}
        directions = [explore_random_orthogonal_ray(
            ["one", "two", "three"], [], FixedEncoder(contexts), seed=seed,
        )["rays"][0]["direction"] for seed in (2, 17)]
        self.assertFalse(np.allclose(directions[0], directions[1]))

    def test_ray_count_and_combined_clusters_keep_per_ray_associations(self):
        vectors = {"one": [0, 0, 0, 0], "two": [1, 0, 0, 0], "three": [2, 0, 0, 0],
                   "north": [0, 0, 2, 0], "north again": [0, 0, 4, 0], "south": [0, 0, -2, 0]}
        result = explore_random_orthogonal_ray(
            ["one", "two", "three"], ["north", "north again", "south"], FixedEncoder(vectors),
            seed=5, ray_count=3,
        )
        self.assertEqual(len(result["rays"]), 3)
        self.assertTrue(result["combined_clusters"])
        self.assertTrue(all("selected_clusters" in ray for ray in result["rays"]))
        memberships = {}
        for ray_index, ray in enumerate(result["rays"]):
            for cluster in ray["selected_clusters"]:
                for hit in cluster["hits"]:
                    memberships.setdefault((cluster["cluster_id"], hit["corpus_index"]), set()).add(ray_index)
        for group in result["combined_clusters"]:
            with self.subTest(group=group):
                self.assertTrue(group["cluster_id"])
                self.assertIsInstance(group["ray_index"], int)
                self.assertTrue(group["ray_indices"])
                self.assertIn(group["ray_index"], group["ray_indices"])
                self.assertTrue(group["hits"])
                corpus_indices = [hit["corpus_index"] for hit in group["hits"]]
                self.assertEqual(len(corpus_indices), len(set(corpus_indices)))
                expected_ray_indices = set()
                for hit in group["hits"]:
                    expected_ray_indices.update(memberships[(group["cluster_id"], hit["corpus_index"])])
                self.assertEqual(set(group["ray_indices"]), expected_ray_indices)

    def test_empty_and_excluded_corpus_report_unavailable_without_hits(self):
        contexts = {"one": [0, 0, 0, 0], "two": [1, 0, 0, 0], "three": [2, 0, 0, 0], "row": [0, 1, 0, 0]}
        for corpus, kwargs in (([], {}), (["row"], {"exclude": ["row"]})):
            with self.subTest(corpus=corpus, kwargs=kwargs):
                result = explore_random_orthogonal_ray(
                    ["one", "two", "three"], corpus, FixedEncoder(contexts), seed=2, **kwargs,
                )
                self.assertEqual(result["search_status"], "unavailable")
                self.assertTrue(all(not ray["hits"] for ray in result["rays"]))

    def test_encoded_nonfinite_vectors_and_invalid_seed_are_rejected(self):
        with self.assertRaises(ValueError):
            explore_random_orthogonal_ray(
                ["one", "two"], [], FixedEncoder({"one": [0, 0], "two": [float("inf"), 1]}), seed=1,
            )
        vectors = {"one": [0, 0, 0], "two": [1, 0, 0]}
        for seed in (-1, 1.5, True, "x"):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                explore_random_orthogonal_ray(["one", "two"], [], FixedEncoder(vectors), seed=seed)

    def test_exclusion_preserves_original_corpus_index(self):
        rows = ["skip", "keep"]
        vectors = {"one": [0, 0, 0], "two": [1, 0, 0], "skip": [0, 1, 0], "keep": [0, 2, 0]}
        result = explore_random_orthogonal_ray(["one", "two"], rows, FixedEncoder(vectors), seed=3, exclude=["skip"])
        self.assertTrue(all(hit["corpus_index"] == 1 for ray in result["rays"] for hit in ray["hits"]))

    def test_geometry_uses_origin_ray_start_and_reports_actual_distances(self):
        origin = np.asarray([2.0, -1.0, 5.0])
        direction = np.asarray([0.0, 0.0, 1.0])
        vectors = np.asarray([origin + [0, 0, 2], origin + [3, 0, 2], origin + [0, 0, -2]])
        rows = ray_geometry(vectors, origin, direction)

        self.assertEqual([row["axial_distance"] for row in rows], [2.0, 2.0, -2.0])
        self.assertEqual([row["radial_distance"] for row in rows], [0.0, 3.0, 0.0])
        self.assertEqual([row["ray_distance"] for row in rows], [0.0, 3.0, 2.0])
        self.assertEqual([row["behind_origin"] for row in rows], [False, False, True])
        self.assertEqual([row["in_tube"] for row in rows], [True, False, False])

    def test_no_tube_hit_falls_back_by_distance_then_index_and_keeps_provenance(self):
        corpus = [{"a": "A0", "b": "B0", "line": "first"},
                  {"a": "A1", "b": "B1", "line": "second"}]
        vectors = {
            "ctx one": [0, 0, 0, 0], "ctx two": [1, 0, 0, 0],
            "ctx three": [2, 0, 0, 0],
            "A0とB0。first": [0, 3, 1, 0], "A1とB1。second": [0, -3, 1, 0],
        }
        encoder = FixedEncoder(vectors)
        result = explore_random_orthogonal_ray(
            ["ctx one", "ctx two", "ctx three"], corpus, encoder,
            seed=19, radius=0.01,
        )

        ray = result["rays"][0]
        self.assertEqual(ray["fallback_reason"], "no_tube_hits")
        self.assertEqual(ray["hits"][0]["corpus_index"], 0)
        self.assertEqual(ray["hits"][0]["a"], "A0")
        self.assertEqual(ray["hits"][0]["b"], "B0")
        self.assertEqual(ray["hits"][0]["line"], "first")
        self.assertAlmostEqual(ray["hits"][0]["ray_distance"], np.linalg.norm(
            np.asarray(vectors["A0とB0。first"]) - np.asarray(ray["origin"])
            - max(ray["hits"][0]["axial_distance"], 0) * np.asarray(ray["direction"])
        ))
        json.dumps(result, ensure_ascii=False, allow_nan=False)

    def test_level_maps_to_radius_cosine_and_tone_bands_and_override_wins(self):
        corpus = ["row"]
        vectors = {"ctx one": [0, 0, 0], "ctx two": [1, 0, 0], "ctx three": [2, 0, 0], "row": [0, 1, 1]}
        for level in (0, 25, 60, 75, 100):
            with self.subTest(level=level):
                result = explore_random_orthogonal_ray(
                    ["ctx one", "ctx two", "ctx three"], corpus, FixedEncoder(vectors), level=level,
                )
                params = result["params"]
                strength = level / 100
                self.assertEqual(params["level"], level)
                self.assertAlmostEqual(params["effective_radius"], 0.25 * (0.5 + strength))
                self.assertAlmostEqual(params["cosine_threshold"], 0.85 - 0.35 * strength)
                self.assertIsInstance(params["tone_band"], str)
                self.assertTrue(params["tone_band"])
        overridden = explore_random_orthogonal_ray(
            ["ctx one", "ctx two", "ctx three"], corpus, FixedEncoder(vectors),
            level=60, radius=0.7, cosine_threshold=0.9,
        )
        self.assertEqual(overridden["params"]["effective_radius"], 0.7)
        self.assertEqual(overridden["params"]["cosine_threshold"], 0.9)

    def test_level_requires_an_integer_between_zero_and_one_hundred(self):
        vectors = {"ctx one": [0, 0, 0], "ctx two": [1, 0, 0], "ctx three": [2, 0, 0], "row": [0, 1, 1]}
        for level in (-1, 101, 1.5, True, None, "60"):
            with self.subTest(level=level), self.assertRaises(ValueError):
                explore_random_orthogonal_ray(
                    ["ctx one", "ctx two", "ctx three"], ["row"], FixedEncoder(vectors), level=level,
                )

    def test_in_plane_rows_remain_retrieval_candidates_and_residual_clusters_are_deterministic(self):
        vectors = {"ctx one": [0, 0, 0, 0], "ctx two": [1, 0, 0, 0], "ctx three": [2, 0, 0, 0],
                   "plane one": [0.5, 0, 0, 0], "north one": [0, 0, 2, 0],
                   "north two": [0, 0, 4, 0], "south": [0, 0, -2, 0]}
        texts = ["plane one", "north one", "north two", "south"]
        results = [explore_random_orthogonal_ray(
            ["ctx one", "ctx two", "ctx three"], texts, FixedEncoder(vectors),
            seed=5, level=60, cosine_threshold=0.9,
        ) for _ in range(2)]

        self.assertEqual(results[0]["rays"], results[1]["rays"])
        self.assertTrue(any(hit["line"] == "plane one" for ray in results[0]["rays"] for hit in ray["hits"]))
        self.assertEqual(results[0]["params"]["encoding_path"], "encode_passage")


if __name__ == "__main__":
    unittest.main()
