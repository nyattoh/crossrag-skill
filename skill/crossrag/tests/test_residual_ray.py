"""Synthetic contracts for affine residual clusters and corpus-only rays."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from residual_ray import (  # noqa: E402
    affine_residuals,
    cluster_direction,
    cluster_residual_directions,
    explore_residual_ray,
    fit_affine_plane,
)
from third_concept import ConceptEncoder, corpus_id  # noqa: E402


class FixedEncoder:
    name = "fixed"
    model_id = "fixed-residual-test"
    prefix = None

    def __init__(self, vectors):
        self.vectors = {text: np.asarray(vector, dtype=float) for text, vector in vectors.items()}
        self.calls = []

    def encode_query(self, texts):
        raise AssertionError("residual rays must only use encode_passage")

    def encode_passage(self, texts):
        self.calls.append(("passage", list(texts)))
        return np.asarray([self.vectors[text] for text in texts])


def fixture(points, rows=None, anchors=None):
    anchors = anchors if anchors is not None else ([1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0])
    rows = rows if rows is not None else [f"source-{index}" for index in range(len(points))]
    vectors = dict(zip(("alpha", "beta", "source article"), anchors))
    for row, point in zip(rows, points):
        if isinstance(row, str):
            text = row
        elif isinstance(row, dict):
            text = row["line"] if "a" not in row else f"{row['a']}と{row['b']}。{row['line']}"
        else:
            a, b, line = row
            text = f"{a}と{b}。{line}"
        vectors[text] = point
    vectors.update(zip(("alpha", "beta", "source article"), anchors))
    return rows, FixedEncoder(vectors)


def explore(points, *, rows=None, anchors=None, **kwargs):
    corpus, encoder = fixture(points, rows, anchors)
    return explore_residual_ray("alpha", "beta", "source article", corpus, encoder, **kwargs)


class AffinePlaneTest(unittest.TestCase):
    def test_known_residual_uses_nonzero_origin(self):
        c = np.array([7, -2, 3, 4], dtype=float)
        plane = fit_affine_plane(c + [1, 0, 0, 0], c + [0, 1, 0, 0], c)
        self.assertEqual(plane["status"], "available")
        self.assertEqual(plane["dimension"], 4)
        self.assertEqual(plane["origin"], c.tolist())
        self.assertEqual(len(plane["basis"]), 2)
        np.testing.assert_allclose(plane["singular_values"], [1, 1], atol=1e-12)
        np.testing.assert_allclose(affine_residuals(plane, [c + [3, 4, 5, 6]]), [[0, 0, 5, 6]], atol=1e-12)

    def test_non_axis_plane_is_orthogonal_idempotent_and_translation_invariant(self):
        c = np.array([7, -2, 3, 4], dtype=float)
        first, second = np.array([1, 1, 0, 0]), np.array([0, 1, 1, 0])
        x = c + [3, 4, 5, 6]
        plane = fit_affine_plane(c + first, c + second, c)
        residual = affine_residuals(plane, [x])[0]
        np.testing.assert_allclose([residual @ first, residual @ second], 0, atol=1e-12)
        np.testing.assert_allclose(affine_residuals(plane, [c + residual])[0], residual, atol=1e-12)
        np.testing.assert_allclose(affine_residuals(plane, [x + first])[0], residual, atol=1e-12)
        shift = np.array([-4, 9, 1, 2])
        shifted = fit_affine_plane(c + first + shift, c + second + shift, c + shift)
        np.testing.assert_allclose(affine_residuals(shifted, [x + shift])[0], residual, atol=1e-12)

    def test_unavailable_planes_have_explicit_reasons(self):
        cases = [
            ([1, 0], [0, 1], [0, 0], "embedding_dimension_below_three"),
            ([2, 3, 4], [2, 3, 4], [2, 3, 4], "source_anchors_coincide"),
            ([1, 0, 0], [2, 0, 0], [0, 0, 0], "plane_rank_deficient"),
            ([0, 0, 0], [0, 1, 0], [0, 0, 0], "plane_rank_deficient"),
            ([1, 0, 0], [0, 1e-6, 0], [0, 0, 0], "plane_rank_deficient"),
            ([1e-12, 0, 0], [0, 0, 0], [0, 0, 0], "source_anchors_coincide"),
        ]
        for a, b, c, reason in cases:
            with self.subTest(reason=reason, a=a):
                plane = fit_affine_plane(a, b, c)
                self.assertEqual(plane["status"], "unavailable")
                self.assertEqual(plane["reason"], reason)
                self.assertNotIn("basis", plane)
                with self.assertRaises(ValueError):
                    affine_residuals(plane, [c])

    def test_invalid_shapes_and_nonfinite_anchors_are_errors(self):
        for a in ([1, 2], [[1, 0, 0]], [1, np.nan, 0], [1, 0, np.inf]):
            with self.subTest(a=a), self.assertRaises(ValueError):
                fit_affine_plane(a, [0, 1, 0], [0, 0, 0])

    def test_unrepresentable_singular_values_raise_value_error(self):
        a = np.full(8, 1e308)
        b = np.array([1e308, -1e308] * 4)
        c = np.zeros(8)
        with self.assertRaisesRegex(ValueError, "plane anchors must be finite and of representable scale"):
            fit_affine_plane(a, b, c)

    def test_residual_input_validation_and_empty_matrix(self):
        plane = fit_affine_plane([1, 0, 0], [0, 1, 0], [0, 0, 0])
        self.assertEqual(affine_residuals(plane, np.empty((0, 3))).shape, (0, 3))
        for vectors in ([[1, 2]], [1, 2, 3], [[1, 0, np.nan]], [[1, 0, np.inf]]):
            with self.subTest(vectors=vectors), self.assertRaises(ValueError):
                affine_residuals(plane, vectors)


class ResidualClusterTest(unittest.TestCase):
    def test_duplicate_vectors_cluster_at_cosine_one_across_seeds(self):
        split_seeds = []
        for seed in range(30):
            vector = np.random.RandomState(seed).normal(size=16)
            result = cluster_residual_directions(
                np.tile(vector, (3, 1)), cosine_threshold=1.0, min_residual_norm=1e-9,
            )
            expected = [{"seed_index": 0, "member_indices": [0, 1, 2]}]
            if result["clusters"] != expected:
                split_seeds.append(seed)
        self.assertEqual(split_seeds, [])

    def test_equal_directions_join_and_opposites_stay_separate(self):
        result = cluster_residual_directions(
            [[0, 0, 2, 0], [0, 0, 7, 0], [0, 0, -3, 0], [0, 0, 0, 0], [0, 0, 1e-9, 0]],
            cosine_threshold=1.0, min_residual_norm=1e-9,
        )
        self.assertEqual(result, {
            "clusters": [{"seed_index": 0, "member_indices": [0, 1]}, {"seed_index": 2, "member_indices": [2]}],
            "unusable_indices": [3, 4],
        })

    def test_membership_uses_seed_not_transitive_links_or_updated_mean(self):
        angles = np.deg2rad([0, 40, 70])
        residuals = np.column_stack([np.cos(angles), np.sin(angles)])
        result = cluster_residual_directions(residuals, cosine_threshold=0.7, min_residual_norm=1e-9)
        self.assertEqual(result["clusters"], [
            {"seed_index": 0, "member_indices": [0, 1]}, {"seed_index": 2, "member_indices": [2]},
        ])
        reordered = cluster_residual_directions(residuals[[1, 0, 2]], cosine_threshold=0.7, min_residual_norm=1e-9)
        self.assertEqual(reordered["clusters"], [{"seed_index": 0, "member_indices": [0, 1, 2]}])

    def test_cluster_cosine_boundary_is_inclusive(self):
        result = cluster_residual_directions([[1, 0], [0.6, 0.8]], cosine_threshold=0.6, min_residual_norm=1e-9)
        self.assertEqual(result["clusters"][0]["member_indices"], [0, 1])

    def test_raw_mean_direction_coherence_and_repeat_are_deterministic(self):
        residuals = np.array([[0, 0, 4, 0], [0, 0, 1, 1]], dtype=float)
        summary = cluster_direction(residuals)
        mean = residuals.mean(axis=0)
        np.testing.assert_allclose(summary["mean_residual"], mean, atol=1e-12)
        np.testing.assert_allclose(summary["direction"], mean / np.linalg.norm(mean), atol=1e-12)
        units = residuals / np.linalg.norm(residuals, axis=1)[:, None]
        unit_mean = units.mean(axis=0)
        self.assertGreater(np.linalg.norm(np.asarray(summary["direction"]) - unit_mean / np.linalg.norm(unit_mean)), 0.1)
        self.assertAlmostEqual(summary["resultant_norm"], np.sqrt(26))
        self.assertAlmostEqual(summary["coherence"], np.sqrt(26) / (4 + np.sqrt(2)))
        self.assertEqual(summary["member_count"], 2)
        self.assertEqual(summary["status"], "available")
        self.assertEqual(summary, cluster_direction(residuals))
        first = cluster_residual_directions(residuals, cosine_threshold=0.7, min_residual_norm=1e-9)
        self.assertEqual(first, cluster_residual_directions(residuals, cosine_threshold=0.7, min_residual_norm=1e-9))

    def test_permutation_preserves_well_separated_source_member_sets(self):
        residuals = np.array([[1, 0], [2, 0.1], [-1, 0], [-2, -0.1], [0, 1], [0.1, 2]])
        permutation = np.array([5, 3, 1, 4, 0, 2])
        first = cluster_residual_directions(residuals, cosine_threshold=0.9, min_residual_norm=1e-9)
        second = cluster_residual_directions(residuals[permutation], cosine_threshold=0.9, min_residual_norm=1e-9)
        self.assertEqual(
            {frozenset(cluster["member_indices"]) for cluster in first["clusters"]},
            {frozenset(int(permutation[i]) for i in cluster["member_indices"]) for cluster in second["clusters"]},
        )

    def test_zero_resultant_is_not_a_direction(self):
        for residuals in ([[1, 0], [-1, 0]], [[0, 0], [0, 0]], [[1, 0], [-1 + 1e-13, 0]]):
            with self.subTest(residuals=residuals):
                result = cluster_direction(residuals)
                self.assertEqual(result["status"], "zero_resultant")
                self.assertIsNone(result["direction"])
                self.assertGreaterEqual(result["coherence"], 0)
                self.assertLessEqual(result["coherence"], 1)

    def test_empty_clusters_and_direction_are_serialisable(self):
        result = cluster_residual_directions(np.empty((0, 4)), cosine_threshold=0.7, min_residual_norm=1e-9)
        self.assertEqual(result, {"clusters": [], "unusable_indices": []})
        summary = cluster_direction(np.empty((0, 4)))
        self.assertEqual(summary["status"], "zero_resultant")
        self.assertEqual(summary["member_count"], 0)
        json.dumps(summary, allow_nan=False)

    def test_cluster_helpers_reject_invalid_parameters_and_vectors(self):
        for threshold in (0, -1, 1.1, np.inf, np.nan, None, "x"):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                cluster_residual_directions([[1, 0]], cosine_threshold=threshold, min_residual_norm=1e-9)
        for minimum in (0, -1, np.inf, np.nan, None):
            with self.subTest(minimum=minimum), self.assertRaises(ValueError):
                cluster_residual_directions([[1, 0]], cosine_threshold=0.7, min_residual_norm=minimum)
        for values in ([1, 2], [[1, np.nan]], [[1, np.inf]]):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    cluster_residual_directions(values, cosine_threshold=0.7, min_residual_norm=1e-9)
                with self.assertRaises(ValueError):
                    cluster_direction(values)


class ResidualExploreTest(unittest.TestCase):
    def test_a_b_c_rows_hit_every_ray_at_zero_start_across_non_axis_planes(self):
        missing = []
        distances = []
        for seed in range(60):
            rng = np.random.RandomState(seed)
            axes, _ = np.linalg.qr(rng.normal(size=(8, 8)))
            c = rng.normal(size=8)
            a, b = c + axes[:, 0], c + axes[:, 1]
            points = [a, b, c, c + axes[:, 2], c + 2 * axes[:, 2], c + axes[:, 3], c + 2 * axes[:, 3]]
            result = explore(
                points, rows=["alpha", "beta", "source article", "first", "second", "third", "fourth"],
                anchors=(a, b, c), start_distance=0, radius=100,
            )
            self.assertEqual(len(result["rays"]), 2, f"seed={seed}")
            for ray in result["rays"]:
                hits = {hit["line"]: hit for hit in ray["hits"]}
                for text, point in (("alpha", a), ("beta", b), ("source article", c)):
                    if text not in hits:
                        missing.append((seed, ray["ray_id"], text))
                        continue
                    hit = hits[text]
                    distances.append(hit["axial_distance"])
                    self.assertAlmostEqual(hit["radial_distance"], float(np.linalg.norm(point - c)), places=12)
            json.dumps(result, allow_nan=False)
        self.assertEqual(missing, [])
        self.assertEqual(distances, [0.0] * (60 * 2 * 3))

    def test_unrepresentable_plane_raises_instead_of_returning_nonfinite_result(self):
        anchors = (np.full(8, 1e308), np.array([1e308, -1e308] * 4), np.zeros(8))
        with self.assertRaisesRegex(ValueError, "plane anchors must be finite and of representable scale"):
            explore([], anchors=anchors)

    def test_unrelated_labels_cluster_by_residual_direction(self):
        result = explore([[0, 0, 2, 0], [0, 0, 4, 0], [0, 0, -2, 0]], rows=["teapot", "galactic dust", "violin"])
        self.assertEqual(result["status"], "hits_found")
        self.assertEqual(result["clustering"]["cluster_count"], 2)
        self.assertEqual(result["clustering"]["dropped_small_cluster_count"], 1)
        self.assertEqual(result["rays"][0]["member_corpus_indices"], [0, 1])
        self.assertEqual([hit["line"] for hit in result["rays"][0]["hits"]], ["teapot", "galactic dust"])

    def test_source_rows_geometry_boundaries_and_membership_flags(self):
        c = np.array([7, -2, 3, 4], dtype=float)
        points = c + np.array([[0.25, 0, 0.5, 0], [0, 0, 0.5, 0], [0, 0, 0, 0], [0, 0, 0.5, 0.1]])
        rows = [("copper", "fog", "boundary"), {"a": "sand", "b": "glass", "line": "axis"}, {"line": "in plane"}, "outside cluster"]
        result = explore(
            points, rows=rows, anchors=(c + [1, 0, 0, 0], c + [0, 1, 0, 0], c),
            radius=0.25, start_distance=0, cluster_cosine=1.0,
        )
        ray = result["rays"][0]
        self.assertEqual(ray["member_corpus_indices"], [0, 1])
        self.assertEqual(result["clustering"]["unusable_corpus_indices"], [2])
        self.assertEqual([hit["corpus_index"] for hit in ray["hits"]], [1, 2, 3, 0])
        expected_rows = [("copper", "fog", "boundary"), ("sand", "glass", "axis"), (None, None, "in plane"), (None, None, "outside cluster")]
        for hit in ray["hits"]:
            index = hit["corpus_index"]
            self.assertEqual((hit["a"], hit["b"], hit["line"]), expected_rows[index])
            u = points[index] - c
            direction = np.asarray(ray["direction"])
            t = float(u @ direction)
            radial = float(np.linalg.norm(u - t * direction))
            self.assertAlmostEqual(hit["axial_distance"], t, places=12)
            self.assertAlmostEqual(hit["radial_distance"], radial, places=12)
            self.assertGreaterEqual(t, 0)
            self.assertLessEqual(radial, 0.25)
            self.assertEqual(hit["in_cluster"], index in [0, 1])
        self.assertEqual(ray["hits"][-1]["radial_distance"], 0.25)
        boundary = explore([[0.25, 0, 0.5, 0], [0, 0, 1, 0]], radius=0.25, start_distance=0.5)
        self.assertEqual(boundary["rays"][0]["hits"][-1]["axial_distance"], 0.5)
        self.assertEqual(boundary["rays"][0]["hits"][-1]["radial_distance"], 0.25)

    def test_a1_input_text_and_high_anchor_cosine_are_kept(self):
        rows = ["alpha appears here", "near anchor", "source article", "other"]
        points = [[0, 0, 1, 0], [10, 0, 1, 0], [0, 0, 0, 0], [0, 0, 3, 0]]
        self.assertGreaterEqual(10 / np.sqrt(101), 0.98)
        result = explore(points, rows=rows, radius=10, start_distance=0)
        self.assertEqual(result["clustering"]["excluded_count"], 0)
        self.assertEqual(result["rays"][0]["member_corpus_indices"], [0, 1, 3])
        self.assertEqual(result["clustering"]["unusable_corpus_indices"], [2])
        self.assertEqual({hit["corpus_index"] for hit in result["rays"][0]["hits"]}, {0, 1, 2, 3})

    def test_a1_only_explicit_exclude_removes_rows_and_preserves_indices(self):
        rows = ["alpha appears here", ("SKIP ME", "fog", "paired source"), "near anchor", "source article", "other", "another"]
        points = [[0, 0, 1, 0], [0, 0, 1, 0], [10, 0, 1, 0], [0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 4, 0]]
        result = explore(points, rows=rows, radius=10, start_distance=0, exclude=["ＡＬＰＨＡ", "skip me", "near anchor"])
        self.assertEqual(result["clustering"]["excluded_count"], 3)
        self.assertEqual(result["clustering"]["candidate_count"], 3)
        self.assertEqual(result["rays"][0]["seed_corpus_index"], 4)
        self.assertEqual(result["rays"][0]["member_corpus_indices"], [4, 5])
        self.assertEqual([hit["corpus_index"] for hit in result["rays"][0]["hits"]], [3, 4, 5])

    def test_exclude_terms_apply_to_each_paired_row_value(self):
        rows = [("skip", "fog", "first"), ("sand", "SKIP", "second"), ("sand", "fog", "skip third"), "keep one", "keep two"]
        result = explore([[0, 0, 1, 0]] * 5, rows=rows, exclude=["skip"])
        self.assertEqual(result["clustering"]["excluded_count"], 3)
        self.assertEqual(result["rays"][0]["member_corpus_indices"], [3, 4])

    def test_nonzero_residual_below_cutoff_stays_eligible_for_hits(self):
        result = explore([[0, 0, 1, 0], [0, 0, 2, 0], [0, 0, 0.01, 0]], min_residual_norm=0.1, start_distance=0)
        self.assertEqual(result["clustering"]["usable_residual_count"], 2)
        self.assertEqual(result["clustering"]["unusable_corpus_indices"], [2])
        self.assertEqual(result["rays"][0]["hits"][2]["corpus_index"], 2)
        self.assertFalse(result["rays"][0]["hits"][2]["in_cluster"])

    def test_a_candidate_can_be_retrieved_under_multiple_rays(self):
        result = explore([[0, 0, 1, 0], [0, 0, 2, 0], [0, 0, 0, 1], [0, 0, 0, 2], [0, 0, 0.5, 0.5]], radius=1, cluster_cosine=0.9)
        self.assertEqual(len(result["rays"]), 2)
        self.assertTrue(all(4 in [hit["corpus_index"] for hit in ray["hits"]] for ray in result["rays"]))

    def test_unavailable_plane_precedes_empty_corpus(self):
        cases = [
            (([1, 0], [0, 1], [0, 0]), "embedding_dimension_below_three"),
            (([0, 0, 0], [0, 0, 0], [0, 0, 0]), "source_anchors_coincide"),
            (([1, 0, 0], [2, 0, 0], [0, 0, 0]), "plane_rank_deficient"),
            (([0, 0, 0], [0, 1, 0], [0, 0, 0]), "plane_rank_deficient"),
        ]
        for anchors, reason in cases:
            with self.subTest(reason=reason):
                result = explore([], anchors=anchors)
                self.assertEqual(result["status"], "plane_unavailable")
                self.assertEqual(result["plane"]["reason"], reason)
                self.assertEqual(result["rays"], [])

    def test_empty_corpus_and_all_excluded_have_no_clusters(self):
        for points, params in (([], {}), ([[0, 0, 1, 0]], {"exclude": ["source-"]})):
            with self.subTest(points=points):
                result = explore(points, **params)
                self.assertEqual(result["status"], "no_residual_clusters")
                self.assertEqual(result["rays"], [])
                self.assertEqual(result["clustering"]["usable_residual_count"], 0)

    def test_in_plane_candidates_have_no_usable_residuals(self):
        result = explore([[1, 2, 0, 0], [-3, 4, 0, 0]])
        self.assertEqual(result["status"], "no_residual_clusters")
        self.assertEqual(result["clustering"]["unusable_corpus_indices"], [0, 1])
        self.assertEqual(result["clustering"]["cluster_count"], 0)
        self.assertEqual(result["rays"], [])

    def test_small_clusters_do_not_become_rays(self):
        result = explore([[0, 0, 1, 0], [0, 0, -1, 0]])
        self.assertEqual(result["status"], "no_residual_clusters")
        self.assertEqual(result["clustering"]["dropped_small_cluster_count"], 2)
        self.assertEqual(result["rays"], [])

    def test_rays_without_hits_report_no_hits(self):
        result = explore([[2, 0, 1, 0], [3, 0, 1, 0]], radius=0.25)
        self.assertEqual(result["status"], "no_hits")
        self.assertEqual(len(result["rays"]), 1)
        self.assertEqual(result["rays"][0]["hit_count"], 0)
        self.assertEqual(result["rays"][0]["hits"], [])

    def test_max_rays_orders_by_size_then_original_seed_and_reports_truncation(self):
        points = [[0, 0, 1, 0], [0, 0, 2, 0], [0, 0, 0, 1], [0, 0, 0, 2], [0, 0, 0, 3], [0, 0, -1, 0], [0, 0, -2, 0]]
        result = explore(points, max_rays=2)
        self.assertEqual([ray["seed_corpus_index"] for ray in result["rays"]], [2, 0])
        self.assertEqual([ray["ray_id"] for ray in result["rays"]], [0, 1])
        self.assertEqual(result["clustering"]["truncated_ray_count"], 1)
        unlimited = explore(points, max_rays=None)
        self.assertEqual([ray["seed_corpus_index"] for ray in unlimited["rays"]], [2, 0, 5])
        self.assertEqual(unlimited["clustering"]["truncated_ray_count"], 0)

    def test_schema_single_encoding_path_and_json_preserve_raw_coordinates(self):
        rows, encoder = fixture([[0, 0, 1, 0], [0, 0, 2, 0]], ["first", ("sand", "fog", "second")])
        encoder.prefix = {"query": "query: ", "passage": "passage: "}
        result = explore_residual_ray("alpha", "beta", "source article", rows, encoder)
        self.assertEqual(encoder.calls, [("passage", ["alpha", "beta", "source article", "first", "sandとfog。second"])])
        self.assertEqual(result["mode"], "normal-ray")
        self.assertEqual(result["method"], "affine_residual_cluster")
        self.assertEqual(result["input"], {"A": "alpha", "B": "beta", "C": "source article"})
        self.assertEqual(result["encoder"], {"name": encoder.name, "model_id": encoder.model_id, "prefix": encoder.prefix})
        self.assertEqual(result["corpus_id"], corpus_id([(None, None, "first"), ("sand", "fog", "second")]))
        self.assertEqual(result["corpus_size"], 2)
        self.assertEqual(result["params"], {
            "coordinates": "raw_encoder_embedding", "origin": "C", "plane": "C + span(A-C, B-C)",
            "encoding_path": "encode_passage",
            "radius": 0.25, "start_distance": 0.25, "cluster_cosine": 0.7, "min_cluster_size": 2,
            "min_residual_norm": 1e-9, "max_rays": 5, "exclude": [],
        })
        self.assertEqual(result["clustering"], {
            "candidate_count": 2, "excluded_count": 0, "usable_residual_count": 2,
            "unusable_corpus_indices": [], "cluster_count": 1, "dropped_small_cluster_count": 0,
            "zero_resultant_cluster_count": 0, "truncated_ray_count": 0,
        })
        self.assertEqual(result["warnings"], [])
        self.assertIn("no semantic", result["disclaimer"])
        self.assertEqual(json.loads(json.dumps(result, allow_nan=False)), result)
        self.assertEqual(explore([[0, 0, 1, 0]])["warnings"], [])

    def test_a2_plain_row_equal_to_a_has_same_passage_vector_and_remains_retrievable(self):
        rows, encoder = fixture([[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 2, 0]], ["alpha", "first", "second"])
        encoder.prefix = {"query": "query: ", "passage": "passage: "}
        result = explore_residual_ray("alpha", "beta", "source article", rows, encoder, radius=1, start_distance=0)
        self.assertEqual(encoder.calls, [("passage", ["alpha", "beta", "source article", "alpha", "first", "second"])])
        self.assertEqual(result["params"]["encoding_path"], "encode_passage")
        self.assertEqual(result["clustering"]["unusable_corpus_indices"], [0])
        self.assertEqual(result["clustering"]["excluded_count"], 0)
        self.assertEqual(result["rays"][0]["member_corpus_indices"], [1, 2])
        hit = next(hit for hit in result["rays"][0]["hits"] if hit["corpus_index"] == 0)
        self.assertFalse(hit["in_cluster"])
        self.assertEqual(hit["axial_distance"], 0)
        self.assertEqual(hit["radial_distance"], 1)
        self.assertEqual(result["warnings"], [])

    def test_a2_concept_encoder_normalises_query_prefixed_inputs_to_passage_prefix(self):
        vectors = {
            "query: alpha": [1, 0, 0, 0],
            "passage: alpha": [1, 0, 0, 0],
            "passage: beta": [0, 1, 0, 0],
            "passage: source article": [0, 0, 0, 0],
            "passage: first": [0, 0, 1, 0],
            "passage: second": [0, 0, 2, 0],
        }

        class RecordingBackend:
            def __init__(self):
                self.calls = []

            def encode(self, texts, **kwargs):
                self.calls.append(list(texts))
                return np.asarray([vectors[text] for text in texts], dtype=float)

        backend = RecordingBackend()
        encoder = ConceptEncoder(
            backend, "e5-test", "intfloat/test-e5", {"query": "query: ", "passage": "passage: "},
        )
        result = explore_residual_ray(
            "query: alpha", "beta", "source article", ["passage: query: alpha", "first", "second"],
            encoder, radius=1, start_distance=0,
        )

        self.assertEqual(backend.calls, [[
            "passage: alpha", "passage: beta", "passage: source article",
            "passage: alpha", "passage: first", "passage: second",
        ]])
        self.assertEqual(result["params"]["encoding_path"], "encode_passage")
        self.assertIn("passage: query: alpha", [hit["line"] for ray in result["rays"] for hit in ray["hits"]])

    def test_float_plane_member_axial_distance_is_clamped_to_zero(self):
        rng = np.random.default_rng(7)
        a, b, c = rng.normal(size=(3, 16))
        plane = fit_affine_plane(a, b, c)
        basis = np.asarray(plane["basis"])
        residual = rng.normal(size=16)
        residual -= (residual @ basis.T) @ basis
        residual /= np.linalg.norm(residual)
        first = c + 0.4 * (a - c) + residual
        second = c - 0.2 * (b - c) + 2 * residual
        encoder = FixedEncoder({
            "A": a, "B": b, "C": c, "anchor A row": a, "first": first, "second": second,
        })

        result = explore_residual_ray(
            "A", "B", "C", ["anchor A row", "first", "second"], encoder,
            start_distance=0, radius=100,
        )
        anchor_hit = next(
            hit for ray in result["rays"] for hit in ray["hits"] if hit["line"] == "anchor A row"
        )
        self.assertEqual(anchor_hit["axial_distance"], 0.0)

    def test_floats_are_not_rounded(self):
        result = explore([[0, 0, 1.123456789123, 0], [0, 0, 2, 0]])
        self.assertEqual(result["rays"][0]["hits"][0]["axial_distance"], 1.123456789123)

    def test_invalid_parameters_fail_before_any_encoding(self):
        invalid = [
            {"radius": value} for value in (0, -1, np.nan, np.inf, None, "x")
        ] + [
            {"start_distance": value} for value in (-1, np.nan, np.inf, None)
        ] + [
            {"cluster_cosine": value} for value in (0, -1, 1.1, np.nan, np.inf, None)
        ] + [
            {"min_cluster_size": value} for value in (0, -1, 1.5, True, None)
        ] + [
            {"max_rays": value} for value in (0, -1, 1.5, True)
        ] + [
            {"min_residual_norm": value} for value in (0, -1, np.nan, np.inf, None)
        ] + [{"exclude": [""]}, {"exclude": [None]}]
        for params in invalid:
            with self.subTest(params=params):
                rows, encoder = fixture([])
                with self.assertRaises(ValueError):
                    explore_residual_ray("alpha", "beta", "source article", rows, encoder, **params)
                self.assertEqual(encoder.calls, [])

    def test_empty_or_non_string_inputs_fail_before_encoding(self):
        for index in range(3):
            for invalid in ("", "  ", None, 42):
                inputs = ["alpha", "beta", "source article"]
                inputs[index] = invalid
                rows, encoder = fixture([])
                with self.subTest(inputs=inputs), self.assertRaises(ValueError):
                    explore_residual_ray(*inputs, rows, encoder)
                self.assertEqual(encoder.calls, [])

    def test_nonfinite_or_mismatched_encoder_outputs_raise_value_error(self):
        for point in ([0, np.nan, 1, 0], [0, np.inf, 1, 0], [1, 2, 3]):
            with self.subTest(point=point), self.assertRaises(ValueError):
                explore([point])
        rows, encoder = fixture([[0, 0, 1, 0]])
        encoder.vectors["alpha"][0] = np.nan
        with self.assertRaises(ValueError):
            explore_residual_ray("alpha", "beta", "source article", rows, encoder)


if __name__ == "__main__":
    unittest.main()
