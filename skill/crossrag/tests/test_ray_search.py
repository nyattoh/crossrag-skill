"""Synthetic tests for one-ray, corpus-set exploration."""
from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from third_concept import explore_ray, fit_ray_frame, search_ray_sets  # noqa: E402


def pca_reference_vectors(dimension=3):
    rows = []
    for axis, scale in enumerate((3.0, 2.0, 1.0)):
        positive = np.zeros(dimension)
        positive[axis] = scale
        rows.extend((positive, -positive))
    return np.asarray(rows)


class RayFrameTest(unittest.TestCase):
    def test_pca_lift_produces_one_unit_direction_orthogonal_to_anchors(self):
        a = np.eye(5)[0]
        b = np.eye(5)[1]
        frame = fit_ray_frame(a, b, pca_reference_vectors(5))

        self.assertEqual(frame["status"], "available")
        direction = np.asarray(frame["direction"])
        np.testing.assert_allclose(np.linalg.norm(direction), 1.0, atol=1e-12)
        self.assertAlmostEqual(float(np.dot(direction, a)), 0.0, places=12)
        self.assertAlmostEqual(float(np.dot(direction, b)), 0.0, places=12)
        self.assertEqual(len(direction), 5)

    def test_anchor_midpoint_is_the_ray_origin_and_anchor_swap_reverses_ray(self):
        a = np.eye(3)[0]
        b = np.eye(3)[1]
        reference = pca_reference_vectors()
        frame = fit_ray_frame(a, b, reference)
        swapped = fit_ray_frame(b, a, reference)

        np.testing.assert_allclose(frame["origin"], [0.5, 0.5, 0.0], atol=1e-12)
        np.testing.assert_allclose(frame["direction"], -np.asarray(swapped["direction"]), atol=1e-12)

    def test_ray_is_orthogonal_to_the_centered_anchors(self):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        pca_center = np.array([4.0, -2.0, 1.0])
        frame = fit_ray_frame(a, b, pca_reference_vectors() + pca_center)
        direction = np.asarray(frame["direction"])

        self.assertAlmostEqual(float(np.dot(direction, (a - pca_center) / np.linalg.norm(a - pca_center))), 0.0, places=12)
        self.assertAlmostEqual(float(np.dot(direction, (b - pca_center) / np.linalg.norm(b - pca_center))), 0.0, places=12)

    def test_collinear_projected_anchors_report_no_ray(self):
        frame = fit_ray_frame([1, 0, 0], [1, 0, 0], pca_reference_vectors())

        self.assertEqual(frame["status"], "unavailable")
        self.assertEqual(frame["reason"], "projected_anchors_collinear")

    def test_reference_space_with_rank_below_three_reports_no_ray(self):
        references = np.asarray([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]])
        frame = fit_ray_frame([1, 0, 0], [0, 1, 0], references)

        self.assertEqual(frame["status"], "unavailable")
        self.assertEqual(frame["reason"], "reference_rank_below_three")

    def test_weak_third_component_does_not_define_a_ray(self):
        references = np.asarray([
            [3, 0, 0], [-3, 0, 0], [0, 2, 0], [0, -2, 0],
            [0, 0, 1e-4], [0, 0, -1e-4],
        ], dtype=float)
        frame = fit_ray_frame([1, 0, 0], [0, 1, 0], references)

        self.assertEqual(frame["status"], "unavailable")
        self.assertEqual(frame["reason"], "third_pca_component_too_weak")

    def test_near_equal_leading_pca_components_warn_about_orientation_stability(self):
        references = np.asarray([
            [3, 0, 0], [-3, 0, 0], [0, 3, 0], [0, -3, 0],
            [0, 0, 1], [0, 0, -1],
        ], dtype=float)
        frame = fit_ray_frame([1, 0, 0], [0, 1, 0], references)

        self.assertEqual(frame["status"], "available")
        self.assertLess(frame["pca_leading_component_relative_gaps"][0], 0.05)
        self.assertTrue(any("上位3成分" in warning for warning in frame["warnings"]))

    def test_anchor_outside_the_pca_frame_reports_low_projection_energy(self):
        frame = fit_ray_frame([0, 0, 0, 1], [0, 0, 0, 0.5], pca_reference_vectors(4))

        self.assertEqual(frame["status"], "unavailable")
        self.assertEqual(frame["reason"], "anchor_projection_energy_too_low")

    def test_rotated_high_dimensional_frame_still_lifts_an_orthogonal_ray(self):
        rng = np.random.default_rng(19)
        raw = rng.normal(size=(8, 8))
        rotation, _ = np.linalg.qr(raw)
        references = pca_reference_vectors(8) @ rotation.T
        a = np.eye(8)[0] @ rotation.T
        b = np.eye(8)[1] @ rotation.T
        frame = fit_ray_frame(a, b, references)
        direction = np.asarray(frame["direction"])

        self.assertEqual(frame["status"], "available")
        self.assertAlmostEqual(float(np.dot(direction, a)), 0.0, places=12)
        self.assertAlmostEqual(float(np.dot(direction, b)), 0.0, places=12)
        self.assertAlmostEqual(float(np.linalg.norm(direction)), 1.0, places=12)

    def test_no_center_uses_the_uncentered_svd_frame(self):
        frame = fit_ray_frame([1, 0, 0], [0, 1, 0], pca_reference_vectors(), center=False)

        self.assertEqual(frame["method"], "global_corpus_uncentred_pca_3d_lift")
        self.assertEqual(frame["origin_rule"], "anchor_midpoint_at_embedding_origin")


class RaySetSearchTest(unittest.TestCase):
    def setUp(self):
        self.frame = fit_ray_frame([1, 0, 0], [0, 1, 0], pca_reference_vectors())
        self.origin = np.asarray(self.frame["origin"])
        self.direction = np.asarray(self.frame["direction"])

    def test_ray_returns_forward_nearby_items_as_axial_sets(self):
        points = np.asarray([
            self.origin + 0.1 * self.direction,
            self.origin + 0.2 * self.direction,
            self.origin - 0.1 * self.direction,
            self.origin + 0.2 * self.direction + np.array([0.3, 0.0, 0.0]),
        ])
        result = search_ray_sets(
            self.frame, points, ["near-1", "near-2", "behind", "off-ray"],
            radius=0.1, start_distance=0.0, cluster_gap=0.5, min_set_size=2,
        )

        self.assertEqual(result["status"], "sets_found")
        self.assertEqual(len(result["sets"]), 1)
        hit_set = result["sets"][0]
        self.assertEqual(hit_set["count"], 2)
        self.assertEqual({item["label"] for item in hit_set["items"]}, {"near-1", "near-2"})
        self.assertAlmostEqual(hit_set["t_start"], 0.1)
        self.assertAlmostEqual(hit_set["t_end"], 0.2)

    def test_ray_returns_geometric_hits_without_semantic_filter_and_preserves_sources(self):
        points = np.asarray([
            self.origin + 0.1 * self.direction,
            self.origin + 0.2 * self.direction,
        ])
        sources = [
            {"corpus_index": 17, "line": "a deliberately unrelated item", "extra": "kept"},
            {"corpus_index": 23, "line": "another arbitrary item"},
        ]

        result = search_ray_sets(
            self.frame, points, sources,
            radius=0.01, start_distance=0.0, cluster_gap=0.5, min_set_size=2,
        )

        self.assertEqual(result["status"], "sets_found")
        items = result["sets"][0]["items"]
        self.assertEqual([item["label"] for item in items], ["a deliberately unrelated item", "another arbitrary item"])
        self.assertEqual([item["source_index"] for item in items], [17, 23])
        self.assertEqual([item["source"] for item in items], sources)
        self.assertTrue(all(item["radial_distance"] <= result["radius"] for item in items))
        self.assertTrue(all(item["axial_distance"] >= result["start_distance"] for item in items))

    def test_nonzero_center_is_applied_to_candidate_coordinates(self):
        shift = np.array([4.0, -2.0, 1.0])
        a = np.array([1.0, 0.0, 0.0]) + shift
        b = np.array([0.0, 1.0, 0.0]) + shift
        shifted_frame = fit_ray_frame(a, b, pca_reference_vectors() + shift)
        shifted_origin = np.asarray(shifted_frame["origin"])
        shifted_direction = np.asarray(shifted_frame["direction"])
        points = np.asarray([
            shift + shifted_origin + 0.1 * shifted_direction,
            shift + shifted_origin + 0.2 * shifted_direction,
        ])
        result = search_ray_sets(
            shifted_frame, points, ["shifted-1", "shifted-2"],
            radius=0.01, start_distance=0.0, cluster_gap=0.25, min_set_size=2,
        )

        self.assertEqual(result["status"], "sets_found")
        self.assertEqual(result["sets"][0]["count"], 2)
        self.assertLess(result["sets"][0]["mean_radial_distance"], 1e-12)

    def test_nearby_points_in_separate_axial_bands_form_separate_sets(self):
        points = np.asarray([
            self.origin + 0.1 * self.direction,
            self.origin + 0.2 * self.direction,
            self.origin + 0.7 * self.direction,
            self.origin + 0.8 * self.direction,
        ])
        result = search_ray_sets(
            self.frame, points, ["a1", "a2", "b1", "b2"],
            radius=0.05, start_distance=0.0, cluster_gap=0.25, min_set_size=2,
        )

        self.assertEqual([item["count"] for item in result["sets"]], [2, 2])
        self.assertAlmostEqual(result["sets"][0]["t_start"], 0.1)
        self.assertAlmostEqual(result["sets"][1]["t_start"], 0.7)

    def test_connected_axial_neighbours_do_not_split_at_a_fixed_bin_edge(self):
        points = np.asarray([
            self.origin + 0.49 * self.direction,
            self.origin + 0.51 * self.direction,
        ])
        result = search_ray_sets(
            self.frame, points, ["left-edge", "right-edge"],
            radius=0.05, start_distance=0.0, cluster_gap=0.05, min_set_size=2,
        )

        self.assertEqual(len(result["sets"]), 1)
        self.assertEqual(result["sets"][0]["count"], 2)

    def test_start_distance_skips_the_plane_neighbourhood(self):
        points = np.asarray([
            self.origin + 0.1 * self.direction,
            self.origin + 0.2 * self.direction,
            self.origin + 0.6 * self.direction,
            self.origin + 0.7 * self.direction,
        ])
        result = search_ray_sets(
            self.frame, points, ["near-1", "near-2", "far-1", "far-2"],
            radius=0.05, start_distance=0.5, cluster_gap=0.25, min_set_size=2,
        )

        self.assertEqual(len(result["sets"]), 1)
        self.assertEqual({item["label"] for item in result["sets"][0]["items"]}, {"far-1", "far-2"})

    def test_max_sets_keeps_nearest_axial_sets_and_reports_truncation(self):
        points = np.asarray([
            self.origin + t * self.direction
            for t in (0.1, 0.2, 0.8, 0.9, 1.5, 1.6)
        ])
        result = search_ray_sets(
            self.frame, points, [str(index) for index in range(len(points))],
            radius=0.05, start_distance=0.0, cluster_gap=0.25, min_set_size=2, max_sets=2,
        )

        self.assertEqual(len(result["sets"]), 2)
        self.assertEqual(result["truncated_set_count"], 1)
        self.assertAlmostEqual(result["sets"][0]["t_start"], 0.1)

    def test_empty_ray_returns_an_empty_set_without_fabricating_hits(self):
        points = np.asarray([self.origin - 0.1 * self.direction])
        result = search_ray_sets(
            self.frame, points, ["behind"], radius=0.1, start_distance=0.0, cluster_gap=0.5, min_set_size=1,
        )

        self.assertEqual(result["status"], "no_sets")
        self.assertEqual(result["sets"], [])

    def test_empty_candidate_pool_returns_no_sets(self):
        result = search_ray_sets(
            self.frame, np.empty((0, 3)), [], radius=0.1, start_distance=0.0, cluster_gap=0.5, min_set_size=2,
        )

        self.assertEqual(result["status"], "no_sets")
        self.assertEqual(result["near_ray_count"], 0)

    def test_singleton_bins_are_not_reported_as_sets(self):
        points = np.asarray([self.origin + 0.1 * self.direction])
        result = search_ray_sets(
            self.frame, points, ["singleton"], radius=0.1, start_distance=0.0, cluster_gap=0.5, min_set_size=2,
        )

        self.assertEqual(result["status"], "no_sets")
        self.assertEqual(result["sets"], [])

    def test_invalid_ray_search_parameters_are_rejected(self):
        vectors = np.asarray([self.origin + 0.1 * self.direction])
        for parameters in (
            {"radius": 0},
            {"cluster_gap": 0},
            {"start_distance": -0.1},
            {"min_set_size": 0},
        ):
            with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                search_ray_sets(self.frame, vectors, ["x"], **parameters)

    def test_invalid_parameters_are_rejected_even_when_ray_is_unavailable(self):
        unavailable = {"status": "unavailable", "reason": "rank"}
        with self.assertRaises(ValueError):
            search_ray_sets(unavailable, np.empty((0, 3)), [], radius=-1)


class RayCliTest(unittest.TestCase):
    def test_cli_ray_mode_returns_a_structured_empty_result(self):
        command = [
            sys.executable, str(SCRIPTS / "third_concept_cli.py"), "--A", "alpha", "--B", "beta",
            "--corpus", str(ROOT / "tests" / "fixtures" / "synthetic_corpus.json"),
            "--encoder", "hash", "--search-mode", "ray", "--ray-radius", "0.4",
            "--ray-start-distance", "0", "--ray-cluster-gap", "0.5", "--ray-min-set-size", "2",
        ]
        result = json.loads(subprocess.run(command, cwd=ROOT, capture_output=True, check=True).stdout.decode("utf-8"))

        self.assertEqual(result["mode"], "normal-ray")
        self.assertIn(result["status"], {"sets_found", "no_sets", "ray_unavailable"})
        if result["status"] == "ray_unavailable":
            self.assertEqual(result["ray"]["status"], "unavailable")
        else:
            self.assertIn("direction", result["ray"])
        self.assertIn("concept_sets", result)
        self.assertIn("passage_sets", result)

    def test_cli_rejects_invalid_radius_even_when_frame_may_be_unavailable(self):
        command = [
            sys.executable, str(SCRIPTS / "third_concept_cli.py"), "--A", "alpha", "--B", "beta",
            "--corpus", str(ROOT / "tests" / "fixtures" / "synthetic_corpus.json"),
            "--encoder", "hash", "--search-mode", "ray", "--ray-radius", "0",
        ]
        result = subprocess.run(command, cwd=ROOT, capture_output=True, check=False)

        self.assertEqual(result.returncode, 2)
        self.assertIn("ray radius must be finite and positive", result.stderr.decode("utf-8"))

    def test_explore_ray_returns_a_set_of_corpus_passages_near_the_lifted_ray(self):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        origin = (a + b) / 2
        shift = np.array([0.2, -0.1, 0.05])
        rows = []
        vectors = {"A": a, "B": b, "ref_left": a, "ref_right": b}
        reference_pairs = (
            (30 * np.eye(3)[0], -30 * np.eye(3)[0]),
            (20 * np.eye(3)[1], -20 * np.eye(3)[1]),
            (10 * np.eye(3)[2], -10 * np.eye(3)[2]),
            (origin + 0.1 * np.eye(3)[2], -origin - 0.1 * np.eye(3)[2]),
            (origin + 0.2 * np.eye(3)[2], -origin - 0.2 * np.eye(3)[2]),
        )
        for pair_index, pair in enumerate(reference_pairs):
            for side, vector in enumerate(pair):
                line = f"reference-{pair_index}-{side}"
                rows.append(("ref_left", "ref_right", line))
                vectors[f"ref_leftとref_right。{line}"] = vector

        vectors = {key: vector + shift for key, vector in vectors.items()}

        class FixedEncoder:
            name = "fixed-test"
            model_id = "fixed-test"
            prefix = None

            def encode_query(self, texts):
                return np.asarray([vectors[text] for text in texts])

            def encode_passage(self, texts):
                return np.asarray([vectors[text] for text in texts])

        result = explore_ray(
            "A", "B", rows, FixedEncoder(), center=True,
            radius=0.08, start_distance=0.0, cluster_gap=0.25, min_set_size=2,
        )

        self.assertEqual(result["status"], "sets_found")
        np.testing.assert_allclose(result["ray"]["pca_center"], shift, atol=1e-12)
        self.assertEqual(result["params"]["origin"], "centered_anchor_midpoint")
        self.assertEqual(len(result["passage_sets"]["sets"]), 1)
        labels = {item["label"] for item in result["passage_sets"]["sets"][0]["items"]}
        self.assertEqual(labels, {"reference-3-0", "reference-4-0"})


if __name__ == "__main__":
    unittest.main()
