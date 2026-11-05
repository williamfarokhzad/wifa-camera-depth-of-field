import itertools
import json
import subprocess
import sys
import tempfile
import unittest
from fractions import Fraction as Q
from pathlib import Path

from dof import evaluate

ROOT = Path(__file__).resolve().parents[1]
BASE = dict(focal_mm=16, focus_mm=300, f_number=8, pixel_pitch_um=3.45,
            budget_px=2, target_mm=[280, 320])


def cone_reference(f, s, n, pitch, z):
    """Intersect the image cone at the sensor using exact thin-lens distances."""
    f, s, n, pitch, z = map(lambda x: Q(str(x)), (f, s, n, pitch, z))
    sensor = 1 / (1/f - 1/s)
    convergence = 1 / (1/f - 1/z)
    return abs(convergence - sensor) / convergence * (f / n) * 1000 / pitch


class DepthOfField(unittest.TestCase):
    def test_examples(self):
        expected = {"f8": False, "f11": True, "one-pixel": False}
        for name, verdict in expected.items():
            with self.subTest(example=name):
                result = subprocess.check_output(
                    [sys.executable, "dof.py", f"examples/{name}.json"], cwd=ROOT)
                self.assertEqual(json.loads(result)["geometric_pass"], verdict)

    def test_ray_geometry_on_72_cases(self):
        for f, s, n, pitch, scale in itertools.product(
                (8, 16, 25), (150, 300), (4, 8, 11), (2.74, 3.45), (0.8, 1.2)):
            z = s * scale
            observed = evaluate(focal_mm=f, focus_mm=s, f_number=n, pixel_pitch_um=pitch,
                                budget_px=2, target_mm=[z, z])["endpoint_blur_px"][0]
            self.assertAlmostEqual(observed, float(cone_reference(f, s, n, pitch, z)), places=11)

    def test_two_boundaries_equal_budget_and_are_asymmetric(self):
        r = evaluate(**BASE)
        near, far = r["near_mm"], r["far_mm"]
        self.assertAlmostEqual(float(cone_reference(16, 300, 8, 3.45, near)), 2)
        self.assertAlmostEqual(float(cone_reference(16, 300, 8, 3.45, far)), 2)
        self.assertGreater(far - 300, 300 - near)
        self.assertTrue(evaluate(**(BASE | {"target_mm": [near, far]}))["geometric_pass"])

    def test_exact_focus_has_zero_geometric_defocus(self):
        r = evaluate(**(BASE | {"target_mm": [300, 300]}))
        self.assertEqual(r["endpoint_blur_px"], [0, 0])
        self.assertTrue(r["geometric_pass"])

    def test_range_endpoints_cover_interior(self):
        end = max(evaluate(**BASE)["endpoint_blur_px"])
        for z in range(280, 321):
            self.assertLessEqual(float(cone_reference(16, 300, 8, 3.45, z)), end + 1e-12)

    def test_unbounded_far_at_exact_transition(self):
        r = evaluate(focal_mm=10, focus_mm=110, f_number=1, pixel_pitch_um=10,
                     budget_px=100, target_mm=[110, 1e7])
        self.assertTrue(r["geometrically_unbounded_far"])
        self.assertIsNone(r["far_mm"])
        self.assertTrue(r["geometric_pass"])

    def test_near_limit_outside_model_domain_is_flagged(self):
        r = evaluate(**(BASE | {"budget_px": 1e6}))
        self.assertIsNone(r["near_mm"])
        self.assertTrue(r["near_limited_by_model_domain"])

    def test_invalid_values(self):
        for patch in ({"focus_mm": 16}, {"target_mm": [320, 280]}, {"target_mm": [16, 300]},
                      {"f_number": 0}, {"budget_px": True}, {"pixel_pitch_um": float("nan")},
                      {"focal_mm": float("inf")}, {"target_mm": []}):
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                evaluate(**(BASE | patch))

    def test_bad_json_reports_clean_cli_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad.json"
            bad.write_text("[", encoding="utf-8")
            p = subprocess.run([sys.executable, "dof.py", str(bad)], cwd=ROOT, capture_output=True)
        self.assertEqual(p.returncode, 2)
        self.assertIn(b"Input error:", p.stderr)
        self.assertNotIn(b"Traceback", p.stderr)


if __name__ == "__main__":
    unittest.main()
