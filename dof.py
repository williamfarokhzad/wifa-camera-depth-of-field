"""Geometric blur on a fixed sensor plane; all object distances are in mm."""
import argparse
import json
import math
from pathlib import Path


def number(value, label):
    # Broad numerical bounds keep unsupported, nonphysical inputs explicit.
    if (isinstance(value, bool) or not isinstance(value, (float, int))
            or not 1e-6 <= value <= 1e9):
        raise ValueError(f"{label} must be a number from 1e-6 through 1e9")
    return float(value)


def evaluate(*, focal_mm, focus_mm, f_number, pixel_pitch_um, budget_px, target_mm):
    f = number(focal_mm, "focal_mm")
    s = number(focus_mm, "focus_mm")
    n = number(f_number, "f_number")
    pitch = number(pixel_pitch_um, "pixel_pitch_um")
    budget = number(budget_px, "budget_px")
    if not isinstance(target_mm, list) or len(target_mm) != 2:
        raise ValueError("target_mm must contain the near and far distances")
    a, b = (number(z, "target distance") for z in target_mm)
    if a > b or min(a, s) <= f:
        raise ValueError("distances must exceed focal length; list near before far")

    blur_scale_px = (f * f / (n * (s - f))) * 1000 / pitch
    ratio = budget / blur_scale_px
    near = s / (1 + ratio)
    # The algebraic near bound can fall outside the model's z > f domain.
    far = None if ratio >= 1 else s / (1 - ratio)
    blur = [blur_scale_px * abs((z - s) / z) for z in (a, b)]
    worst = max(blur)  # Monotone on each side of best focus.
    return {
        "near_mm": near if near > f else None,
        "near_limited_by_model_domain": near <= f,
        "far_mm": far,
        "geometrically_unbounded_far": far is None,
        "endpoint_blur_px": blur,
        "geometric_pass": worst <= budget or math.isclose(worst, budget, rel_tol=1e-12),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("example", type=Path, help="JSON containing the lens and target inputs")
    args = parser.parse_args(argv)
    try:
        inputs = json.loads(args.example.read_text(encoding="utf-8"))
        result = evaluate(**inputs)
    except (OSError, ValueError, TypeError) as error:
        parser.exit(2, f"Input error: {error}\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
