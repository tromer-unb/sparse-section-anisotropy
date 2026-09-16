"""Command-line interface for tensor inference from oriented 2D sections."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .correlation import measure_directional_lengths
from .io import load_section, pore_indicator
from .tensor import infer_tensor, principal_properties


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def run_manifest(path: str | Path) -> dict:
    path = Path(path)
    spec = json.loads(path.read_text(encoding="utf-8"))
    root = path.parent

    spacing = float(spec.get("pixel_size", 1.0))
    if spacing <= 0:
        raise ValueError("pixel_size must be positive")
    unit = str(spec.get("length_unit", "pixel"))
    pore_value = spec.get("pore_value", 0)
    theta_step = float(spec.get("theta_step_deg", 5.0))
    max_lag = spec.get("max_lag_pixels")

    measurements = []
    section_report = []

    for item in spec["sections"]:
        image_path = root / item["path"]
        image = load_section(image_path)
        binary = pore_indicator(image, pore_value=item.get("pore_value", pore_value))
        theta, ell_pixels = measure_directional_lengths(
            binary,
            theta_step_deg=theta_step,
            max_lag=max_lag,
        )
        ell = ell_pixels * spacing
        basis = np.asarray(item["basis"], dtype=float)
        measurements.append({"basis": basis, "theta_deg": theta, "length": ell})
        section_report.append(
            {
                "name": item.get("name", image_path.name),
                "path": str(image_path),
                "shape": list(image.shape),
                "porosity_2d": float(binary.mean()),
                "finite_direction_count": int(np.isfinite(ell).sum()),
                "direction_count": int(len(ell)),
            }
        )

    fit = infer_tensor(measurements)
    result = {
        "status": "ok" if fit.Q is not None else "rank_deficient",
        "rank": fit.rank,
        "condition_number": fit.condition_number,
        "measurement_count": fit.measurement_count,
        "length_unit": unit,
        "tensor_unit": f"1/{unit}^2",
        "sections": section_report,
    }

    if fit.Q is not None:
        props = principal_properties(fit.Q)
        result["Q"] = fit.Q
        result.update(props)

    return {k: _jsonable(v) if not isinstance(v, dict) else {kk: _jsonable(vv) for kk, vv in v.items()} for k, v in result.items()}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Infer a 3D pore-correlation tensor from oriented binary 2D sections."
    )
    parser.add_argument("manifest", help="JSON section manifest")
    parser.add_argument("--output", "-o", help="write JSON result to this file")
    args = parser.parse_args()

    result = run_manifest(args.manifest)
    text = json.dumps(result, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)

    if result["status"] != "ok":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
