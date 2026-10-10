from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Keep this test offline: the GitHub Action downloaded the pinned, hash-verified model already.
os.environ["MODEL_AUTO_DOWNLOAD"] = "false"
os.environ["MODEL_PATH"] = str(ROOT / "model" / "best_chest_xray_model.h5")
os.environ["MODEL_CLASSES_PATH"] = str(ROOT / "model" / "model_classes.txt")
os.environ.setdefault("XRAY_PREPROCESSING", "auto")
os.environ.setdefault("XRAY_COLOR_ORDER", "rgb")

from app.model_service import ModelService  # noqa: E402


def main() -> int:
    weights = Path(os.environ["MODEL_PATH"])
    if not weights.is_file():
        print(f"Model weights missing: {weights}", file=sys.stderr)
        return 1

    service = ModelService()
    if not service.load():
        print(f"Model load failed: {service.unavailable_message}", file=sys.stderr)
        return 1

    if service.output_mode != "binary_sigmoid":
        print(
            f"Expected binary sigmoid output, got {service.output_mode!r}.",
            file=sys.stderr,
        )
        return 1

    # Non-patient synthetic input tests shape and API plumbing only, not clinical accuracy.
    probe = Image.new("RGB", (320, 240), (96, 104, 112))
    result = service.predict(probe, include_explanation=True)
    if result.get("label") not in {"NORMAL", "PNEUMONIA"}:
        print("Unexpected class label in smoke test.", file=sys.stderr)
        return 1
    if not np.isfinite(float(result.get("score", float("nan")))):
        print("Model returned a non-finite score.", file=sys.stderr)
        return 1
    if not np.isclose(sum(float(item["score"]) for item in result["scores"]), 1.0, atol=1e-5):
        print("Class scores do not sum to 1.", file=sys.stderr)
        return 1

    explanation = result.get("explanation", {})
    print("Model smoke test passed.")
    print(f"Input: {result['input_size']['width']}x{result['input_size']['height']} RGB")
    print(f"Output mode: {result['output_mode']}")
    print(f"Preprocessing: {result['preprocessing']}")
    print(f"Prediction interface: {result['label']} (synthetic input only; no clinical meaning)")
    print(f"Grad-CAM available: {bool(explanation.get('available'))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
