from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.model_download import (  # noqa: E402
    MODEL_FILENAME,
    MODEL_LICENSE_URL,
    MODEL_REPO_ID,
    MODEL_REVISION,
    MODEL_SHA256,
    ModelDownloadError,
    download_verified_model,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download the pinned PulmoSight chest X-ray model and verify SHA-256."
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=ROOT / "model" / MODEL_FILENAME,
        help="Output file path (defaults to model/best_chest_xray_model.h5).",
    )
    parser.add_argument("--force", action="store_true", help="Force a fresh download.")
    args = parser.parse_args()

    try:
        target = download_verified_model(args.target, force=args.force)
    except ModelDownloadError as exc:
        print(f"Model download failed: {exc}", file=sys.stderr)
        return 1

    print("Pretrained model downloaded and verified.")
    print(f"Path: {target}")
    print(f"Size: {target.stat().st_size:,} bytes")
    print(f"SHA-256: {MODEL_SHA256}")
    print(f"Source: {MODEL_REPO_ID} @ {MODEL_REVISION}")
    print(f"License / model card: {MODEL_LICENSE_URL}")
    print("The file is intentionally excluded from Git; the application downloads/verifies it locally.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
