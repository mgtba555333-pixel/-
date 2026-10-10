from __future__ import annotations

import io
import unittest

from fastapi.testclient import TestClient
from PIL import Image

from app.main import app


def make_png() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), (120, 120, 120)).save(buffer, format="PNG")
    return buffer.getvalue()


class APISafetyTests(unittest.TestCase):
    def test_health_reports_setup_required_without_weight_file(self) -> None:
        with TestClient(app) as client:
            response = client.get("/api/health")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertFalse(payload["model_ready"])
        self.assertEqual(payload["status"], "setup_required")

    def test_prediction_refuses_to_fabricate_when_model_is_missing(self) -> None:
        with TestClient(app) as client:
            response = client.post(
                "/api/predict",
                files={"file": ("sample.png", make_png(), "image/png")},
            )

        self.assertEqual(response.status_code, 503)
        detail = response.json()["detail"]
        self.assertEqual(detail["code"], "model_not_ready")
        self.assertNotIn("label", response.json())


if __name__ == "__main__":
    unittest.main()
