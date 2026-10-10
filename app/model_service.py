from __future__ import annotations

import os
import re
import threading
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps


ROOT_DIR = Path(__file__).resolve().parents[1]


class ModelUnavailableError(RuntimeError):
    """Raised when prediction is requested before valid weights are loaded."""


class ModelConfigurationError(RuntimeError):
    """Raised when model output does not match the configured label metadata."""


class ModelService:
    def __init__(self) -> None:
        self.model_path = Path(os.getenv("MODEL_PATH", str(ROOT_DIR / "model" / "xray_model.hdf5")))
        self.labels_path = Path(os.getenv("MODEL_CLASSES_PATH", str(ROOT_DIR / "model" / "model_classes.txt")))
        self.preprocessing = os.getenv("XRAY_PREPROCESSING", "raw").strip().lower()
        self.color_order = os.getenv("XRAY_COLOR_ORDER", "rgb").strip().lower()
        self.model: Any | None = None
        self.labels: list[str] = []
        self.load_error: str | None = None
        self._lock = threading.Lock()

    @property
    def ready(self) -> bool:
        return self.model is not None and len(self.labels) > 1 and self.load_error is None

    @property
    def unavailable_message(self) -> str:
        if self.load_error:
            return self.load_error
        return "النموذج غير جاهز. أضف ملفات النموذج المطلوبة ثم أعد تشغيل الخادم."

    def health(self) -> dict[str, Any]:
        return {
            "service": "PulmoSight AI",
            "status": "ready" if self.ready else "setup_required",
            "model_ready": self.ready,
            "model_name": "Chest X-ray classifier",
            "message": "النموذج محمّل وجاهز للاختبار البحثي." if self.ready else self.unavailable_message,
            "supported_labels": self.labels if self.ready else ["NORMAL", "PNEUMONIA"],
            "preprocessing": self.preprocessing,
            "color_order": self.color_order,
        }

    def _read_labels(self) -> list[str]:
        if not self.labels_path.is_file():
            # These are the two labels documented by the upstream prototype.
            # A real model output must still have the same number/order of classes.
            return ["NORMAL", "PNEUMONIA"]

        labels: list[str] = []
        try:
            lines = self.labels_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            return []
        for raw in lines:
            line = raw.strip()
            # The upstream text file contains unrelated shell text after a blank line.
            if not line:
                if labels:
                    break
                continue
            if re.fullmatch(r"[A-Za-z][A-Za-z0-9 _-]{0,39}", line):
                labels.append(line.upper())
            else:
                break
        return labels

    def load(self) -> bool:
        self.model = None
        self.labels = []
        if not self.model_path.is_file():
            self.load_error = (
                "ملف النموذج غير موجود. أضف model/xray_model.hdf5 من نسخة تملك حق استخدامها، "
                "ثم أعد تشغيل الخادم."
            )
            return False

        try:
            import tensorflow as tf  # Imported only when model weights are present.

            loaded_model = tf.keras.models.load_model(str(self.model_path), compile=False)
            labels = self._read_labels()
            shape = loaded_model.output_shape
            if isinstance(shape, list):
                if len(shape) != 1:
                    raise ModelConfigurationError("النموذج متعدد المخارج؛ يلزم تحديد مخرج التصنيف الصحيح.")
                shape = shape[0]
            output_count = int(shape[-1]) if shape and shape[-1] is not None else 0
            if output_count < 2:
                raise ModelConfigurationError(
                    "هذا التكامل يتوقع نموذج تصنيف متعدد الفئات (مخرجين على الأقل)."
                )
            if len(labels) != output_count:
                raise ModelConfigurationError(
                    f"عدد أسماء الفئات ({len(labels)}) لا يطابق مخارج النموذج ({output_count}). "
                    "راجع ترتيب model_classes.txt مقابل بيانات التدريب."
                )
            if self.preprocessing not in {"raw", "unit", "resnet50", "mobilenet_v2"}:
                raise ModelConfigurationError(
                    "XRAY_PREPROCESSING يجب أن يكون raw أو unit أو resnet50 أو mobilenet_v2."
                )
            if self.color_order not in {"rgb", "bgr"}:
                raise ModelConfigurationError("XRAY_COLOR_ORDER يجب أن يكون rgb أو bgr.")
            self.model = loaded_model
            self.labels = labels
            self.load_error = None
            return True
        except Exception as exc:
            self.model = None
            if isinstance(exc, ModelConfigurationError):
                self.load_error = str(exc)
            else:
                self.load_error = (
                    "تعذر تحميل النموذج. تحقّق من إصدار TensorFlow وتوافق ملف الأوزان."
                )
            return False

    def _input_size(self) -> tuple[int, int]:
        if self.model is None:
            raise ModelUnavailableError(self.unavailable_message)
        shape = self.model.input_shape
        if isinstance(shape, list):
            shape = shape[0]
        if not shape or len(shape) < 3:
            return (180, 180)
        height = int(shape[1]) if shape[1] else 180
        width = int(shape[2]) if shape[2] else 180
        return width, height

    def predict(self, image: Image.Image) -> dict[str, Any]:
        if not self.ready or self.model is None:
            raise ModelUnavailableError(self.unavailable_message)

        width, height = self._input_size()
        prepared = ImageOps.fit(
            image.convert("RGB"),
            (width, height),
            method=Image.Resampling.LANCZOS,
        )
        pixels = np.asarray(prepared, dtype=np.float32)
        if self.color_order == "bgr":
            pixels = pixels[..., ::-1].copy()

        # Must match training-time preprocessing; raw is the apparent upstream default.
        if self.preprocessing == "unit":
            pixels = pixels / 255.0
        elif self.preprocessing == "resnet50":
            import tensorflow as tf
            pixels = tf.keras.applications.resnet50.preprocess_input(pixels)
        elif self.preprocessing == "mobilenet_v2":
            import tensorflow as tf
            pixels = tf.keras.applications.mobilenet_v2.preprocess_input(pixels)

        with self._lock:
            raw = self.model.predict(np.expand_dims(pixels, axis=0), verbose=0)

        values = np.asarray(raw, dtype=np.float64).reshape(-1)
        if values.size != len(self.labels):
            raise ModelConfigurationError(
                f"عدد القيم الخارجة ({values.size}) لا يطابق عدد أسماء الفئات ({len(self.labels)})."
            )
        if not np.all(np.isfinite(values)):
            raise ModelConfigurationError("أنتج النموذج قيماً غير صالحة.")

        # Keep a proper probability-like distribution for display, but do not call it calibrated.
        if np.all(values >= 0) and np.all(values <= 1) and np.isclose(values.sum(), 1.0, atol=1e-3):
            scores = values
        else:
            shifted = values - np.max(values)
            exp_values = np.exp(shifted)
            scores = exp_values / exp_values.sum()

        winner = int(np.argmax(scores))
        return {
            "label": self.labels[winner],
            "score": round(float(scores[winner]), 6),
            "scores": [
                {"label": label, "score": round(float(score), 6)}
                for label, score in sorted(
                    zip(self.labels, scores.tolist()), key=lambda item: item[1], reverse=True
                )
            ],
            "input_size": {"width": width, "height": height},
            "preprocessing": self.preprocessing,
            "color_order": self.color_order,
            "interpretation": "model_output_score",
        }
