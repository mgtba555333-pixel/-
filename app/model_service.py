from __future__ import annotations

import os
import re
import threading
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


ROOT_DIR = Path(__file__).resolve().parents[1]


class ModelUnavailableError(RuntimeError):
    """Raised when prediction is requested before valid weights are loaded."""


class ModelConfigurationError(RuntimeError):
    """Raised when model output does not match the configured label metadata."""


class ModelService:
    def __init__(self) -> None:
        self.model_path = Path(os.getenv("MODEL_PATH", str(ROOT_DIR / "model" / "xray_model.hdf5")))
        self.labels_path = Path(os.getenv("MODEL_CLASSES_PATH", str(ROOT_DIR / "model" / "model_classes.txt")))
        # The upstream notebook uses Rescaling(1./255) as its first model layer.
        self.preprocessing = os.getenv("XRAY_PREPROCESSING", "unit").strip().lower()
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
            import tensorflow as tf

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
            input_shape = loaded_model.input_shape
            if isinstance(input_shape, list):
                if len(input_shape) != 1:
                    raise ModelConfigurationError("النموذج متعدد المدخلات؛ يلزم تحديد مدخل الصورة.")
                input_shape = input_shape[0]
            if not input_shape or len(input_shape) != 4 or input_shape[-1] not in (None, 3):
                raise ModelConfigurationError(
                    "هذا التكامل يتوقع مدخلاً واحداً لصور RGB بالشكل (batch, height, width, 3)."
                )
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
        height = int(shape[1]) if shape and shape[1] else 180
        width = int(shape[2]) if shape and shape[2] else 180
        return width, height

    def _prepare_input(self, image: Image.Image) -> np.ndarray:
        """Prepare input to match the model notebook: RGB, direct resize, then scaling."""
        if self.model is None:
            raise ModelUnavailableError(self.unavailable_message)
        width, height = self._input_size()
        prepared = image.convert("RGB").resize(
            (width, height),
            resample=Image.Resampling.BILINEAR,
        )
        pixels = np.asarray(prepared, dtype=np.float32)

        if self.color_order == "bgr":
            pixels = pixels[..., ::-1].copy()

        if self.preprocessing == "unit":
            pixels = pixels / 255.0
        elif self.preprocessing == "resnet50":
            import tensorflow as tf
            pixels = tf.keras.applications.resnet50.preprocess_input(pixels)
        elif self.preprocessing == "mobilenet_v2":
            import tensorflow as tf
            pixels = tf.keras.applications.mobilenet_v2.preprocess_input(pixels)
        elif self.preprocessing != "raw":
            raise ModelConfigurationError("إعداد معالجة الصورة غير صالح.")

        return np.expand_dims(pixels, axis=0)

    def _compute_gradcam(self, batch: np.ndarray, class_index: int) -> dict[str, Any]:
        """Return normalized Grad-CAM values; this is attention visualization, not segmentation."""
        if self.model is None:
            raise ModelUnavailableError(self.unavailable_message)

        import tensorflow as tf

        conv_layer = next(
            (layer for layer in reversed(self.model.layers) if isinstance(layer, tf.keras.layers.Conv2D)),
            None,
        )
        if conv_layer is None:
            raise ModelConfigurationError("لم يتم العثور على طبقة Conv2D مناسبة لإنشاء Grad-CAM.")

        grad_model = tf.keras.Model(
            inputs=self.model.inputs,
            outputs=[conv_layer.output, self.model.output],
        )
        with self._lock:
            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(batch, training=False)
                class_score = predictions[:, class_index]
            gradients = tape.gradient(class_score, conv_outputs)

        if gradients is None:
            raise ModelConfigurationError("تعذر حساب التدرجات لطبقة الالتفاف المختارة.")

        weights = tf.reduce_mean(gradients, axis=(0, 1, 2))
        activations = conv_outputs[0]
        heatmap = tf.reduce_sum(activations * weights, axis=-1)
        heatmap = tf.maximum(heatmap, 0)
        max_value = tf.reduce_max(heatmap)
        heatmap = tf.where(max_value > 0, heatmap / (max_value + tf.keras.backend.epsilon()), heatmap)
        values = np.asarray(heatmap.numpy(), dtype=np.float32)

        if not np.isfinite(values).all() or float(values.max(initial=0.0)) <= 0:
            raise ModelConfigurationError("خريطة Grad-CAM الناتجة فارغة أو غير صالحة.")

        return {
            "available": True,
            "method": "Grad-CAM",
            "target_layer": conv_layer.name,
            "width": int(values.shape[1]),
            "height": int(values.shape[0]),
            "heatmap": np.round(values, 3).tolist(),
            "notice": "الخريطة توضح تأثيرات النموذج الحسابية بشكل تقريبي، ولا تحدد الإصابة أو تستبدل تفسير اختصاصي الأشعة.",
        }

    def predict(self, image: Image.Image, include_explanation: bool = True) -> dict[str, Any]:
        if not self.ready or self.model is None:
            raise ModelUnavailableError(self.unavailable_message)

        batch = self._prepare_input(image)
        with self._lock:
            raw = self.model.predict(batch, verbose=0)

        values = np.asarray(raw, dtype=np.float64).reshape(-1)
        if values.size != len(self.labels):
            raise ModelConfigurationError(
                f"عدد القيم الخارجة ({values.size}) لا يطابق عدد أسماء الفئات ({len(self.labels)})."
            )
        if not np.all(np.isfinite(values)):
            raise ModelConfigurationError("أنتج النموذج قيماً غير صالحة.")

        # The upstream notebook's final Dense layer has softmax activation.
        # Preserve normalized probabilities when present; otherwise apply softmax to logits.
        if np.all(values >= 0) and np.all(values <= 1) and np.isclose(values.sum(), 1.0, atol=1e-3):
            scores = values
        else:
            shifted = values - np.max(values)
            exp_values = np.exp(shifted)
            scores = exp_values / exp_values.sum()

        winner = int(np.argmax(scores))
        result: dict[str, Any] = {
            "label": self.labels[winner],
            "score": round(float(scores[winner]), 6),
            "scores": [
                {"label": label, "score": round(float(score), 6)}
                for label, score in sorted(
                    zip(self.labels, scores.tolist()), key=lambda item: item[1], reverse=True
                )
            ],
            "input_size": {"width": int(batch.shape[2]), "height": int(batch.shape[1])},
            "preprocessing": self.preprocessing,
            "color_order": self.color_order,
            "interpretation": "model_output_score_not_clinically_calibrated",
        }

        if include_explanation:
            try:
                result["explanation"] = self._compute_gradcam(batch, winner)
            except Exception:
                # A visualization failure must not fabricate an explanation or hide the model result.
                result["explanation"] = {
                    "available": False,
                    "method": "Grad-CAM",
                    "message": "لم تتوفر خريطة Grad-CAM لهذا النموذج. النتيجة الحسابية أعلاه مستقلة عن الخريطة.",
                }

        return result
