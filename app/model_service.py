from __future__ import annotations

import os
import re
import threading
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from app.model_download import (
    MODEL_FILENAME,
    MODEL_SHA256,
    MODEL_REVISION,
    MODEL_LICENSE_URL,
    ModelDownloadError,
    download_verified_model,
)


ROOT_DIR = Path(__file__).resolve().parents[1]


class ModelUnavailableError(RuntimeError):
    """Raised when prediction is requested before valid weights are loaded."""


class ModelConfigurationError(RuntimeError):
    """Raised when model output does not match the configured label metadata."""


def decode_scores(raw: Any, output_mode: str, labels: list[str]) -> tuple[list[str], np.ndarray]:
    """Convert model-head output into ordered class scores without applying softmax twice."""
    values = np.asarray(raw, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(values)):
        raise ModelConfigurationError("أنتج النموذج قيماً غير صالحة.")

    if output_mode == "binary_sigmoid":
        if values.size != 1:
            raise ModelConfigurationError("مخرج sigmoid الثنائي يجب أن يحتوي على قيمة واحدة.")
        positive = float(values[0])
        if positive < 0.0 or positive > 1.0:
            positive = float(1.0 / (1.0 + np.exp(-np.clip(positive, -80, 80))))
        return ["NORMAL", "PNEUMONIA"], np.asarray([1.0 - positive, positive], dtype=np.float64)

    if values.size != len(labels):
        raise ModelConfigurationError(
            f"عدد القيم الخارجة ({values.size}) لا يطابق عدد أسماء الفئات ({len(labels)})."
        )
    if np.all(values >= 0) and np.all(values <= 1) and np.isclose(values.sum(), 1.0, atol=1e-3):
        scores = values
    else:
        shifted = values - np.max(values)
        exp_values = np.exp(shifted)
        scores = exp_values / exp_values.sum()
    return labels, scores


class ModelService:
    def __init__(self) -> None:
        self.model_path = Path(
            os.getenv("MODEL_PATH", str(ROOT_DIR / "model" / MODEL_FILENAME))
        )
        self.labels_path = Path(
            os.getenv("MODEL_CLASSES_PATH", str(ROOT_DIR / "model" / "model_classes.txt"))
        )
        self.requested_preprocessing = os.getenv("XRAY_PREPROCESSING", "auto").strip().lower()
        self.preprocessing = self.requested_preprocessing
        self.color_order = os.getenv("XRAY_COLOR_ORDER", "rgb").strip().lower()
        self.auto_download = os.getenv("MODEL_AUTO_DOWNLOAD", "true").strip().lower() in {
            "1", "true", "yes", "on"
        }
        self.model: Any | None = None
        self.labels: list[str] = []
        self.output_mode: str | None = None
        self.internal_rescaling = False
        self.load_error: str | None = None
        self.model_source = "local"
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
            "model_name": "MobileNetV2 chest X-ray classifier" if self.model_source == "huggingface" else "Chest X-ray classifier",
            "model_source": self.model_source,
            "model_license_url": MODEL_LICENSE_URL if self.model_source == "huggingface" else None,
            "model_revision": MODEL_REVISION if self.model_source == "huggingface" else None,
            "model_sha256": MODEL_SHA256 if self.model_source == "huggingface" else None,
            "output_mode": self.output_mode,
            "message": (
                "الأوزان محمّلة والتحقق من سلامتها ناجح؛ يلزم تقييم مستقل قبل تفسير النتائج."
                if self.ready else self.unavailable_message
            ),
            "supported_labels": self.labels if self.ready else ["NORMAL", "PNEUMONIA"],
            "preprocessing": self.preprocessing,
            "color_order": self.color_order,
            "input_size": (
                {"width": self._input_size()[0], "height": self._input_size()[1]}
                if self.ready else None
            ),
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
            if not line:
                if labels:
                    break
                continue
            if re.fullmatch(r"[A-Za-z][A-Za-z0-9 _-]{0,39}", line):
                labels.append(line.upper())
            else:
                break
        return labels

    def _obtain_model_file(self) -> bool:
        # The preferred model is always checksum-verified, including after a prior download.
        if self.model_path.is_file() and self.model_path.name == MODEL_FILENAME:
            try:
                download_verified_model(self.model_path)
                self.model_source = "huggingface"
                return True
            except ModelDownloadError as exc:
                self.load_error = str(exc)
                return False

        if self.model_path.is_file():
            self.model_source = "local"
            return True

        if not self.auto_download:
            self.load_error = (
                f"ملف الأوزان غير موجود: {self.model_path}. "
                "شغّل python scripts/download_model.py لتنزيل النموذج والتحقق من بصمته."
            )
            return False

        try:
            download_verified_model(self.model_path)
            self.model_source = "huggingface"
            return True
        except ModelDownloadError as exc:
            self.load_error = str(exc)
            return False

    def load(self) -> bool:
        self.model = None
        self.labels = []
        self.output_mode = None
        self.load_error = None

        if not self._obtain_model_file():
            return False

        try:
            import tensorflow as tf

            loaded_model = tf.keras.models.load_model(str(self.model_path), compile=False)
            output_shape = loaded_model.output_shape
            if isinstance(output_shape, list):
                if len(output_shape) != 1:
                    raise ModelConfigurationError("النموذج متعدد المخارج؛ يلزم تحديد مخرج التصنيف الصحيح.")
                output_shape = output_shape[0]
            output_count = int(output_shape[-1]) if output_shape and output_shape[-1] is not None else 0

            if output_count == 1:
                # The pinned MobileNetV2 model outputs P(PNEUMONIA) with a sigmoid unit.
                self.output_mode = "binary_sigmoid"
                labels = self._read_labels()
                if len(labels) != 2 or "NORMAL" not in labels[0] or "PNEUMONIA" not in labels[1]:
                    labels = ["NORMAL", "PNEUMONIA"]
            elif output_count >= 2:
                self.output_mode = "multiclass"
                labels = self._read_labels()
                if len(labels) != output_count:
                    raise ModelConfigurationError(
                        f"عدد أسماء الفئات ({len(labels)}) لا يطابق مخارج النموذج ({output_count}). "
                        "راجع ترتيب model_classes.txt مقابل بيانات التدريب."
                    )
            else:
                raise ModelConfigurationError("شكل مخرج النموذج غير مدعوم.")

            input_shape = loaded_model.input_shape
            if isinstance(input_shape, list):
                if len(input_shape) != 1:
                    raise ModelConfigurationError("النموذج متعدد المدخلات؛ يلزم تحديد مدخل الصورة.")
                input_shape = input_shape[0]
            if not input_shape or len(input_shape) != 4 or input_shape[-1] not in (None, 3):
                raise ModelConfigurationError(
                    "هذا التكامل يتوقع مدخلاً واحداً لصور RGB بالشكل (batch, height, width, 3)."
                )

            self.internal_rescaling = any(
                isinstance(layer, tf.keras.layers.Rescaling) for layer in loaded_model.layers
            )

            if self.requested_preprocessing == "auto":
                if self.internal_rescaling:
                    self.preprocessing = "raw"
                elif self.model_source == "huggingface":
                    self.preprocessing = "unit"
                else:
                    self.preprocessing = "raw"
            else:
                self.preprocessing = self.requested_preprocessing

            if self.preprocessing not in {"raw", "unit", "resnet50", "mobilenet_v2"}:
                raise ModelConfigurationError(
                    "XRAY_PREPROCESSING يجب أن يكون auto أو raw أو unit أو resnet50 أو mobilenet_v2."
                )
            if self.internal_rescaling and self.preprocessing != "raw":
                raise ModelConfigurationError(
                    "النموذج يحتوي على طبقة Rescaling داخلية؛ اختر XRAY_PREPROCESSING=raw "
                    "أو auto حتى لا تتم معالجة البكسلات مرتين."
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
                    "تعذر تحميل النموذج. تحقق من إصدار TensorFlow وتوافق ملف H5، "
                    "ثم راجع سجل الخادم."
                )
            return False

    def _input_size(self) -> tuple[int, int]:
        if self.model is None:
            raise ModelUnavailableError(self.unavailable_message)
        shape = self.model.input_shape
        if isinstance(shape, list):
            shape = shape[0]
        height = int(shape[1]) if shape and shape[1] else 224
        width = int(shape[2]) if shape and shape[2] else 224
        return width, height

    def _prepare_input(self, image: Image.Image) -> np.ndarray:
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
        elif self.preprocessing not in {"raw", "auto"}:
            raise ModelConfigurationError("إعداد معالجة الصورة غير صالح.")

        return np.expand_dims(pixels, axis=0)

    def _compute_gradcam(self, batch: np.ndarray, class_index: int) -> dict[str, Any]:
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
                if self.output_mode == "binary_sigmoid":
                    positive_score = predictions[:, 0]
                    class_score = positive_score if class_index == 1 else (1.0 - positive_score)
                else:
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

        labels, scores = decode_scores(raw, self.output_mode or "multiclass", self.labels)
        winner = int(np.argmax(scores))
        result: dict[str, Any] = {
            "label": labels[winner],
            "score": round(float(scores[winner]), 6),
            "scores": [
                {"label": label, "score": round(float(score), 6)}
                for label, score in sorted(
                    zip(labels, scores.tolist()), key=lambda item: item[1], reverse=True
                )
            ],
            "input_size": {"width": int(batch.shape[2]), "height": int(batch.shape[1])},
            "preprocessing": self.preprocessing,
            "color_order": self.color_order,
            "output_mode": self.output_mode,
            "interpretation": "model_output_score_not_clinically_calibrated",
            "model_source": self.model_source,
        }

        if include_explanation:
            try:
                result["explanation"] = self._compute_gradcam(batch, winner)
            except Exception:
                result["explanation"] = {
                    "available": False,
                    "method": "Grad-CAM",
                    "message": "لم تتوفر خريطة Grad-CAM لهذا النموذج. النتيجة الحسابية أعلاه مستقلة عن الخريطة.",
                }

        return result
