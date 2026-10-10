from __future__ import annotations

import io
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from app.model_service import ModelService, ModelUnavailableError, ModelConfigurationError

ROOT_DIR = Path(__file__).resolve().parents[1]
WEB_DIR = ROOT_DIR / "web"
MAX_UPLOAD_MB = max(1, int(os.getenv("MAX_UPLOAD_MB", "12")))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
ALLOWED_FORMATS = {"JPEG", "PNG"}

model_service = ModelService()


@asynccontextmanager
async def lifespan(_: FastAPI):
    # The website can start without weights; the UI will show setup-required status.
    await run_in_threadpool(model_service.load)
    yield


app = FastAPI(
    title="PulmoSight AI",
    description="Research prototype for chest X-ray model evaluation. Not for clinical use.",
    version="0.1.0",
    lifespan=lifespan,
)
app.mount("/assets", StaticFiles(directory=str(WEB_DIR)), name="assets")


@app.get("/", include_in_schema=False)
async def home() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/health")
async def health() -> dict:
    return {**model_service.health(), "max_upload_mb": MAX_UPLOAD_MB}


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    if not model_service.ready:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "model_not_ready",
                "message": model_service.unavailable_message,
            },
        )

    contents = await file.read(MAX_UPLOAD_BYTES + 1)
    if not contents:
        raise HTTPException(status_code=400, detail={"code": "empty_file", "message": "الملف فارغ."})
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail={
                "code": "file_too_large",
                "message": f"حجم الصورة يتجاوز الحد المسموح ({MAX_UPLOAD_MB} MB).",
            },
        )

    try:
        with Image.open(io.BytesIO(contents)) as probe:
            image_format = (probe.format or "").upper()
            if image_format not in ALLOWED_FORMATS:
                raise HTTPException(
                    status_code=415,
                    detail={
                        "code": "unsupported_image",
                        "message": "الصيغ المدعومة هي PNG وJPG وJPEG فقط.",
                    },
                )
            probe.verify()
        with Image.open(io.BytesIO(contents)) as opened:
            image = opened.convert("RGB")
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(
            status_code=400,
            detail={"code": "invalid_image", "message": "تعذر قراءة الصورة. جرّب ملف PNG أو JPEG صالحاً."},
        )

    try:
        result = await run_in_threadpool(model_service.predict, image, True)
    except ModelUnavailableError as exc:
        raise HTTPException(status_code=503, detail={"code": "model_not_ready", "message": str(exc)})
    except ModelConfigurationError as exc:
        raise HTTPException(status_code=500, detail={"code": "model_configuration_error", "message": str(exc)})
    except Exception:
        # Do not send filesystem paths or internal stack traces to the browser.
        raise HTTPException(
            status_code=500,
            detail={
                "code": "inference_failed",
                "message": "تعذر إكمال التحليل. تحقّق من توافق النموذج وطريقة المعالجة ثم راجع سجل الخادم.",
            },
        )

    return {
        **result,
        "notice": "نتيجة بحثية آلية وليست تشخيصاً طبياً. يلزم تفسير اختصاصي أشعة.",
    }
