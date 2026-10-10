from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from pathlib import Path

MODEL_REPO_ID = "ayushirathour/chest-xray-pneumonia-detection"
MODEL_REVISION = "65ee3acc9ec589cdb0d1b081dd58c31ef9fed675"
MODEL_FILENAME = "best_chest_xray_model.h5"
MODEL_SHA256 = "7c85dc7a21b465b89d4a67a95d3a30a81e3f9748a1bdf048001521c1daa3f9d6"
MODEL_LICENSE_URL = "https://huggingface.co/ayushirathour/chest-xray-pneumonia-detection"


class ModelDownloadError(RuntimeError):
    """The licensed upstream model could not be downloaded or verified."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_verified_model(target_path: Path, force: bool = False) -> Path:
    """Download the pinned MIT-listed model and verify its documented SHA-256 before use."""
    target_path = target_path.resolve()
    if target_path.is_file() and not force:
        existing_hash = sha256_file(target_path)
        if existing_hash == MODEL_SHA256:
            return target_path

    try:
        from huggingface_hub import hf_hub_download

        cached_path = Path(
            hf_hub_download(
                repo_id=MODEL_REPO_ID,
                filename=MODEL_FILENAME,
                revision=MODEL_REVISION,
                force_download=force,
            )
        )
    except Exception as exc:
        raise ModelDownloadError(
            "تعذر تنزيل النموذج من Hugging Face. تحقق من اتصال الإنترنت أو نفّذ "
            "python scripts/download_model.py --force."
        ) from exc

    actual_hash = sha256_file(cached_path)
    if actual_hash.lower() != MODEL_SHA256:
        if not force:
            return download_verified_model(target_path, force=True)
        raise ModelDownloadError(
            f"بصمة النموذج لا تطابق البصمة المثبتة. المتوقع {MODEL_SHA256}، "
            f"والحاصل {actual_hash}. أوقف التحميل حفاظاً على سلامة الملفات."
        )

    target_path.parent.mkdir(parents=True, exist_ok=True)
    if cached_path.resolve() == target_path:
        return target_path

    handle, temporary_name = tempfile.mkstemp(
        prefix=target_path.name + ".",
        suffix=".part",
        dir=str(target_path.parent),
    )
    os.close(handle)
    temporary_path = Path(temporary_name)
    try:
        shutil.copyfile(cached_path, temporary_path)
        copied_hash = sha256_file(temporary_path)
        if copied_hash.lower() != MODEL_SHA256:
            raise ModelDownloadError("فشل التحقق من النسخة المحلية بعد نسخ النموذج.")
        os.replace(temporary_path, target_path)
    finally:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass

    return target_path
