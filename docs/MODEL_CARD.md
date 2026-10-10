# Model Card — Pinned Pediatric Chest X-ray Classifier

## Model artifact in this application

- **Repository:** [ayushirathour/chest-xray-pneumonia-detection](https://huggingface.co/ayushirathour/chest-xray-pneumonia-detection)
- **Revision:** `65ee3acc9ec589cdb0d1b081dd58c31ef9fed675`
- **File:** `best_chest_xray_model.h5`
- **Published size:** 11,376,320 bytes
- **Published SHA-256:** `7c85dc7a21b465b89d4a67a95d3a30a81e3f9748a1bdf048001521c1daa3f9d6`
- **License metadata:** MIT; see [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).
- **Architecture stated by publisher:** MobileNetV2 with a binary sigmoid head.
- **Task:** Normal vs pneumonia classification on pediatric chest radiographs.

## How PulmoSight obtains the weights

On first start, the backend can download the pinned file from Hugging Face using `huggingface_hub`. Before it loads the model, PulmoSight checks the downloaded file against the SHA-256 above. A mismatched file is not accepted. The pinned H5 file is tracked at `model/best_chest_xray_model.h5` after a GitHub Actions workflow verifies its SHA-256. The downloader can restore or re-verify it if it is missing.

Manual download command:

```bash
python scripts/download_model.py
```

The checksum was taken from the model-host file metadata. It verifies file identity, not clinical quality or source integrity beyond that pinned artifact.

## Input/output details documented by the publisher

- Expected input size: 224 × 224 RGB.
- Input resize: 224 × 224.
- Pixel scaling in the published example: convert to float and divide by 255 (range 0–1).
- Model output: a single sigmoid value, interpreted in the example as the pneumonia score.
- Class mapping used by PulmoSight: `NORMAL = 1 - score`, `PNEUMONIA = score`.

PulmoSight uses these published preprocessing instructions for this pinned model. If the loaded file exposes an internal `Rescaling` layer, auto mode switches to raw 0–255 input to prevent double scaling. Always inspect the loaded model metadata when replacing the checkpoint.

## What has and has not been verified here

The public model card states that the model targets pediatric chest X-rays (ages 1–5) and reports results on an independent cohort. Those figures are publisher-reported and have **not** been independently reproduced as part of this coding task. Passing software tests does not validate the medical model.

The repository's automated tests currently cover:
- Image preprocessing shape and value range.
- RGB/BGR channel behavior.
- Binary sigmoid and multiclass score conversion.
- Metric helper calculations.
- The API refusing to fabricate a result when the model is unavailable.

The dedicated model workflow successfully ran a TensorFlow smoke test on a synthetic, non-patient image, verified the output shape and score conversion, and generated a Grad-CAM map for the pinned MobileNetV2 model. This is a software compatibility check only—not clinical validation. Grad-CAM may still be unavailable for other Keras model graph layouts; in that case, the app reports the limitation rather than fabricating a heatmap.

## Limitations and responsible use

This is a research/engineering integration only. It is not a medical device and must not be used to diagnose, triage, or treat patients. Pediatric data must not be assumed to generalize to adults or to different hospitals, scanners, acquisition views, or populations. Model output scores are not established here as calibrated clinical probabilities.

Before any clinical or patient-facing use, a qualified team would need to verify dataset provenance and licensing, ensure patient-level train/test separation, independently reproduce results, assess calibration and subgroup performance, run external validation, and obtain appropriate regulatory and clinical review.
