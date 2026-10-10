# PulmoSight AI — Chest X-ray Research Platform

An Arabic-first research web app for chest X-ray classification, optional Grad-CAM visualization, and evaluation metrics.

> **Research prototype only.** Not a medical device. Never use it for diagnosis, triage, or treatment. A qualified clinician must interpret radiology findings. The model described below is a pediatric research model; its author describes a target group of children aged 1–5 years. Do not assume it generalizes to other ages, hospitals, scanners, or acquisition protocols.

## What's included

- Responsive Arabic RTL dashboard and image preview.
- FastAPI API with upload type/size validation.
- Automatic download of a pinned chest X-ray model to the ignored local `model/` folder.
- SHA-256 verification before a downloaded model is accepted.
- Binary sigmoid decoding for the pretrained MobileNetV2 model.
- Optional Grad-CAM if the loaded Keras architecture exposes a compatible convolution layer.
- Held-out evaluation report: accuracy, precision, recall, F1, specificity, ROC-AUC, Brier score, ECE, and bootstrap intervals.
- Unit tests and GitHub Actions quality checks.
- Model weights are not committed into Git history. They are downloaded locally from the model host, SHA-256 verified, and excluded by `.gitignore`.

## Model source and download

The default checkpoint comes from:

- Model: [ayushirathour/chest-xray-pneumonia-detection](https://huggingface.co/ayushirathour/chest-xray-pneumonia-detection)
- License metadata: MIT; retain attribution and license notice.
- Pinned revision: `65ee3acc9ec589cdb0d1b081dd58c31ef9fed675`
- File: `best_chest_xray_model.h5` (11,376,320 bytes)
- Expected SHA-256: `7c85dc7a21b465b89d4a67a95d3a30a81e3f9748a1bdf048001521c1daa3f9d6`

The model card describes a MobileNetV2 classifier for pediatric chest radiographs, RGB 224×224 input resized from the source image, input pixel scaling to 0–1, and a single sigmoid output representing the pneumonia score. These are the publisher's implementation notes, not an independent validation by PulmoSight. The publisher reports external evaluation metrics, but PulmoSight has not independently reproduced those metrics. See [docs/MODEL_CARD.md](docs/MODEL_CARD.md) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

At first launch, the backend attempts to download the pinned file and verify the SHA-256. To download it explicitly ahead of time:

```bash
python scripts/download_model.py
```

To force a fresh download:

```bash
python scripts/download_model.py --force
```

The downloaded file is saved to `model/best_chest_xray_model.h5`, a local-only folder ignored by Git. A failed or mismatched checksum stops loading rather than accepting an unknown file. The app can still open in setup mode when the download fails, but prediction stays unavailable.

## Run locally (Python 3.11 recommended)

1. Clone the repo and create a virtual environment:

   ```bash
   git clone https://github.com/mgtba555333-pixel/- pulmosight-ai
   cd pulmosight-ai
   py -3.11 -m venv .venv
   # Windows PowerShell:
   .\.venv\Scripts\Activate.ps1
   # macOS / Linux:
   source .venv/bin/activate
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

2. Start the app:

   ```bash
   uvicorn app.main:app --reload
   ```

3. Open http://127.0.0.1:8000. Health: http://127.0.0.1:8000/api/health. API docs: http://127.0.0.1:8000/api/docs.

TensorFlow and the checkpoint take disk space; the first startup needs internet access to fetch the model. The checkpoint is stored outside Git, so it is not added to the user's public commit history.

## Docker

```bash
docker compose up --build
```

The `./model` directory is mounted read/write so the verified model download can persist on the host machine. That folder is ignored by Git. If you need offline deployment, run the model download once before starting Docker and preserve the model file locally.

## Model configuration

- `MODEL_PATH`: path to the H5 weights; defaults to `./model/best_chest_xray_model.h5`.
- `MODEL_AUTO_DOWNLOAD=true`: download the pinned checkpoint when the file is missing.
- `XRAY_PREPROCESSING=auto`: uses the model's internal Rescaling layer when present; for the pinned Hugging Face model, the app follows the model card's external 0–1 scaling.
- `XRAY_COLOR_ORDER=rgb`: matches the published inference example.
- The service reads the model's input shape and output shape rather than assuming a two-unit softmax. A one-unit sigmoid output is converted to `NORMAL = 1 - pneumonia_score` and `PNEUMONIA = pneumonia_score`.

Do not interpret model scores as calibrated clinical probabilities. Grad-CAM is only a rough visualization of model gradients, not a lesion segmentation or proof of disease.

## Evaluate on a held-out dataset

Use a **separate held-out test dataset** with folders that match model classes:

```text
test-data/
  NORMAL/
  PNEUMONIA/
```

Install the optional evaluation dependency and run:

```bash
pip install -r requirements-evaluation.txt
python scripts/evaluate_model.py --dataset ./test-data --output ./reports/evaluation-report.json
```

Keep this dataset out of training and threshold selection, and avoid patient overlap between train and test data. The report includes class metrics, specificity, confusion matrix, ROC-AUC, Brier score, ECE, and percentile bootstrap intervals. These are engineering metrics, not evidence of clinical utility.

## Privacy and security

- Uploaded images are not written to an app folder or database; request handling may buffer image bytes temporarily in memory.
- Session history stores result summaries only; it does not store image bytes or patient names.
- Do not upload identifiable patient images to a public or untrusted server.
- Before public deployment, add suitable authentication, rate limiting, HTTPS, logs that exclude image contents, and a privacy/retention review.

## Attribution

The web application/integration code is separate from the model artifact. The default model is obtained from its model host at a fixed revision with SHA-256 verification. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the license attribution. Check the model card and underlying dataset terms before any redistribution or real-world use.
