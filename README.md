# PulmoSight AI — Chest X-ray Research Platform

A structured, responsive research web application for reviewing chest X-ray images and connecting a TensorFlow model for **Normal vs Pneumonia** classification.

> **Research prototype only.** This software is not a medical device and must not be used to diagnose, triage, or treat patients. A qualified clinician must interpret radiology findings. The UI never fabricates predictions: if the model weights are absent or cannot load, the app clearly reports that analysis is unavailable.

## What is included

- Arabic, right-to-left, responsive research dashboard.
- Image upload with client-side preview, format and size validation.
- FastAPI backend with health status and a protected image-analysis endpoint.
- Lazy-compatible TensorFlow/Keras inference wrapper and configurable preprocessing.
- Local session history containing result summaries only; images are not stored by the application.
- Docker setup and a clear path to add the model weights separately.
- Explicit model availability state so the website cannot present a fake result.

## Project layout

```text
app/
  main.py              # FastAPI routes and static site
  model_service.py     # Model loading, preprocessing and inference
web/
  index.html           # Dashboard interface
  app.css              # Responsive visual system
  app.js               # Upload, API and interaction logic
model/                 # Add model files here locally; weights are not committed
  xray_model.hdf5
  model_classes.txt
Dockerfile
docker-compose.yml
requirements.txt
.env.example
```

## Run locally (Python 3.11 recommended)

1. Create a virtual environment and install dependencies:

   ```bash
   python -m venv .venv
   # Windows PowerShell:
   .venv\Scripts\Activate.ps1
   # macOS / Linux:
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Create the model folder and place the model files in it:

   ```text
   model/xray_model.hdf5
   model/model_classes.txt
   ```

   The upstream research project is [inuwamobarak/chest-xray-pneumonia-detection](https://github.com/inuwamobarak/chest-xray-pneumonia-detection). Review its current license and model/data terms before reusing or redistributing its weights. Do not commit large model weights or patient data to this repository.

3. Start the site:

   ```bash
   uvicorn app.main:app --reload
   ```

4. Open http://127.0.0.1:8000. API health is available at http://127.0.0.1:8000/api/health and interactive API docs at http://127.0.0.1:8000/api/docs.

If the model file is not present, the website still runs in setup mode, but the analysis endpoint will return a clear `503 model_not_ready` response instead of inventing a prediction.

## Docker

```bash
docker compose up --build
```

Put the two model files in `./model/` first. Docker mounts that directory read-only into the container.

## Model and evaluation notes

- The model is loaded from `MODEL_PATH`; label names are read from `MODEL_CLASSES_PATH`.
- `XRAY_PREPROCESSING` supports `raw` (the upstream inference script's apparent raw 0–255 input), `unit` (0–1 scaling), `resnet50`, and `mobilenet_v2`. **This must be matched to the exact preprocessing used during training before interpreting outputs.**
- The original project contains legacy inference code. Validate input dimensions, color channel order, label ordering, calibration, patient-level leakage, sensitivity, specificity, and external generalization before any research claims.
- Confidence values are model scores, not calibrated probabilities and not a measure of clinical certainty.

## Privacy

The API does not persist uploaded images to an application folder or database; upload handling may buffer data temporarily while the request is processed. The browser's recent-results list stores summary data only in the current tab session; it does not store image bytes or patient names. Do not upload identifiable patient images to a public or untrusted server.

## Attribution

This project is inspired by and interoperates with the research implementation linked above. The website and integration code in this repository are a separate development layer. Review upstream licenses before copying code or weights.
