# PulmoSight AI — Development Roadmap

## Current foundation

- [x] Arabic RTL responsive dashboard and upload preview.
- [x] FastAPI health endpoint and model-backed prediction route.
- [x] Explicit setup-required status when weights are missing.
- [x] Upload format and size checks.
- [x] Session-only result summaries without image bytes or patient names.
- [x] Docker setup and static quality checks.

## Phase 1 — Reproducible inference

- [x] Select a pretrained model whose Hugging Face repository advertises an MIT license and document attribution.
- [x] Add a pinned downloader with a fixed revision and SHA-256 verification; keep weights out of Git history.
- [x] Support binary sigmoid model outputs and load the model input dimensions dynamically.
- [x] Inspect the selected model card and implement its documented 224×224 RGB input and 0–1 scaling.
- [x] Validate input/output shapes and preserve already-normalized softmax scores without applying softmax twice.
- [x] Add unit tests for preprocessing dimensions, color order, raw pixel range, and the default configuration.
- [x] Ensure API responses and UI labels match configured model metadata.
- [ ] Download the model in a runtime environment, load the H5 file with TensorFlow 2.19, and verify end-to-end inference against authorized held-out images.

## Phase 2 — Explainability and evaluation

- [x] Add an optional Grad-CAM implementation that searches for the last Conv2D layer.
- [x] Show heatmaps explicitly as model-attention visualizations, never as proof of disease.
- [x] Add a held-out evaluation report with sensitivity/recall, specificity, precision, F1, ROC-AUC, Brier score, ECE, and bootstrap confidence intervals.
- [ ] Split and report data at patient level to reduce leakage.
- [ ] Test external generalization and image quality/unsupported-image handling.

## Phase 3 — Product hardening

- [ ] Add structured server logs that exclude image contents and patient identifiers.
- [ ] Add rate limiting, deployment secrets, HTTPS, and retention policy before public hosting.
- [ ] Add user authentication only if the product genuinely needs it.
- [ ] Add an optional exportable research report that clearly labels model scores as non-clinical outputs.
- [ ] Obtain independent clinical, privacy, and legal review before any real patient use.

## Non-negotiable guardrails

- No fabricated fallback predictions.
- No claim of clinical accuracy until independently evaluated.
- No identifiable patient data in public repositories, logs, or demo fixtures.
- No use for patient diagnosis, triage, or treatment decisions.
