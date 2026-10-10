# PulmoSight AI — Development Roadmap

## Current foundation

- [x] Arabic RTL responsive dashboard and upload preview.
- [x] FastAPI health endpoint and model-backed prediction route.
- [x] Explicit setup-required status when weights are missing.
- [x] Upload format and size checks.
- [x] Session-only result summaries without image bytes or patient names.
- [x] Docker setup and static quality checks.

## Phase 1 — Reproducible inference

- [ ] Confirm rights and terms for the upstream model weights and dataset.
- [x] Inspect the training notebook and record the input size, color order, internal scaling, and output activation.
- [x] Validate input/output shapes and preserve already-normalized softmax scores without applying softmax twice.
- [x] Add unit tests for preprocessing dimensions, color order, raw pixel range, and the default configuration.
- [x] Ensure API responses and UI labels match configured model metadata.
- [ ] Obtain model/data license clarity and run end-to-end inference against the exact weights.

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
