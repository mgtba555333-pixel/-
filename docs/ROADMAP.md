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
- [ ] Inspect the training notebook and record the exact input size, color order, scaling, and class order.
- [ ] Verify the model's output shape and whether outputs are logits or already normalized.
- [ ] Add a tiny, non-identifiable public test fixture and an integration test for inference.
- [ ] Ensure API responses and UI labels match the verified model metadata.

## Phase 2 — Explainability and evaluation

- [ ] Add Grad-CAM only after confirming the correct convolutional layer.
- [ ] Show heatmaps as model-attention visualizations, never as proof of disease.
- [ ] Create an evaluation report with sensitivity, specificity, precision, recall, F1, ROC-AUC, calibration, and confidence intervals.
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
