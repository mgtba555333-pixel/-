# PulmoSight AI — Web Deployment

The repository is a FastAPI + static frontend application. The verified H5 model is committed at `model/best_chest_xray_model.h5`.

## Container deployment

The repository includes `Dockerfile` and `docker-compose.yml`. For a managed deployment, the included `render.yaml` describes a Docker web service with:

- health check: `/api/health`
- the committed model file used directly from the image
- automatic model download disabled

Because TensorFlow is memory-intensive, choose a deployment plan with enough RAM for the pinned TensorFlow 2.19 runtime before exposing the service publicly.

## Required production hardening

Before using a public URL, add HTTPS, authentication if needed, rate limiting, operational logs that exclude image contents and patient identifiers, and a documented retention/privacy policy.

This project is a research prototype. Do not use it to diagnose, triage, or treat patients.
