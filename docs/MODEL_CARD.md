# Model Card — Upstream Chest X-ray Classifier

## Purpose and intended use

This is an inherited educational research classifier for two image folders/classes: `NORMAL` and `PNEUMONIA`. It is not validated for clinical deployment and must not be used for diagnosis, triage, or treatment decisions.

Upstream source: [inuwamobarak/chest-xray-pneumonia-detection](https://github.com/inuwamobarak/chest-xray-pneumonia-detection).

## What was verified from the upstream notebook

The checked-in training notebook `notebooks/chest_xray.ipynb` contains the following configuration:

- Input resize: 180 × 180.
- Training data loader: `image_dataset_from_directory`, whose directory class order is exposed as `train_ds.class_names`.
- Model begins with a `Rescaling(1./255)` layer.
- The CNN includes Conv2D feature layers and ends with a two-unit Dense layer using `softmax`.
- The visible notebook sets `epochs=3`.
- The notebook assigns a folder named `Test` to the `validation_data` argument during training.

The service's default input configuration therefore uses direct 180 × 180 bilinear resize, RGB channels, and 0–1 scaling. These settings should be re-confirmed against the exact weight file before relying on any output.

## Known concerns

1. **Loss/activation mismatch:** the notebook declares `Dense(num_classes, activation='softmax')` and compiles with `SparseCategoricalCrossentropy(from_logits=True)`. This setting is inconsistent: `from_logits=True` is intended for raw logits, not probabilities already produced by softmax. The trained weights may be affected.
2. **Validation protocol:** the folder named `Test` is used as validation data during `fit`; the notebook does not show a separate untouched, independent final test protocol.
3. **Short training run:** the visible notebook trains for only three epochs. This alone is insufficient to establish model quality.
4. **Legacy inference path:** the upstream inference script applies OpenCV channel conversion after obtaining a PIL image array and uses `ImageOps.fit`; these choices do not match the notebook's RGB directory loader and direct image resize. PulmoSight uses the training-notebook path by default, not these apparent inference discrepancies.
5. **Calibration and generalization:** the repository does not establish calibrated probabilities, a reliable operating threshold, robust external generalization, or clinical utility.

## Explainability

Grad-CAM is generated from the last `Conv2D` layer available in the loaded model. It is a rough visualization of activation gradients for the selected class, not a lesion boundary, pathology finding, or proof that the model relied on clinically relevant anatomy. The user interface must keep this limitation visible. If Grad-CAM cannot be generated, the app shows that it is unavailable rather than fabricating a heatmap.

## Evaluation protocol

Use the separate evaluation script only on a held-out folder structure with one folder per model class:

```text
test-data/
  NORMAL/
  PNEUMONIA/
```

Before running, verify the image labels, remove any patient overlap with the training set, check class balance, and keep this evaluation set out of model fitting and threshold selection. The report computes accuracy, class-wise precision/recall/F1, one-vs-rest specificity, a confusion matrix, and ROC-AUC when the two-class data contain both classes. Report metrics with uncertainty estimates and external validation in any future scientific work.

## Privacy, fairness, and distribution shift

The source dataset is described by the upstream project as pediatric chest radiographs. A model trained on such data should not be assumed to generalize to adults, other hospitals, scanner types, acquisition views, or patient populations. Avoid uploading identifiable patient data to public demos or untrusted services.

## License and provenance

Review the upstream repository, model, and dataset licenses/terms before reuse or redistribution. This repository keeps pretrained weights out of Git; local developers must obtain and place authorized files separately.

## Bottom line

The integration is an engineering and research baseline. Its outputs are not clinically validated; independent medical and statistical review is required before any patient-facing use.
