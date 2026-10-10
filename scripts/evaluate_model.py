from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def build_dataset(root: Path, labels: list[str]) -> list[tuple[int, Path]]:
    if not root.is_dir():
        raise ValueError(f"Dataset directory does not exist: {root}")

    dirs = {item.name.strip().upper(): item for item in root.iterdir() if item.is_dir()}
    missing = [label for label in labels if label.upper() not in dirs]
    if missing:
        raise ValueError(
            "Missing class folders for model labels: " + ", ".join(missing) +
            ". Expected one folder per exact class name inside the dataset root."
        )

    samples: list[tuple[int, Path]] = []
    for label_index, label in enumerate(labels):
        class_dir = dirs[label.upper()]
        files = sorted(
            item for item in class_dir.rglob("*")
            if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
        )
        if not files:
            raise ValueError(f"No supported images found in class folder: {class_dir.name}")
        samples.extend((label_index, path) for path in files)

    return samples


def compute_specificity(confusion: np.ndarray, labels: list[str]) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    total = int(confusion.sum())
    for index, label in enumerate(labels):
        tp = int(confusion[index, index])
        fn = int(confusion[index, :].sum()) - tp
        fp = int(confusion[:, index].sum()) - tp
        tn = total - tp - fn - fp
        denominator = tn + fp
        result[label] = round(tn / denominator, 6) if denominator else None
    return result


def evaluate(dataset_dir: Path, output_path: Path) -> dict[str, Any]:
    # Import lazily so --help and unit-testable metric helpers do not need TensorFlow to import.
    from app.model_service import ModelService

    service = ModelService()
    if not service.load() or not service.ready:
        raise RuntimeError("Model is not ready: " + service.unavailable_message)

    samples = build_dataset(dataset_dir, service.labels)
    y_true: list[int] = []
    y_pred: list[int] = []
    scores_by_class: list[list[float]] = []
    failures: list[dict[str, str]] = []

    for index, (label_index, path) in enumerate(samples, start=1):
        try:
            with Image.open(path) as source:
                image = source.convert("RGB")
            result = service.predict(image, include_explanation=False)
            score_map = {item["label"]: float(item["score"]) for item in result["scores"]}
            y_true.append(label_index)
            y_pred.append(service.labels.index(result["label"]))
            scores_by_class.append([score_map[label] for label in service.labels])
        except (OSError, UnidentifiedImageError, ValueError, RuntimeError) as exc:
            failures.append({"file": path.name, "reason": str(exc)[:240]})

        if index % 25 == 0 or index == len(samples):
            print(f"Evaluated {index}/{len(samples)} images...", flush=True)

    if not y_true:
        raise RuntimeError("No images were successfully evaluated.")

    label_indices = list(range(len(service.labels)))
    matrix = confusion_matrix(y_true, y_pred, labels=label_indices)
    report = classification_report(
        y_true,
        y_pred,
        labels=label_indices,
        target_names=service.labels,
        output_dict=True,
        zero_division=0,
    )

    metrics: dict[str, Any] = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "macro_precision": round(float(report["macro avg"]["precision"]), 6),
        "macro_recall": round(float(report["macro avg"]["recall"]), 6),
        "macro_f1": round(float(report["macro avg"]["f1-score"]), 6),
        "specificity_by_class": compute_specificity(matrix, service.labels),
    }

    # The model scores are not known to be clinically calibrated. AUC measures ranking only.
    pneumonia_index = next(
        (i for i, label in enumerate(service.labels) if "PNEUMONIA" in label.upper()),
        None,
    )
    if len(service.labels) == 2 and pneumonia_index is not None and len(set(y_true)) == 2:
        binary_truth = [int(value == pneumonia_index) for value in y_true]
        binary_scores = [values[pneumonia_index] for values in scores_by_class]
        metrics["roc_auc_pneumonia_vs_normal"] = round(
            float(roc_auc_score(binary_truth, binary_scores)), 6
        )
    else:
        metrics["roc_auc_pneumonia_vs_normal"] = None

    report_data: dict[str, Any] = {
        "title": "PulmoSight AI — research evaluation report",
        "model_labels_in_output_order": service.labels,
        "preprocessing": service.preprocessing,
        "color_order": service.color_order,
        "input_size": service.health().get("input_size"),
        "images_found": len(samples),
        "images_evaluated": len(y_true),
        "images_skipped": len(failures),
        "metrics": metrics,
        "confusion_matrix": {
            "labels_in_row_and_column_order": service.labels,
            "values": matrix.tolist(),
        },
        "classification_report": report,
        "skipped_files": failures,
        "limitations": [
            "Research/engineering metrics only; not evidence of clinical utility.",
            "Use an independent, patient-level held-out test set with verified labels.",
            "Check for patient overlap and distribution shift before interpreting results.",
            "Model output scores are not established as calibrated clinical probabilities.",
            "Do not use these results for patient diagnosis, triage, or treatment.",
        ],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report_data, ensure_ascii=False, indent=2), encoding="utf-8")
    return report_data


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate PulmoSight model predictions on a held-out folder dataset."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Folder with one subfolder per exact model class, e.g. NORMAL/ and PNEUMONIA/.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/evaluation-report.json"),
        help="Path for JSON report (default: reports/evaluation-report.json).",
    )
    args = parser.parse_args()

    try:
        report = evaluate(args.dataset, args.output)
    except Exception as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        return 1

    print("\nEvaluation summary")
    print(f"Images evaluated: {report['images_evaluated']} / {report['images_found']}")
    for key, value in report["metrics"].items():
        print(f"{key}: {value}")
    print(f"Report written to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
