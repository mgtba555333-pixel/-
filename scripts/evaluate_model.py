from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score

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


def expected_calibration_error(binary_truth: list[int], positive_scores: list[float], bins: int = 10) -> float:
    truth = np.asarray(binary_truth, dtype=np.int32)
    scores = np.asarray(positive_scores, dtype=np.float64)
    if len(truth) == 0:
        return 0.0
    ece = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        if index == bins - 1:
            selected = (scores >= lower) & (scores <= upper)
        else:
            selected = (scores >= lower) & (scores < upper)
        if np.any(selected):
            ece += float(np.mean(selected)) * abs(
                float(np.mean(scores[selected])) - float(np.mean(truth[selected]))
            )
    return ece


def bootstrap_confidence_intervals(
    y_true: list[int],
    y_pred: list[int],
    positive_scores: list[float] | None,
    num_classes: int,
    resamples: int = 500,
    seed: int = 42,
) -> dict[str, Any]:
    """Percentile bootstrap intervals for metrics, with a fixed seed for reproducibility."""
    actual = np.asarray(y_true, dtype=np.int32)
    predicted = np.asarray(y_pred, dtype=np.int32)
    if len(actual) < 2:
        return {}
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = {"accuracy": [], "macro_f1": []}
    if positive_scores is not None:
        values["roc_auc_pneumonia_vs_normal"] = []
        scores = np.asarray(positive_scores, dtype=np.float64)
    labels = list(range(num_classes))

    for _ in range(resamples):
        sample = rng.integers(0, len(actual), size=len(actual))
        truth_sample = actual[sample]
        pred_sample = predicted[sample]
        values["accuracy"].append(float(accuracy_score(truth_sample, pred_sample)))
        values["macro_f1"].append(float(
            f1_score(truth_sample, pred_sample, labels=labels, average="macro", zero_division=0)
        ))
        if positive_scores is not None and len(np.unique(truth_sample)) == 2:
            values["roc_auc_pneumonia_vs_normal"].append(float(
                roc_auc_score((truth_sample == 1).astype(np.int32), scores[sample])
            ))

    intervals: dict[str, Any] = {}
    for metric, samples in values.items():
        if len(samples) < 25:
            intervals[metric] = {
                "lower_95": None,
                "upper_95": None,
                "valid_resamples": len(samples),
            }
        else:
            intervals[metric] = {
                "lower_95": round(float(np.percentile(samples, 2.5)), 6),
                "upper_95": round(float(np.percentile(samples, 97.5)), 6),
                "valid_resamples": len(samples),
            }
    return intervals


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

    # Scores come from the model head and are not established as calibrated clinical probabilities.
    pneumonia_index = next(
        (i for i, label in enumerate(service.labels) if "PNEUMONIA" in label.upper()),
        None,
    )
    ci_scores: list[float] | None = None
    if len(service.labels) == 2 and pneumonia_index is not None and len(set(y_true)) == 2:
        binary_truth = [int(value == pneumonia_index) for value in y_true]
        binary_scores = [values[pneumonia_index] for values in scores_by_class]
        ci_scores = binary_scores
        metrics["roc_auc_pneumonia_vs_normal"] = round(
            float(roc_auc_score(binary_truth, binary_scores)), 6
        )
        metrics["brier_score_pneumonia"] = round(
            float(np.mean((np.asarray(binary_scores) - np.asarray(binary_truth)) ** 2)), 6
        )
        metrics["expected_calibration_error_10_bins"] = round(
            expected_calibration_error(binary_truth, binary_scores, bins=10), 6
        )
    else:
        metrics["roc_auc_pneumonia_vs_normal"] = None
        metrics["brier_score_pneumonia"] = None
        metrics["expected_calibration_error_10_bins"] = None

    metrics["confidence_intervals_95_bootstrap"] = bootstrap_confidence_intervals(
        y_true,
        y_pred,
        ci_scores,
        num_classes=len(service.labels),
    )

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
