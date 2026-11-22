"""Step 2.3 — one-time, held-out ATE test evaluation (NO training).

Consumes the exact Step 2.1 test split and Step 2.2 best_model artifacts.
Reports MICRO exact-match aspect-span scores at ORIGINAL WORD boundaries.

Use --preflight to inspect artifact metadata without opening the test examples
or loading model weights. Test evaluation should happen only once after model
and hyperparameters are frozen. No ATSC, no Streamlit, no training.
"""
from __future__ import annotations

# Transformers 4.x may otherwise attempt to import installed TensorFlow/Keras 3.
# This evaluation uses *only* the PyTorch model.
import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

import argparse
from collections import Counter
from datetime import datetime, timezone
import importlib.metadata
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from src.ate_train import (
    DATASET_NAME,
    ID2LABEL,
    MODEL_NAME,
    bio_entity_spans,
    exact_span_scores,
    first_wordpiece_positions,
    normalized_review,
    read_and_validate_manifest,
)

EXPECTED_SPLIT = "test"
EXPECTED_METRIC = "eval_span_f1"


def require_artifact_files(data_dir: Path, model_dir: Path, training_summary: Path,
                           output_dir: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Check metadata/paths before any test dataset or model weights are loaded."""
    if output_dir.exists():
        raise ValueError(f"Evaluation output already exists: {output_dir}. Use a NEW --output path; never overwrite a test result.")
    if not training_summary.is_file():
        raise ValueError(f"Missing training summary: {training_summary}")
    if not model_dir.is_dir():
        raise ValueError(f"Missing best_model folder: {model_dir}")
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        if not (model_dir / name).is_file():
            raise ValueError(f"Missing model/tokenizer file: {model_dir / name}")
    if not any((model_dir / name).is_file() for name in ("model.safetensors", "pytorch_model.bin")):
        raise ValueError(f"Missing model weights in {model_dir}")
    for branch in ("raw_splits", "tokenized_splits"):
        if not (data_dir / branch / EXPECTED_SPLIT).is_dir():
            raise ValueError(f"Missing held-out test split: {data_dir / branch / EXPECTED_SPLIT}")
    if not (data_dir / "raw_splits" / "train").is_dir() or not (data_dir / "raw_splits" / "validation").is_dir():
        raise ValueError("Missing raw train/validation splits needed for leakage verification")
    if manifest.get("tokenization", {}).get("test", {}).get("truncated_reviews") != 0:
        raise ValueError("Test reviews have been truncated; exact-span metrics may be invalid")
    if manifest.get("tokenization", {}).get("test", {}).get("truncated_aspect_spans") != 0:
        raise ValueError("Test aspect spans were truncated; exact-span metrics may be invalid")
    test_count = manifest.get("split_counts", {}).get("test")
    if not isinstance(test_count, int) or test_count < 1:
        raise ValueError("Preprocessing manifest lacks a positive test count")

    training = json.loads(training_summary.read_text(encoding="utf-8"))
    if training.get("training_performed") is not True or training.get("test_used") is not False:
        raise ValueError("Training summary does not confirm completed, test-free training")
    if training.get("model") != MODEL_NAME or training.get("dataset") != DATASET_NAME:
        raise ValueError("Training model/dataset disagrees with Step 2.1 preprocessing")
    if training.get("train_examples") != manifest["split_counts"]["train"] or training.get("validation_examples") != manifest["split_counts"]["validation"]:
        raise ValueError("Training example counts disagree with preprocessing manifest")
    config = training.get("training_config", {})
    if config.get("metric_for_best_model") != EXPECTED_METRIC or config.get("load_best_model_at_end") is not True:
        raise ValueError("Training summary did not select best model by validation span-F1")
    if not training.get("best_checkpoint"):
        raise ValueError("Training summary lacks best_checkpoint")
    best_val = training.get("best_validation_span_f1")
    if not isinstance(best_val, (int, float)) or not math.isfinite(best_val) or not (0 <= best_val <= 1):
        raise ValueError("Training summary lacks a valid best validation span-F1")

    return {
        "preflight_status": "PASSED",
        "source_dataset": DATASET_NAME,
        "model": MODEL_NAME,
        "best_checkpoint_from_training": training["best_checkpoint"],
        "best_validation_span_f1": best_val,
        "held_out_test_examples_expected": test_count,
        "metric_definition": "micro exact-match aspect term spans at original-word positions",
        "test_opened": False,
        "model_weights_loaded": False,
        "training_performed": False,
    }


def verify_test_splits(raw_test: Any, tokenized_test: Any,
                       raw_train: Any, raw_validation: Any,
                       manifest: Mapping[str, Any]) -> dict[str, int]:
    """Independently check test size, labels, saved inputs, and review isolation."""
    expected_counts = manifest["split_counts"]
    for name, data in (("test(raw)", raw_test), ("test(tokenized)", tokenized_test)):
        if len(data) != expected_counts["test"]:
            raise ValueError(f"{name} length disagrees with preprocessing manifest")
    for name, data in (("train(raw)", raw_train), ("validation(raw)", raw_validation)):
        split = name.split("(")[0]
        if len(data) != expected_counts[split]:
            raise ValueError(f"{name} length disagrees with preprocessing manifest")

    reference_keys: set[tuple[str, ...]] = set()
    for name, data in (("train", raw_train), ("validation", raw_validation)):
        for index in range(len(data)):
            row = data[index]
            key = normalized_review(row["Tokens"])
            if key in reference_keys:
                raise ValueError(f"Repeated normalized review among raw train/validation: {name}[{index}]")
            reference_keys.add(key)

    test_keys: set[tuple[str, ...]] = set()
    gold_spans = 0
    for index in range(len(raw_test)):
        row, tokenized = raw_test[index], tokenized_test[index]
        words, tags = row["Tokens"], [int(x) for x in row["Tags"]]
        if not words or len(words) != len(tags):
            raise ValueError(f"test[{index}]: word/tag length mismatch")
        gold_spans += len(bio_entity_spans(tags, strict_gold=True))
        key = normalized_review(words)
        if key in reference_keys:
            raise ValueError(f"LEAKAGE: test[{index}] overlaps train/validation")
        if key in test_keys:
            raise ValueError(f"Duplicate normalized test review: test[{index}]")
        test_keys.add(key)
        ids, masks, labels = tokenized["input_ids"], tokenized["attention_mask"], tokenized["labels"]
        if not ids or len(ids) != len(masks) or len(ids) != len(labels):
            raise ValueError(f"test[{index}]: invalid saved tokenized input lengths")
        if any(v not in (0, 1) for v in masks) or not all(v == 1 for v in masks):
            raise ValueError(f"test[{index}]: expected unpadded attention mask of all ones")
        if any(v not in (-100, 0, 1, 2) for v in labels):
            raise ValueError(f"test[{index}]: illegal BIO label in saved data")
    return {"test_examples": len(raw_test), "gold_aspect_count": gold_spans, "test_review_overlap_with_train_validation": 0}


def check_model_labels(config: Any) -> None:
    """Fail if someone accidentally points to the wrong model/checkpoint."""
    raw = config.id2label
    observed = {int(k): v for k, v in raw.items()}
    if getattr(config, "num_labels", None) != 3 or observed != ID2LABEL:
        raise ValueError(f"Unexpected model label mapping: {observed}; expected {ID2LABEL}")


def iter_model_batches(tokenized: Any, *, batch_size: int, pad_token_id: int,
                       device: Any, model: Any) -> Iterable[list[list[int]]]:
    """Run torch inference only. No Trainer, no train(), no gold labels in forward().

    This code dynamically right-pads input_ids and attention_mask, then removes
    padded predictions before alignment with the original words.
    """
    import torch
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if pad_token_id is None:
        raise ValueError("Tokenizer must provide a pad token ID")
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(tokenized), batch_size):
            rows = [tokenized[i] for i in range(start, min(start + batch_size, len(tokenized)))]
            lengths = [len(row["input_ids"]) for row in rows]
            longest = max(lengths)
            ids = [row["input_ids"] + [pad_token_id] * (longest - len(row["input_ids"])) for row in rows]
            masks = [row["attention_mask"] + [0] * (longest - len(row["attention_mask"])) for row in rows]
            batch_inputs = {
                "input_ids": torch.tensor(ids, dtype=torch.long, device=device),
                "attention_mask": torch.tensor(masks, dtype=torch.long, device=device),
            }
            # Gold labels deliberately NOT sent into forward(); no optimizer exists.
            outputs = model(**batch_inputs)
            logits = outputs.logits
            if len(logits.shape) != 3 or logits.shape[0] != len(rows) or logits.shape[2] != 3:
                raise ValueError("Model returned incompatible token logits")
            predicted = logits.argmax(dim=-1).detach().cpu().tolist()
            yield [pred[:length] for pred, length in zip(predicted, lengths)]


def word_tags_from_token_predictions(raw_test: Any, tokenized_test: Any,
                                     first_positions: Sequence[Sequence[int]],
                                     token_predictions: Sequence[Sequence[int]]) -> tuple[list[list[int]], list[list[int]]]:
    """Project first-subword logits to words exactly as in Step 2.2 validation."""
    n = len(raw_test)
    if len(tokenized_test) != n or len(first_positions) != n or len(token_predictions) != n:
        raise ValueError("Prediction and test row counts differ")
    golds: list[list[int]] = []
    preds: list[list[int]] = []
    for index in range(n):
        raw = raw_test[index]
        encoded = tokenized_test[index]
        gold = [int(x) for x in raw["Tags"]]
        positions = first_positions[index]
        token_tags = token_predictions[index]
        if len(positions) != len(gold) or len(token_tags) != len(encoded["input_ids"]):
            raise ValueError(f"test[{index}]: token/word alignment size mismatch")
        pred: list[int] = []
        for word_idx, token_idx in enumerate(positions):
            if encoded["labels"][token_idx] != gold[word_idx]:
                raise ValueError(f"test[{index}]: gold label alignment mismatch")
            predicted = int(token_tags[token_idx])
            if predicted not in ID2LABEL:
                raise ValueError(f"test[{index}]: invalid predicted tag {predicted}")
            pred.append(predicted)
        golds.append(gold)
        preds.append(pred)
    return golds, preds


def display_spans(words: Sequence[str], tag_sequence: Sequence[int], *, strict_gold: bool) -> list[dict[str, Any]]:
    spans = bio_entity_spans(tag_sequence, strict_gold=strict_gold)
    return [
        {"start_word": start, "end_word": end, "term": " ".join(words[start:end])}
        for start, end in sorted(spans)
    ]


def make_error_examples(raw_test: Any, gold_tags: Sequence[Sequence[int]],
                        predicted_tags: Sequence[Sequence[int]],
                        max_examples: int) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Deterministically report review-level errors; exact word offsets are half-open."""
    if len(raw_test) != len(gold_tags) or len(raw_test) != len(predicted_tags):
        raise ValueError("Error analysis row counts differ")
    if max_examples < 0:
        raise ValueError("max_examples must be >= 0")
    mistakes = 0
    affected_fp = affected_fn = 0
    examples: list[dict[str, Any]] = []
    for idx, (gold, pred) in enumerate(zip(gold_tags, predicted_tags)):
        words = raw_test[idx]["Tokens"]
        g = bio_entity_spans(gold, strict_gold=True)
        p = bio_entity_spans(pred, strict_gold=False)
        false_pos, missed = p - g, g - p
        if false_pos or missed:
            mistakes += 1
            affected_fp += bool(false_pos)
            affected_fn += bool(missed)
            if len(examples) < max_examples:
                examples.append({
                    "test_index": idx,
                    "review_tokens": list(words),
                    "gold_spans": display_spans(words, gold, strict_gold=True),
                    "predicted_spans": display_spans(words, pred, strict_gold=False),
                    "false_positive_spans": [
                        {"start_word": s, "end_word": e, "term": " ".join(words[s:e])}
                        for s, e in sorted(false_pos)
                    ],
                    "missed_gold_spans": [
                        {"start_word": s, "end_word": e, "term": " ".join(words[s:e])}
                        for s, e in sorted(missed)
                    ],
                })
    return examples, {
        "test_reviews_with_any_span_error": mistakes,
        "test_reviews_with_false_positive": affected_fp,
        "test_reviews_with_missed_gold": affected_fn,
        "stored_error_examples": len(examples),
    }


def make_report(manifest: Mapping[str, Any], training: Mapping[str, Any],
                validation: Mapping[str, int], scores: Mapping[str, float | int],
                errors: Mapping[str, int], *, model_dir: Path,
                test_examples: int, batch_size: int, device: str) -> dict[str, Any]:
    return {
        "step": "2.3",
        "status": "TEST EVALUATION FINISHED",
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": training["model"],
        "model_dir": str(model_dir),
        "checkpoint_selected_before_test": training["best_checkpoint"],
        "checkpoint_selection_metric": EXPECTED_METRIC,
        "best_validation_span_f1_prior_to_test": training["best_validation_span_f1"],
        "dataset": manifest["dataset"],
        "split_evaluated": EXPECTED_SPLIT,
        "test_examples": test_examples,
        "leakage_validation": validation,
        "metric_definition": "micro exact-match aspect spans over ORIGINAL word indices; (start,end) half-open, matched within each review; a predicted I-Term without preceding B-Term begins a new span",
        "test_metrics": dict(scores),
        "error_analysis_counts": dict(errors),
        "inference_batch_size": batch_size,
        "inference_device": device,
        "test_used": True,
        "model_trained_during_evaluation": False,
        "atsc_performed": False,
    }


def choose_device(requested: str, torch_module: Any) -> str:
    if requested == "cpu":
        return "cpu"
    available = bool(torch_module.cuda.is_available())
    if requested == "cuda" and not available:
        raise ValueError("--device cuda requested, but torch.cuda.is_available() is False")
    return "cuda" if requested == "cuda" or (requested == "auto" and available) else "cpu"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Step 2.3 one-time ATE held-out test evaluation; NO training")
    parser.add_argument("--data", default="artifacts/ate_preprocessing")
    parser.add_argument("--model", default="artifacts/ate_training/best_model")
    parser.add_argument("--training-summary", default="artifacts/ate_training/run_summary.json")
    parser.add_argument("--output", default="artifacts/ate_evaluation")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-error-examples", type=int, default=25)
    parser.add_argument("--preflight", action="store_true", help="Check metadata and files only; do NOT open test examples/model weights")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.batch_size < 1 or args.max_error_examples < 0:
        raise ValueError("--batch-size must be positive and --max-error-examples must be >= 0")
    data_dir, model_dir = Path(args.data), Path(args.model)
    summary_path, output_dir = Path(args.training_summary), Path(args.output)
    manifest = read_and_validate_manifest(data_dir / "manifest.json")
    preflight = require_artifact_files(data_dir, model_dir, summary_path, output_dir, manifest)
    print(json.dumps(preflight, ensure_ascii=False, indent=2), flush=True)
    if args.preflight:
        print("PREFLIGHT FINISHED (test data not opened; no model weights loaded; no training).")
        return 0

    # Do not load DatasetDict root, which might silently mix splits.
    from datasets import load_from_disk
    from transformers import AutoModelForTokenClassification, AutoTokenizer
    import torch
    print("Loading held-out test split for FINAL evaluation (no training).", flush=True)
    raw_test = load_from_disk(str(data_dir / "raw_splits" / "test"))
    tokenized_test = load_from_disk(str(data_dir / "tokenized_splits" / "test"))
    # Read-only duplicate check; these train/val rows never enter the model or evaluation metric.
    raw_train = load_from_disk(str(data_dir / "raw_splits" / "train"))
    raw_val = load_from_disk(str(data_dir / "raw_splits" / "validation"))
    validation = verify_test_splits(raw_test, tokenized_test, raw_train, raw_val, manifest)

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True, local_files_only=True)
    if not tokenizer.is_fast or tokenizer.padding_side != "right":
        raise ValueError("Expected a saved FAST right-padding tokenizer from DistilBERT")
    positions = first_wordpiece_positions(tokenizer, raw_test, tokenized_test, manifest["max_length"])
    model = AutoModelForTokenClassification.from_pretrained(str(model_dir), local_files_only=True)
    check_model_labels(model.config)
    selected_device = choose_device(args.device, torch)
    model.to(selected_device)
    print(f"Inference device: {selected_device}; held-out test reviews: {len(raw_test)}", flush=True)

    token_predictions: list[list[int]] = []
    for batch in iter_model_batches(tokenized_test, batch_size=args.batch_size,
                                    pad_token_id=tokenizer.pad_token_id,
                                    device=selected_device, model=model):
        token_predictions.extend(batch)

    gold, predicted = word_tags_from_token_predictions(raw_test, tokenized_test, positions, token_predictions)
    scores = exact_span_scores(gold, predicted)
    examples, error_counts = make_error_examples(raw_test, gold, predicted, args.max_error_examples)
    training_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    report = make_report(manifest, training_summary, validation, scores, error_counts,
                         model_dir=model_dir, test_examples=len(raw_test),
                         batch_size=args.batch_size, device=selected_device)
    versions: dict[str, str] = {}
    for package in ("torch", "transformers", "datasets"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    report["library_versions"] = versions

    # Only write reports after successful COMPLETE evaluation; refuses overwrite.
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "test_metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    (output_dir / "error_examples.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False, allow_nan=False) + "\n" for x in examples), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "test_examples": len(raw_test),
        "span_precision": scores["span_precision"],
        "span_recall": scores["span_recall"],
        "span_f1": scores["span_f1"],
        "span_tp": scores["span_tp"],
        "span_fp": scores["span_fp"],
        "span_fn": scores["span_fn"],
        "report": str(output_dir / "test_metrics.json"),
        "error_examples": str(output_dir / "error_examples.jsonl"),
        "model_trained_during_evaluation": False,
    }, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
