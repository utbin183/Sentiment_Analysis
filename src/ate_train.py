"""Step 2.2 — DistilBERT ATE fine-tuning on Step 2.1 artifacts.

ONLY train and validation are loaded. The held-out test dataset is never opened.
No train() or model loading at module import time.

Run `python ate_train.py --dry-run` to validate input and inspect the plan
WITHOUT downloading model weights, instantiating a model, or training.
"""

from __future__ import annotations

import argparse
from collections import Counter
import importlib.metadata
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

DATASET_NAME = "thainq107/abte-restaurants"
MODEL_NAME = "distilbert/distilbert-base-uncased"
ID2LABEL = {0: "O", 1: "B-Term", 2: "I-Term"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}
SPLITS_FOR_TRAINING = ("train", "validation")


def read_and_validate_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"Missing Step 2.1 manifest: {path}")
    info = json.loads(path.read_text(encoding="utf-8"))
    if info.get("dataset") != DATASET_NAME:
        raise ValueError(f"Unexpected dataset in manifest: {info.get('dataset')!r}")
    if info.get("tokenizer") != MODEL_NAME:
        raise ValueError("Step 2.1 tokenizer does not match the DistilBERT model for this step")
    if info.get("id2label") != {str(k): v for k, v in ID2LABEL.items()}:
        raise ValueError("Manifest BIO label mapping differs from the source ATE code")
    if info.get("review_overlap_between_final_splits") != 0:
        raise ValueError("Manifest does not confirm zero cross-split review overlap")
    if info.get("training_performed") is not False:
        raise ValueError("Expected preprocessing-only input artifact")
    counts = info.get("split_counts", {})
    if not all(isinstance(counts.get(s), int) and counts[s] > 0 for s in SPLITS_FOR_TRAINING):
        raise ValueError("Manifest has no nonempty train/validation split counts")
    for split in SPLITS_FOR_TRAINING:
        if info.get("tokenization", {}).get(split, {}).get("truncated_aspect_spans") != 0:
            raise ValueError(f"{split} has missing/truncated annotated aspect spans")
        if info.get("tokenization", {}).get(split, {}).get("truncated_reviews") != 0:
            raise ValueError(f"{split} contains truncated reviews; check max_length before training")
    if not isinstance(info.get("max_length"), int) or info["max_length"] < 3:
        raise ValueError("Invalid max_length in preprocessing manifest")
    return info


def normalized_review(tokens: Sequence[str]) -> tuple[str, ...]:
    return tuple(str(tok).strip().casefold() for tok in tokens)


def check_splits(train_tokens: Any, validation_tokens: Any,
                 train_raw: Any, validation_raw: Any,
                 manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Inspect ONLY train and validation, independently confirm group isolation.

    The loader works with HF Dataset objects or a list of dicts for unit tests.
    """
    expected = manifest["split_counts"]
    for split_name, tokenized, raw in (
        ("train", train_tokens, train_raw),
        ("validation", validation_tokens, validation_raw),
    ):
        if len(raw) != expected[split_name] or len(tokenized) != expected[split_name]:
            raise ValueError(f"{split_name} dataset length differs from manifest")

    keys: dict[str, set[tuple[str, ...]]] = {}
    distribution: dict[str, dict[str, int]] = {}
    for name, raw, encoded in (
        ("train", train_raw, train_tokens),
        ("validation", validation_raw, validation_tokens),
    ):
        keys[name] = set()
        count: Counter[int] = Counter()
        for idx in range(len(encoded)):
            original = raw[idx]
            item = encoded[idx]
            words = original["Tokens"]
            word_tags = [int(tag) for tag in original["Tags"]]
            if not words or len(words) != len(word_tags) or any(x not in ID2LABEL for x in word_tags):
                raise ValueError(f"{name}[{idx}]: invalid source word BIO tags")
            key = normalized_review(words)
            if key in keys[name]:
                raise ValueError(f"{name} has repeated normalized reviews")
            keys[name].add(key)
            input_ids = item["input_ids"]
            masks = item["attention_mask"]
            labels = item["labels"]
            if not input_ids or len(input_ids) != len(masks) or len(input_ids) != len(labels):
                raise ValueError(f"{name}[{idx}]: input_ids/attention_mask/labels mismatch")
            if any(not isinstance(label, int) or label not in (-100, 0, 1, 2) for label in labels):
                raise ValueError(f"{name}[{idx}]: illegal aligned label")
            if any(mask not in (0, 1) for mask in masks):
                raise ValueError(f"{name}[{idx}]: illegal attention_mask")
            if not any(label != -100 for label in labels):
                raise ValueError(f"{name}[{idx}]: all tokens are ignored")
            count.update(label for label in labels if label != -100)
        distribution[name] = {ID2LABEL[i]: count[i] for i in ID2LABEL}
    if keys["train"] & keys["validation"]:
        raise ValueError("LEAKAGE: same normalized review in train and validation")
    if not distribution["train"]["B-Term"]:
        raise ValueError("Training split has no annotated aspects")
    if not distribution["validation"]["B-Term"]:
        raise ValueError("Validation split has no annotated aspects")
    return {
        "split_counts": {"train": len(train_tokens), "validation": len(validation_tokens)},
        "train_validation_review_overlap": 0,
        "aligned_label_distribution": distribution,
    }


def load_two_splits(artifact_dir: Path, loader: Callable[[str], Any]) -> tuple[Any, Any, Any, Any]:
    """Deliberately never refer to `test` or call load_from_disk on DatasetDict root."""
    tokenized_dir = artifact_dir / "tokenized_splits"
    raw_dir = artifact_dir / "raw_splits"
    for base in (tokenized_dir, raw_dir):
        if not base.is_dir():
            raise ValueError(f"Missing Step 2.1 saved dataset directory: {base}")
    train_tokens = loader(str(tokenized_dir / "train"))
    validation_tokens = loader(str(tokenized_dir / "validation"))
    train_raw = loader(str(raw_dir / "train"))
    validation_raw = loader(str(raw_dir / "validation"))
    return train_tokens, validation_tokens, train_raw, validation_raw


def first_wordpiece_positions(tokenizer: Any, raw_validation: Any,
                              encoded_validation: Any, max_length: int) -> list[list[int]]:
    """Match saved token IDs to original words for word-level entity/span F1.

    Does not create/modify model input labels or edit the saved artifacts.
    """
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("Training evaluation requires the Step 2.1 fast tokenizer")
    first_positions: list[list[int]] = []
    for idx in range(len(raw_validation)):
        row = raw_validation[idx]
        saved = encoded_validation[idx]
        encoding = tokenizer(
            [row["Tokens"]], is_split_into_words=True, truncation=True,
            max_length=max_length, padding=False, return_attention_mask=True,
            add_special_tokens=True,
        )
        if encoding["input_ids"][0] != saved["input_ids"]:
            raise ValueError(f"validation[{idx}]: locally saved tokenizer disagrees with tokenized input_ids")
        word_ids = encoding.word_ids(batch_index=0)
        positions: list[int] = []
        previous = None
        for pos, word_id in enumerate(word_ids):
            if word_id is not None and word_id != previous:
                positions.append(pos)
            previous = word_id
        if len(positions) != len(row["Tokens"]):
            raise ValueError(f"validation[{idx}]: tokenization lost words")
        for pos, source_tag in zip(positions, row["Tags"]):
            if saved["labels"][pos] != int(source_tag):
                raise ValueError(f"validation[{idx}]: first-subword labels disagree with BIO annotations")
        first_positions.append(positions)
    return first_positions


def bio_entity_spans(tags: Sequence[int], *, strict_gold: bool) -> set[tuple[int, int]]:
    """Exact word offsets from BIO, with explicit invalid-prediction policy.

    An unexpected predicted I-Term starts a new entity (counts as prediction).
    Invalid gold I-Term raises an error. Offsets are half-open.
    """
    spans: set[tuple[int, int]] = set()
    start: int | None = None
    for index, tag in enumerate([*tags, 0]):
        if tag not in ID2LABEL:
            raise ValueError(f"Invalid BIO tag {tag}")
        if tag == 0:
            if start is not None:
                spans.add((start, index))
                start = None
        elif tag == 1:
            if start is not None:
                spans.add((start, index))
            start = index
        elif start is None:
            if strict_gold:
                raise ValueError(f"Gold BIO starts with I-Term at word {index}")
            start = index
    return spans


def exact_span_scores(gold_sequences: Sequence[Sequence[int]],
                      pred_sequences: Sequence[Sequence[int]]) -> dict[str, float | int]:
    """Micro precision/recall/F1 of exact aspect spans in ORIGINAL WORD units."""
    if len(gold_sequences) != len(pred_sequences):
        raise ValueError("Gold/prediction batch size mismatch")
    tp = fp = fn = 0
    for gold, pred in zip(gold_sequences, pred_sequences):
        if len(gold) != len(pred):
            raise ValueError("Gold/prediction word count mismatch")
        a = bio_entity_spans(gold, strict_gold=True)
        b = bio_entity_spans(pred, strict_gold=False)
        tp += len(a & b)
        fp += len(b - a)
        fn += len(a - b)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "span_precision": precision, "span_recall": recall,
        "span_f1": f1, "span_tp": tp, "span_fp": fp, "span_fn": fn,
    }


def make_compute_metrics(positions: Sequence[Sequence[int]]) -> Callable[[Any], dict[str, float | int]]:
    """Transformers Trainer metrics on validation ONLY, grouped by original words."""
    def compute_metrics(eval_prediction: Any) -> dict[str, float | int]:
        import numpy as np
        logits, labels = eval_prediction
        if isinstance(logits, tuple):
            logits = logits[0]
        predicted_tags = np.asarray(logits).argmax(axis=-1)
        if len(predicted_tags) != len(positions) or len(labels) != len(positions):
            raise ValueError("Eval predictions do not match validation row count")
        all_gold: list[list[int]] = []
        all_pred: list[list[int]] = []
        for row_index, firsts in enumerate(positions):
            gold = []
            pred = []
            for pos in firsts:
                if pos >= len(labels[row_index]) or labels[row_index][pos] == -100:
                    raise ValueError("Missing non-ignored label for an original word")
                gold.append(int(labels[row_index][pos]))
                pred.append(int(predicted_tags[row_index][pos]))
            all_gold.append(gold)
            all_pred.append(pred)
        return exact_span_scores(all_gold, all_pred)
    return compute_metrics


def training_plan(*, output_dir: str, epochs: int, batch_size: int, eval_batch_size: int,
                  learning_rate: float, seed: int, patience: int, weight_decay: float,
                  fp16: bool) -> dict[str, Any]:
    if epochs < 1 or batch_size < 1 or eval_batch_size < 1 or patience < 1:
        raise ValueError("epochs, batch sizes and patience must be positive")
    if not (0 < learning_rate < 1):
        raise ValueError("learning_rate must be between 0 and 1")
    if not (0 <= weight_decay < 1):
        raise ValueError("weight_decay must be in [0,1)")
    return {
        "output_dir": output_dir,
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "logging_strategy": "epoch",
        "learning_rate": learning_rate,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": eval_batch_size,
        "num_train_epochs": epochs,
        "weight_decay": weight_decay,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_span_f1",
        "greater_is_better": True,
        "save_total_limit": 2,
        "seed": seed,
        "data_seed": seed,
        "fp16": fp16,
        "report_to": "none",
        "dataloader_num_workers": 0,  # Windows spawn safety
        "save_safetensors": True,
        "optim": "adamw_torch",
    }


def output_can_be_created(output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"Training output is not empty: {output_dir}. Choose a NEW --output directory.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Step 2.2 ATE training: train+validation ONLY; no test set")
    parser.add_argument("--data", default="artifacts/ate_preprocessing", help="Step 2.1 output folder")
    parser.add_argument("--output", default="artifacts/ate_training", help="Fresh directory for training outputs")
    parser.add_argument("--dry-run", action="store_true", help="Validate saved dataset, tokenizer and plan; NO model/train")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=2, help="Number of epochs with no improvement in validation F1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true", help="Only for a CUDA GPU supporting fp16")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    artifact_dir = Path(args.data)
    output = Path(args.output)
    info = read_and_validate_manifest(artifact_dir / "manifest.json")
    plan = training_plan(
        output_dir=str(output / "checkpoints"), epochs=args.epochs,
        batch_size=args.batch_size, eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate, seed=args.seed, patience=args.patience,
        weight_decay=args.weight_decay, fp16=args.fp16,
    )
    output_can_be_created(output)

    try:
        from datasets import load_from_disk
        from transformers import AutoTokenizer
    except (ImportError, OSError) as exc:
        raise RuntimeError("Cannot import datasets/transformers; check requirements_step_2_2.txt") from exc

    print("Loading train and validation only (held-out test is NOT opened).", flush=True)
    train_ds, val_ds, raw_train, raw_val = load_two_splits(artifact_dir, load_from_disk)
    validation = check_splits(train_ds, val_ds, raw_train, raw_val, info)
    tokenizer = AutoTokenizer.from_pretrained(str(artifact_dir / "tokenizer"), use_fast=True, local_files_only=True)
    if not tokenizer.is_fast:
        raise ValueError("Expected a saved fast tokenizer")
    first_positions = first_wordpiece_positions(tokenizer, raw_val, val_ds, max_length=info["max_length"])

    print(json.dumps({
        "status": "PRE-FLIGHT PASSED", "model": MODEL_NAME,
        "data": str(artifact_dir), "validation_checks": validation,
        "training_plan": plan, "early_stopping_patience": args.patience,
        "held_out_test_used": False, "training_performed": False,
    }, indent=2, ensure_ascii=False), flush=True)
    if args.dry_run:
        print("DRY RUN FINISHED (no model weights loaded; no training).")
        return 0

    import torch
    from transformers import (
        AutoModelForTokenClassification, DataCollatorForTokenClassification,
        EarlyStoppingCallback, Trainer, TrainingArguments, set_seed,
    )
    if args.fp16 and not torch.cuda.is_available():
        raise ValueError("--fp16 needs a CUDA GPU; omit --fp16 to train on CPU")
    print("Compute device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU", flush=True)
    set_seed(args.seed)
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME, num_labels=3, id2label=ID2LABEL, label2id=LABEL2ID,
    )
    train_args = TrainingArguments(**plan)
    trainer = Trainer(
        model=model, args=train_args, train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer),
        processing_class=tokenizer,
        compute_metrics=make_compute_metrics(first_positions),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )
    print("Training DistilBERT ATE on training split, evaluating ONLY validation...", flush=True)
    training_result = trainer.train()
    if not trainer.state.best_model_checkpoint or trainer.state.best_metric is None:
        raise RuntimeError("No best validation checkpoint/metric found; model will not be published as best_model")
    # With load_best_model_at_end=True, trainer.model contains best checkpoint weights.
    validation_metrics = trainer.evaluate(eval_dataset=val_ds, metric_key_prefix="eval")
    best_model_dir = output / "best_model"
    trainer.save_model(str(best_model_dir))
    tokenizer.save_pretrained(str(best_model_dir))
    history_path = output / "training_history.json"
    history_path.write_text(json.dumps(trainer.state.log_history, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    versions = {}
    for package in ("torch", "transformers", "datasets", "accelerate"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    summary = {
        "step": "2.2", "training_performed": True,
        "dataset": info["dataset"], "preprocessing_seed": info["seed"],
        "training_seed": args.seed, "model": MODEL_NAME,
        "train_examples": len(train_ds), "validation_examples": len(val_ds),
        "test_used": False, "validation_check": validation,
        "metric_definition": "micro exact aspect-span P/R/F1 over original WORD positions; illegal predicted I starts a new span",
        "training_config": plan, "early_stopping_patience": args.patience,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_validation_span_f1": trainer.state.best_metric,
        "train_result_metrics": training_result.metrics,
        "best_model_validation_metrics": validation_metrics,
        "versions": versions,
        "saved_model_dir": str(best_model_dir),
    }
    (output / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "TRAINING FINISHED", "best_checkpoint": summary["best_checkpoint"],
        "best_validation_span_f1": summary["best_validation_span_f1"],
        "model_dir": summary["saved_model_dir"],
        "summary_path": str(output / "run_summary.json"),
        "test_used": False,
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
