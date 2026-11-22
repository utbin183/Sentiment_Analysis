"""Step 3.3: fine-tune DistilBERT for Aspect Term Sentiment Classification.

Train/validation ONLY. Reads artifacts/atsc_preprocessing/{manifest.json,
 tokenized_splits/train, tokenized_splits/validation, tokenizer}. The held-out
 test directory is never opened or used for checkpoint selection.

Dry-run validates inputs/configuration and NEVER loads model weights or trains.
"""
from __future__ import annotations

# The ATSC project uses PyTorch. Prevent the Transformers/Keras 3 import issue
# in Anaconda; set these before ANY import from transformers.
import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

import argparse
from collections import Counter
import importlib.metadata
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

SPLITS_FOR_TRAINING = ("train", "validation")
ALL_SPLITS = ("train", "validation", "test")
DATASET_NAME = "thainq107/abte-restaurants"
MODEL_NAME = "distilbert/distilbert-base-uncased"
ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}
LABEL2ID = {name: idx for idx, name in ID2LABEL.items()}
TARGET_MARKERS = {"[unused1]": 2, "[unused2]": 3}


def _integer(value: Any, field: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{field}: expected integer >= {minimum}")
    return value


def read_and_validate_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Step 3.2 manifest not found: {path}")
    info = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(info, dict) or info.get("status") != "PREPROCESSING FINISHED":
        raise ValueError("ATSC Step 3.2 preprocessing must be finished")
    if info.get("source_dataset") != DATASET_NAME or info.get("tokenizer_name") != MODEL_NAME:
        raise ValueError("Dataset/tokenizer mismatch with audited ATSC source")
    if info.get("gold_pairs_only") is not True or info.get("ate_predictions_used") is not False:
        raise ValueError("Only GOLD ATSC pairs may be used for supervised training")
    if info.get("held_out_test_kept_separate") is not True:
        raise ValueError("Held-out test must remain separate")
    if info.get("training_performed") is not False or info.get("model_weights_loaded") is not False:
        raise ValueError("Input must be preprocessing-only artifacts")
    if info.get("review_overlap_between_splits") != 0:
        raise ValueError("Manifest reports review-level data leakage")
    if info.get("vocabulary_resized") is not False:
        raise ValueError("Preprocessing must not resize DistilBERT vocabulary")
    if info.get("truncation") is not False:
        raise ValueError("Preprocessing must not silently truncate samples")
    if info.get("target_disambiguation") != "span_markers_from_gold_start_end_offsets":
        raise ValueError("Missing aspect occurrence disambiguation")
    if info.get("target_markers") != TARGET_MARKERS:
        raise ValueError("Mismatch with registered target marker token IDs")
    if info.get("label_mapping") != {str(k): v for k, v in ID2LABEL.items()}:
        raise ValueError("ATSC polarity mapping must be 0 negative, 1 neutral, 2 positive")
    if info.get("input_format") != "[CLS] review with [unused1] target [unused2] [SEP] aspect [SEP]":
        raise ValueError("Unexpected ATSC sequence-pair encoding convention")
    max_length = _integer(info.get("max_length"), "max_length", minimum=8)
    if max_length > 512:
        raise ValueError("DistilBERT max_length exceeds 512")
    counts = info.get("pair_counts")
    reviews = info.get("review_counts")
    if not isinstance(counts, dict) or not isinstance(reviews, dict):
        raise ValueError("Missing pair_counts/review_counts")
    for split in ALL_SPLITS:
        if _integer(counts.get(split), f"pair_counts/{split}", minimum=1) < 1:
            raise ValueError("Empty split")
        _integer(reviews.get(split), f"review_counts/{split}", minimum=1)
        if reviews[split] > counts[split]:
            raise ValueError(f"More reviews than labeled pairs in {split}")
    imb = info.get("imbalance", {})
    class_counts = imb.get("class_counts") if isinstance(imb, dict) else None
    if not isinstance(class_counts, dict):
        raise ValueError("Missing class imbalance report")
    for split in ALL_SPLITS:
        if not isinstance(class_counts.get(split), dict) or set(class_counts[split]) != set(ID2LABEL.values()):
            raise ValueError(f"Missing/invalid class counts for {split}")
        n = sum(_integer(class_counts[split][c], f"{split}/{c}") for c in ID2LABEL.values())
        if n != counts[split]:
            raise ValueError(f"Class counts do not match {split} pair count")
    if any(class_counts["train"][name] == 0 for name in ID2LABEL.values()):
        raise ValueError("All 3 sentiment classes must be present in train")
    for split in SPLITS_FOR_TRAINING:
        lengths = info.get("encoded_token_lengths", {}).get(split, {})
        if lengths.get("over_max_length") != 0:
            raise ValueError(f"{split} includes truncated/overlong samples")
    return info


def output_can_be_created(output: Path) -> None:
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"Output is not empty: {output}. Use a fresh --output directory")


def load_two_splits(data_dir: Path, loader: Any) -> tuple[Any, Any]:
    """Open ONLY the two split subdirectories, never DatasetDict/test."""
    root = data_dir / "tokenized_splits"
    locations = [root / name for name in SPLITS_FOR_TRAINING]
    for path in locations:
        if not path.is_dir():
            raise FileNotFoundError(f"Training/validation split missing: {path}")
    return loader(str(locations[0])), loader(str(locations[1]))


def validate_encoded_rows(train_ds: Any, val_ds: Any, info: Mapping[str, Any], *, vocab_size: int) -> dict[str, Any]:
    """Validate label counts, model inputs and exact encoded train/val overlap.

    Review-level non-overlap was verified during Steps 3.1/3.2 and recorded
    in the manifest; encoded overlap adds a separate runtime safeguard.
    """
    _integer(vocab_size, "vocab_size", minimum=1)
    all_ids: set[tuple[int, ...]] = set()
    dist: dict[str, dict[str, int]] = {}
    shape_stats: dict[str, dict[str, Any]] = {}
    for name, ds in zip(SPLITS_FOR_TRAINING, (train_ds, val_ds)):
        if len(ds) != info["pair_counts"][name]:
            raise ValueError(f"{name}: dataset count != manifest pair count")
        counts: Counter[int] = Counter()
        lengths: list[int] = []
        own_ids: set[tuple[int, ...]] = set()
        for i, row in enumerate(ds):
            if not isinstance(row, Mapping):
                raise ValueError(f"{name}[{i}]: row must be mapping")
            if set(row) != {"input_ids", "attention_mask", "labels"}:
                raise ValueError(f"{name}[{i}]: unexpected fields (only model inputs allowed)")
            ids, mask, y = row["input_ids"], row["attention_mask"], row["labels"]
            if not isinstance(ids, (list, tuple)) or not isinstance(mask, (list, tuple)) or len(ids) != len(mask):
                raise ValueError(f"{name}[{i}]: invalid input_ids/attention_mask lengths")
            if not 8 <= len(ids) <= info["max_length"]:
                raise ValueError(f"{name}[{i}]: sequence length out of bounds")
            if any(type(x) is not int or not 0 <= x < vocab_size for x in ids):
                raise ValueError(f"{name}[{i}]: invalid token ID")
            if any(type(x) is not int or x != 1 for x in mask):
                raise ValueError(f"{name}[{i}]: unpadded input must have all-one attention_mask")
            if type(y) is not int or y not in ID2LABEL:
                raise ValueError(f"{name}[{i}]: invalid sentiment label (must be 0/1/2)")
            a, b = TARGET_MARKERS.values()
            if ids.count(a) != 1 or ids.count(b) != 1 or ids.index(a) >= ids.index(b):
                raise ValueError(f"{name}[{i}]: missing/ambiguous target marker occurrence")
            key = tuple(ids)
            if key in own_ids:
                raise ValueError(f"{name}[{i}]: duplicated encoded input")
            own_ids.add(key)
            counts[y] += 1
            lengths.append(len(ids))
        if name == "validation" and own_ids & all_ids:
            raise ValueError("LEAKAGE: encoded input appears in both train and validation")
        all_ids.update(own_ids)
        dist[name] = {ID2LABEL[j]: counts[j] for j in ID2LABEL}
        if dist[name] != info["imbalance"]["class_counts"][name]:
            raise ValueError(f"{name}: sentiment distribution != Step 3.2 manifest")
        shape_stats[name] = {"samples":len(ds), "min_tokens":min(lengths),
                             "max_tokens":max(lengths), "mean_tokens":round(sum(lengths)/len(lengths), 4)}
    return {
        "pair_counts":{name:info["pair_counts"][name] for name in SPLITS_FOR_TRAINING},
        "class_distribution":dist,
        "sequence_stats":shape_stats,
        "encoded_train_validation_overlap":0,
        "review_overlap_reported_by_preprocessing": info["review_overlap_between_splits"],
    }


def check_tokenizer(tokenizer: Any, info: Mapping[str, Any]) -> None:
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("A saved FAST DistilBERT tokenizer is required")
    if getattr(tokenizer, "pad_token_id", None) is None:
        raise ValueError("Tokenizer is missing pad token")
    if not 8 <= info["max_length"] <= getattr(tokenizer, "model_max_length", 0):
        raise ValueError("Tokenizer model capacity differs from preprocessing")
    if len(tokenizer) != 30522:
        raise ValueError("Tokenizer vocabulary resized: incompatible with base DistilBERT")
    for marker, idx in TARGET_MARKERS.items():
        if tokenizer.convert_tokens_to_ids(marker) != idx:
            raise ValueError(f"Tokenizer missing target marker ID {marker}={idx}")
    special = getattr(tokenizer, "additional_special_tokens", ())
    if not all(marker in special for marker in TARGET_MARKERS):
        raise ValueError("Marker tokens must be registered as additional_special_tokens")


def classification_metrics(gold: Sequence[int], predictions: Sequence[int]) -> dict[str, float | int]:
    """Multiclass metrics including all three fixed classes, no model or sklearn needed."""
    if len(gold) != len(predictions) or not gold:
        raise ValueError("Metric input lengths must match and be nonzero")
    confusion = [[0] * 3 for _ in range(3)]
    for y, p in zip(gold, predictions):
        if type(y) is not int or y not in ID2LABEL or type(p) is not int or p not in ID2LABEL:
            raise ValueError("Metric labels must be 0/1/2")
        confusion[y][p] += 1
    scores: dict[str, float | int] = {}
    per_f1: list[float] = []
    supports: list[int] = []
    tp_sum = 0
    for i, name in ID2LABEL.items():
        tp = confusion[i][i]
        support = sum(confusion[i])
        predicted_count = sum(confusion[j][i] for j in ID2LABEL)
        fp, fn = predicted_count - tp, support - tp
        precision = tp/(tp+fp) if tp+fp else 0.0
        recall = tp/(tp+fn) if tp+fn else 0.0
        f1 = 2*precision*recall/(precision+recall) if precision+recall else 0.0
        scores[f"{name}_precision"] = precision
        scores[f"{name}_recall"] = recall
        scores[f"{name}_f1"] = f1
        scores[f"{name}_support"] = support
        per_f1.append(f1)
        supports.append(support)
        tp_sum += tp
    total = sum(supports)
    scores["macro_f1"] = sum(per_f1)/3
    scores["weighted_f1"] = sum(f*s for f, s in zip(per_f1, supports))/total
    scores["accuracy"] = tp_sum/total
    return scores


def make_compute_metrics():
    """Hugging Face Trainer evaluation callback (validation ONLY)."""
    def compute_metrics(eval_prediction: Any) -> dict[str, float | int]:
        logits, labels = eval_prediction
        if isinstance(logits, tuple):
            logits = logits[0]
        import numpy as np
        x = np.asarray(logits)
        y = np.asarray(labels)
        if x.ndim != 2 or x.shape[1] != 3 or y.ndim != 1 or x.shape[0] != y.shape[0]:
            raise ValueError("Expected logits [N,3] and labels [N]")
        if not np.all(np.isfinite(x)):
            raise ValueError("Non-finite model logits")
        return classification_metrics(y.astype(int).tolist(), x.argmax(axis=1).astype(int).tolist())
    return compute_metrics


def training_plan(*, output: str, epochs: int, batch_size: int, eval_batch_size: int,
                  lr: float, weight_decay: float, patience: int, seed: int,
                  fp16: bool, device: str) -> dict[str, Any]:
    for key, value in (("epochs", epochs), ("batch_size",batch_size),
                       ("eval_batch_size",eval_batch_size), ("patience",patience)):
        _integer(value, key, minimum=1)
    _integer(seed, "seed", minimum=0)
    if not isinstance(lr, (float, int)) or not math.isfinite(lr) or not (0 < lr < 1):
        raise ValueError("learning rate must be within (0,1)")
    if not isinstance(weight_decay, (float, int)) or not math.isfinite(weight_decay) or not (0 <= weight_decay < 1):
        raise ValueError("weight_decay must be within [0,1)")
    if device not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be auto, cpu, or cuda")
    if fp16 and device == "cpu":
        raise ValueError("--fp16 cannot be used on CPU")
    return {
        "output_dir": str(output),
        "eval_strategy":"epoch", "save_strategy":"epoch", "logging_strategy":"epoch",
        "learning_rate": float(lr), "per_device_train_batch_size":batch_size,
        "per_device_eval_batch_size":eval_batch_size,
        "num_train_epochs":epochs, "weight_decay":float(weight_decay),
        "load_best_model_at_end":True,
        "metric_for_best_model":"eval_macro_f1", "greater_is_better":True,
        "save_total_limit":2, "seed":seed, "data_seed":seed,
        "fp16":fp16, "use_cpu":device == "cpu", "report_to":"none",
        "dataloader_num_workers":0, "save_safetensors":True,
        "optim":"adamw_torch",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ATSC Step 3.3: train DistilBERT with train/validation only")
    parser.add_argument("--data", default="artifacts/atsc_preprocessing")
    parser.add_argument("--output", default="artifacts/atsc_training")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print plan; no model weights/training")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision ONLY on a compatible CUDA GPU")
    return parser


def _version_info() -> dict[str, str]:
    versions = {}
    for pkg in ("torch", "transformers", "datasets", "accelerate", "numpy"):
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = "not installed"
    return versions


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    data_dir = Path(args.data)
    output = Path(args.output)
    info = read_and_validate_manifest(data_dir / "manifest.json")
    plan = training_plan(
        output=str(output / "checkpoints"), epochs=args.epochs,
        patience=args.patience, batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size, lr=args.learning_rate,
        weight_decay=args.weight_decay, seed=args.seed, fp16=args.fp16,
        device=args.device,
    )
    output_can_be_created(output)

    # Deliberately load train & validation directories individually; never open test.
    from datasets import load_from_disk
    from transformers import AutoTokenizer
    print("Loading train and validation ONLY (held-out test is NOT opened).", flush=True)
    train, validation = load_two_splits(data_dir, load_from_disk)
    tok_path = data_dir / "tokenizer"
    if not tok_path.is_dir():
        raise FileNotFoundError(f"Saved tokenizer missing: {tok_path}")
    tokenizer = AutoTokenizer.from_pretrained(str(tok_path), local_files_only=True, use_fast=True)
    check_tokenizer(tokenizer, info)
    checks = validate_encoded_rows(train, validation, info, vocab_size=len(tokenizer))
    preflight = {
        "status":"PRE-FLIGHT PASSED",
        "source_dataset":info["source_dataset"], "model":MODEL_NAME,
        "split_counts":checks["pair_counts"],
        "validation_checks":checks,
        "training_plan":plan, "early_stopping_patience":args.patience,
        "objective":"unweighted_cross_entropy_baseline",
        "best_checkpoint_metric":"eval_macro_f1",
        "held_out_test_used":False, "training_performed":False,
    }
    print(json.dumps(preflight, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
    if args.dry_run:
        print("DRY RUN FINISHED (NO model weights loaded, NO training, NO test data opened).")
        return 0

    import torch
    from transformers import (AutoModelForSequenceClassification, DataCollatorWithPadding,
                              EarlyStoppingCallback, Trainer, TrainingArguments, set_seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu or auto")
    if args.fp16 and not torch.cuda.is_available():
        raise RuntimeError("--fp16 requires a CUDA GPU")
    print("Compute device:", "CUDA" if torch.cuda.is_available() and args.device != "cpu" else "CPU", flush=True)
    set_seed(args.seed)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=3, id2label=ID2LABEL, label2id=LABEL2ID,
    )
    # Marker IDs are already in base DistilBERT vocab. Do not resize embeddings.
    if model.get_input_embeddings().num_embeddings != len(tokenizer):
        raise RuntimeError("Tokenizer/model vocabulary mismatch; refusing to resize embeddings")
    trainer = Trainer(
        model=model,
        args=TrainingArguments(**plan),
        train_dataset=train,
        eval_dataset=validation,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer, padding=True),
        processing_class=tokenizer,
        compute_metrics=make_compute_metrics(),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )
    print("Fine-tuning ATSC on train, evaluating ONLY validation...", flush=True)
    result = trainer.train()
    best_ckpt = trainer.state.best_model_checkpoint
    best_score = trainer.state.best_metric
    if not best_ckpt or not isinstance(best_score, (float, int)) or not math.isfinite(best_score):
        raise RuntimeError("No valid best checkpoint or validation Macro-F1; model not published")
    if not (0 <= best_score <= 1):
        raise RuntimeError("Best validation Macro-F1 is outside [0,1]")
    # With load_best_model_at_end=True, trainer.model holds best checkpoint weights.
    final_val = trainer.evaluate(eval_dataset=validation, metric_key_prefix="eval")
    if "eval_macro_f1" not in final_val:
        raise RuntimeError("Validation Macro-F1 missing after loading best model")
    best_dir = output / "best_model"
    best_dir.mkdir(parents=True, exist_ok=False)
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))
    history_file = output / "training_history.json"
    history_file.write_text(json.dumps(trainer.state.log_history, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    summary = {
        "step":"3.3", "status":"TRAINING FINISHED", "training_performed":True,
        "dataset":info["source_dataset"], "model":MODEL_NAME,
        "input_format":info["input_format"], "target_markers":info["target_markers"],
        "class_mapping":ID2LABEL,
        "train_examples":len(train), "validation_examples":len(validation),
        "test_used":False, "selection_metric":"eval_macro_f1",
        "loss":"unweighted_cross_entropy",
        "best_checkpoint":best_ckpt, "best_validation_macro_f1":float(best_score),
        "best_model_validation_metrics":final_val,
        "train_result_metrics":result.metrics,
        "early_stopping_patience":args.patience,
        "training_config":plan, "validation_checks":checks,
        "best_model_dir":str(best_dir), "versions":_version_info(),
    }
    (output / "run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")
    (output / "validation_metrics.json").write_text(
        json.dumps(final_val, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({
        "status":"TRAINING FINISHED", "best_checkpoint":best_ckpt,
        "best_validation_macro_f1":float(best_score),
        "model_dir":str(best_dir),
        "summary_path":str(output / "run_summary.json"),
        "test_used":False,
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
