"""Step 3.4: final held-out evaluation of the FIXED best ATSC checkpoint.

Reads only the already-tokenized test split and Step 3.1 gold test pairs.
Never trains, never selects a new checkpoint, never writes to training/data dirs.
The training summary is authoritative for which checkpoint was chosen by val F1.
"""
from __future__ import annotations

import os
# Must be set before importing Transformers in Anaconda/Keras-3 environments.
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}
MARKERS = {"[unused1]": 2, "[unused2]": 3}
EXPECTED_MODEL = "distilbert/distilbert-base-uncased"
MODEL_INPUT_KEYS = {"input_ids", "attention_mask", "labels"}


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing JSON file: {path}")
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def validate_preprocessing_manifest(info: Mapping[str, Any]) -> int:
    if info.get("status") != "PREPROCESSING FINISHED":
        raise ValueError("Step 3.2 has not finished")
    if info.get("source_dataset") != "thainq107/abte-restaurants":
        raise ValueError("Unexpected ATSC dataset")
    if info.get("tokenizer_name") != EXPECTED_MODEL:
        raise ValueError("Unexpected tokenizer base")
    if info.get("gold_pairs_only") is not True or info.get("ate_predictions_used") is not False:
        raise ValueError("Expected only gold pairs, no ATE model predictions")
    if info.get("held_out_test_kept_separate") is not True or info.get("review_overlap_between_splits") != 0:
        raise ValueError("Test split leakage or incorrect split policy")
    if info.get("truncation") is not False or info.get("vocabulary_resized") is not False:
        raise ValueError("Unexpected truncation or vocabulary resize")
    if info.get("target_markers") != MARKERS:
        raise ValueError("Target markers do not match Step 3.2")
    if info.get("target_disambiguation") != "span_markers_from_gold_start_end_offsets":
        raise ValueError("Aspect occurrence marking absent")
    if info.get("label_mapping") != {str(k): v for k, v in ID2LABEL.items()}:
        raise ValueError("Unexpected label mapping")
    if info.get("training_performed") is not False or info.get("model_weights_loaded") is not False:
        raise ValueError("Input data artifacts were not preprocessing-only")
    counts = info.get("pair_counts", {})
    if not isinstance(counts, dict) or type(counts.get("test")) is not int or counts["test"] <= 0:
        raise ValueError("Invalid test pair count")
    if info.get("encoded_token_lengths", {}).get("test", {}).get("over_max_length") != 0:
        raise ValueError("Overlong/truncated test data")
    max_length = info.get("max_length")
    if type(max_length) is not int or not 8 <= max_length <= 512:
        raise ValueError("Invalid max_length")
    dist = info.get("imbalance", {}).get("class_counts", {}).get("test", {})
    if set(dist) != set(ID2LABEL.values()) or any(type(v) is not int or v < 0 for v in dist.values()):
        raise ValueError("Invalid test class distribution")
    if sum(dist.values()) != counts["test"]:
        raise ValueError("Test class counts do not match total")
    return counts["test"]


def validate_training_summary(summary: Mapping[str, Any], info: Mapping[str, Any]) -> str:
    if summary.get("step") != "3.3" or summary.get("status") != "TRAINING FINISHED":
        raise ValueError("ATSC training did not finish")
    if summary.get("training_performed") is not True or summary.get("test_used") is not False:
        raise ValueError("Training summary indicates test was used or no training occurred")
    if summary.get("model") != EXPECTED_MODEL or summary.get("selection_metric") != "eval_macro_f1":
        raise ValueError("Model or checkpoint selection policy mismatch")
    if summary.get("class_mapping") != {str(k): v for k, v in ID2LABEL.items()}:
        raise ValueError("Training label mapping mismatch")
    if summary.get("target_markers") != info["target_markers"]:
        raise ValueError("Training aspect markers do not match preprocessing")
    if summary.get("best_model_dir", "").replace("\\", "/").rstrip("/").split("/")[-1] != "best_model":
        raise ValueError("Training summary does not name best_model")
    ckpt = str(summary.get("best_checkpoint", "")).replace("\\", "/")
    parts = ckpt.rstrip("/").split("/")
    if len(parts) < 2 or parts[-2] != "checkpoints" or not re.fullmatch(r"checkpoint-\d+", parts[-1]):
        raise ValueError("Missing best checkpoint selected by validation")
    f1 = summary.get("best_validation_macro_f1")
    if not isinstance(f1, (float, int)) or isinstance(f1, bool) or not math.isfinite(f1) or not (0 <= f1 <= 1):
        raise ValueError("Invalid best validation Macro-F1")
    if summary.get("train_examples") != info["pair_counts"]["train"]:
        raise ValueError("Train count mismatch")
    if summary.get("validation_examples") != info["pair_counts"]["validation"]:
        raise ValueError("Validation count mismatch")
    return parts[-1]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_unchanged_best_weights(best_dir: Path, ckpt_dir: Path) -> dict[str, Any]:
    """Read-only check; fail instead of selecting/replacing a checkpoint."""
    from safetensors import safe_open
    best = best_dir / "model.safetensors"
    source = ckpt_dir / "model.safetensors"
    for file in (best, source):
        if not file.is_file():
            raise FileNotFoundError(f"Missing weights: {file}")
        try:
            with safe_open(str(file), framework="np") as tensors:
                keys = list(tensors.keys())
                if "classifier.weight" not in keys or "classifier.bias" not in keys:
                    raise ValueError(f"Classification head missing from {file}")
                if tensors.get_slice("classifier.weight").get_shape()[0] != 3:
                    raise ValueError(f"Expected 3 sentiment logits in {file}")
        except Exception as exc:
            raise ValueError(
                f"Cannot parse {file}: {exc}. DO NOT RETRAIN OR EVALUATE. "
                "If checkpoint weights are valid, verify loading the selected checkpoint, "
                "then restore best_model from that SAME checkpoint (keep backup)."
            ) from exc
    if best.stat().st_size != source.stat().st_size:
        raise ValueError("best_model and selected checkpoint sizes DIFFER; restore best_model from selected checkpoint first")
    left, right = sha256_file(best), sha256_file(source)
    if left != right:
        raise ValueError("best_model and selected checkpoint SHA-256 DIFFER; refusing to evaluate the wrong weights")
    return {"weight_sha256": left, "selected_checkpoint":ckpt_dir.name,
            "model_bytes":best.stat().st_size, "checkpoint_identical":True}


def validate_tokenizer(tokenizer: Any) -> None:
    if not getattr(tokenizer, "is_fast", False) or len(tokenizer) != 30522:
        raise ValueError("Saved tokenizer is not the 30522-token DistilBERT FAST tokenizer")
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer missing pad token")
    if any(tokenizer.convert_tokens_to_ids(token) != index for token, index in MARKERS.items()):
        raise ValueError("Target markers have changed IDs")
    if not all(marker in getattr(tokenizer, "additional_special_tokens", []) for marker in MARKERS):
        raise ValueError("Target markers are not registered as special tokens")


def validate_test_rows(dataset: Any, info: Mapping[str, Any], vocab_size: int) -> dict[str, Any]:
    n = validate_preprocessing_manifest(info)
    if len(dataset) != n:
        raise ValueError(f"Test split has {len(dataset)} rows, expected {n}")
    counts: Counter[int] = Counter()
    seen: set[tuple[int, ...]] = set()
    lengths = []
    for i, row in enumerate(dataset):
        if not isinstance(row, Mapping) or set(row) != MODEL_INPUT_KEYS:
            raise ValueError(f"test[{i}]: unexpected input columns")
        ids, mask, label = row["input_ids"], row["attention_mask"], row["labels"]
        if type(label) is not int or label not in ID2LABEL:
            raise ValueError(f"test[{i}]: invalid gold sentiment")
        if (not isinstance(ids, (list, tuple)) or not isinstance(mask, (list, tuple))
                or not 8 <= len(ids) <= info["max_length"] or len(mask) != len(ids)):
            raise ValueError(f"test[{i}]: invalid sequence shape")
        if any(type(token) is not int or not 0 <= token < vocab_size for token in ids):
            raise ValueError(f"test[{i}]: token outside model vocab")
        if any(type(value) is not int or value != 1 for value in mask):
            raise ValueError(f"test[{i}]: expected no pre-padding")
        if ids.count(2) != 1 or ids.count(3) != 1 or ids.index(2) >= ids.index(3):
            raise ValueError(f"test[{i}]: invalid occurrence marker positions")
        key = tuple(ids)
        if key in seen:
            raise ValueError(f"test[{i}]: duplicated encoded input")
        seen.add(key)
        counts[label] += 1
        lengths.append(len(ids))
    expected = info["imbalance"]["class_counts"]["test"]
    actual = {ID2LABEL[i]: counts[i] for i in ID2LABEL}
    if actual != expected:
        raise ValueError(f"Test gold distribution mismatch: {actual} != {expected}")
    return {"pairs":n, "class_counts":actual, "min_tokens":min(lengths),
            "max_tokens":max(lengths), "mean_tokens":round(sum(lengths) / n, 4)}


def read_gold_test_pairs(path: Path, n: int) -> list[dict[str, Any]]:
    """Read only test.jsonl, not train/validation raw pairs."""
    from atsc_preprocessing import validate_pair
    if not path.is_file():
        raise FileNotFoundError(f"Missing test gold pairs for error analysis: {path}")
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            raise ValueError(f"Blank line in {path}:{line_no}")
        rows.append(validate_pair(json.loads(line), split="test", index=line_no))
    if len(rows) != n:
        raise ValueError(f"Gold pair count {len(rows)} != tokenized count {n}")
    return rows


def verify_gold_alignment(rows: Sequence[Mapping[str, Any]], dataset: Any, tokenizer: Any) -> None:
    """Important: error examples follow the exact tokenized TEST order, no guessing."""
    from atsc_preprocessing import mark_target_span
    if len(rows) != len(dataset):
        raise ValueError("Gold/encoded test count mismatch")
    for i, (gold, record) in enumerate(zip(rows, dataset)):
        if record["labels"] != gold["sentiment_id"]:
            raise ValueError(f"test[{i}]: gold label and encoded label mismatch")
        marked = mark_target_span(gold["review"], gold["start"], gold["end"])
        encoded = tokenizer(marked, gold["aspect"], add_special_tokens=True,
                            padding=False, truncation=False, return_attention_mask=True)
        if encoded["input_ids"] != record["input_ids"] or encoded["attention_mask"] != record["attention_mask"]:
            raise ValueError(f"test[{i}]: GOLD PAIR IS MISALIGNED WITH SAVED TOKENIZED TEST — abort")


def confusion_and_metrics(gold: Sequence[int], pred: Sequence[int]) -> dict[str, Any]:
    if not gold or len(gold) != len(pred):
        raise ValueError("Non-empty gold and predictions must have the same length")
    mat = [[0 for _ in range(3)] for _ in range(3)]
    for y, p in zip(gold, pred):
        if type(y) is not int or y not in ID2LABEL or type(p) is not int or p not in ID2LABEL:
            raise ValueError("Metrics require integer labels 0/1/2")
        mat[y][p] += 1
    result: dict[str, Any] = {"confusion_matrix":mat, "confusion_matrix_rows":"gold", "confusion_matrix_columns":"predicted"}
    f1s, weighted_f1 = [], 0.0
    for i, label in ID2LABEL.items():
        tp = mat[i][i]
        support = sum(mat[i])
        predicted = sum(mat[j][i] for j in ID2LABEL)
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        result.setdefault("per_class", {})[label] = {"precision":precision, "recall":recall,
                                                  "f1":f1, "support":support, "predicted":predicted}
        f1s.append(f1)
        weighted_f1 += f1 * support
    n = len(gold)
    result["total"] = n
    result["correct"] = sum(mat[i][i] for i in ID2LABEL)
    result["errors"] = n - result["correct"]
    result["accuracy"] = result["correct"] / n
    result["macro_f1"] = sum(f1s) / 3
    result["weighted_f1"] = weighted_f1 / n
    return result


def validate_probability_rows(probabilities: Sequence[Sequence[float]], n: int) -> list[int]:
    if len(probabilities) != n:
        raise ValueError("Wrong number of probability rows")
    preds = []
    for i, row in enumerate(probabilities):
        if len(row) != 3 or any(not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0 or x > 1 for x in row):
            raise ValueError(f"Invalid probability vector at row {i}")
        if not math.isclose(sum(row), 1.0, abs_tol=1e-4):
            raise ValueError(f"Probabilities at row {i} do not sum to 1")
        preds.append(max(range(3), key=lambda j: row[j]))
    return preds


def inference_probabilities(model: Any, tokenizer: Any, dataset: Any, batch_size: int, device: str) -> list[list[float]]:
    """Inference-only; torch.no_grad disables training and gradients."""
    import torch
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be positive")
    if device not in ("cpu", "cuda"):
        raise ValueError("Expected resolved CPU/CUDA device")
    model.to(device)
    model.eval()
    if model.training:
        raise RuntimeError("Model remained in training mode")
    all_probs = []
    with torch.inference_mode():
        for start in range(0, len(dataset), batch_size):
            rows = [dataset[i] for i in range(start, min(start + batch_size, len(dataset)))]
            packed = tokenizer.pad({"input_ids":[r["input_ids"] for r in rows],
                                    "attention_mask":[r["attention_mask"] for r in rows]},
                                   padding=True, return_tensors="pt")
            logits = model(input_ids=packed["input_ids"].to(device),
                           attention_mask=packed["attention_mask"].to(device)).logits
            if logits.ndim != 2 or logits.shape != (len(rows), 3) or not torch.isfinite(logits).all().item():
                raise ValueError("Unexpected/non-finite ATSC model logits")
            probs = torch.softmax(logits.float(), dim=-1).cpu().tolist()
            all_probs.extend(probs)
    validate_probability_rows(all_probs, len(dataset))
    return all_probs


def create_error_analysis(rows: Sequence[Mapping[str, Any]], probabilities: Sequence[Sequence[float]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    preds = validate_probability_rows(probabilities, len(rows))
    mistakes: list[dict[str, Any]] = []
    transitions: Counter[str] = Counter()
    for i, (gold, p, values) in enumerate(zip(rows, preds, probabilities)):
        y = gold["sentiment_id"]
        if y == p:
            continue
        key = f"{ID2LABEL[y]} -> {ID2LABEL[p]}"
        transitions[key] += 1
        mistakes.append({"test_index":i, "review":gold["review"], "aspect":gold["aspect"],
                         "start":gold["start"], "end":gold["end"],
                         "gold_label":ID2LABEL[y], "predicted_label":ID2LABEL[p],
                         "predicted_confidence_UNCALIBRATED":values[p],
                         "probabilities_UNCALIBRATED":{ID2LABEL[j]:values[j] for j in ID2LABEL}})
    return mistakes, {"n_errors":len(mistakes), "by_gold_to_predicted":dict(sorted(transitions.items())),
                      "note":"Softmax probabilities are NOT calibrated confidence estimates."}


def output_guard(output: Path) -> None:
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"Output already exists and is nonempty: {output}. Refusing to overwrite final test report")


def write_reports(output: Path, metrics: Mapping[str, Any], mistakes: Sequence[Mapping[str, Any]], error_summary: Mapping[str, Any]) -> None:
    """Only called after model inference and ALL validations succeed."""
    output_guard(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "test_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    (output / "error_analysis.json").write_text(json.dumps(error_summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    with (output / "error_examples.jsonl").open("w", encoding="utf-8") as f:
        for error in mistakes:
            f.write(json.dumps(error, ensure_ascii=False, allow_nan=False) + "\n")
    with (output / "confusion_matrix.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["gold\\predicted", *ID2LABEL.values()])
        for label, row in zip(ID2LABEL.values(), metrics["confusion_matrix"]):
            writer.writerow([label, *row])


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Step 3.4: final ATSC test evaluation (fixed checkpoint, no training)")
    p.add_argument("--data", default="artifacts/atsc_preprocessing")
    p.add_argument("--audit", default="artifacts/atsc_audit")
    p.add_argument("--training", default="artifacts/atsc_training")
    p.add_argument("--output", default="artifacts/atsc_evaluation")
    p.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--dry-run", action="store_true", help="Checks artifacts and alignment; no inference, no output")
    return p


def run(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    data, audit, training, output = map(Path, (args.data, args.audit, args.training, args.output))
    output_guard(output)
    pre = read_json(data / "manifest.json")
    count = validate_preprocessing_manifest(pre)
    summary = read_json(training / "run_summary.json")
    ckpt_name = validate_training_summary(summary, pre)
    best_dir = training / "best_model"
    checkpoint = training / "checkpoints" / ckpt_name
    # Strict read-only integrity check before any test prediction.
    identity = verify_unchanged_best_weights(best_dir, checkpoint)

    from datasets import load_from_disk
    from transformers import AutoConfig, AutoTokenizer
    test_path = data / "tokenized_splits" / "test"
    if not test_path.is_dir():
        raise FileNotFoundError(f"Missing held-out test dataset: {test_path}")
    print("Loading ONLY the held-out test split (NO train/validation, NO training)...", flush=True)
    test = load_from_disk(str(test_path))
    tok = AutoTokenizer.from_pretrained(str(best_dir), local_files_only=True, use_fast=True)
    validate_tokenizer(tok)
    config = AutoConfig.from_pretrained(str(best_dir), local_files_only=True)
    if (config.model_type != "distilbert" or config.num_labels != 3 or
            {int(k):v for k,v in config.id2label.items()} != ID2LABEL):
        raise ValueError("Best model config must be DistilBERT with correct 3 label names")
    if len(tok) != config.vocab_size:
        raise ValueError("Model/tokenizer vocab mismatch")
    checks = validate_test_rows(test, pre, vocab_size=len(tok))
    if checks["pairs"] != count:
        raise AssertionError("Unexpected test count")
    audit_report = read_json(audit / "audit_report.json")
    if audit_report.get("status") != "AUDIT PASSED" or audit_report.get("pair_counts", {}).get("test") != count:
        raise ValueError("Audit/test counts disagree")
    pairs = read_gold_test_pairs(audit / "pairs" / "test.jsonl", count)
    verify_gold_alignment(pairs, test, tok)
    preflight = {"status":"PRE-FLIGHT PASSED", "evaluation_split":"test", "test_pairs":count,
                 "class_counts":checks["class_counts"], "token_lengths":checks,
                 "checkpoint_integrity":identity, "selected_by":"validation_macro_f1",
                 "validation_macro_f1":summary["best_validation_macro_f1"],
                 "gold_alignment":"ALL TEST ROWS EXACTLY MATCH", "training_performed":False,
                 "model_predictions_computed":False}
    print(json.dumps(preflight, indent=2, ensure_ascii=False, allow_nan=False), flush=True)
    if args.dry_run:
        print("DRY RUN FINISHED: no inference, no training, no output files.")
        return 0

    import torch
    from transformers import AutoModelForSequenceClassification
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; pass --device cpu")
    device = "cuda" if (args.device in ("cuda", "auto") and torch.cuda.is_available()) else "cpu"
    print(f"Loading FIXED best model on {device} and evaluating test ONCE...", flush=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(best_dir), local_files_only=True)
    if (model.config.model_type != "distilbert" or model.config.num_labels != 3 or
            model.get_input_embeddings().num_embeddings != len(tok)):
        raise ValueError("Loaded model does not match preprocessing")
    probs = inference_probabilities(model, tok, test, args.batch_size, device)
    pred = validate_probability_rows(probs, count)
    gold = [row["labels"] for row in test]
    result = confusion_and_metrics(gold, pred)
    mistakes, errors = create_error_analysis(pairs, probs)
    if result["errors"] != errors["n_errors"] or sum(map(sum, result["confusion_matrix"])) != count:
        raise AssertionError("Metrics and error examples disagree")
    metrics = {"step":"3.4", "status":"FINAL TEST EVALUATED", "dataset":pre["source_dataset"],
               "model":summary["model"], "split":"test", "test_count":count,
               "selected_checkpoint":ckpt_name, "best_validation_macro_f1_FOR_CONTEXT_ONLY":summary["best_validation_macro_f1"],
               "checkpoint_weight_sha256":identity["weight_sha256"],
               "checkpoint_selection_based_on_test":False,
               "training_performed":False, "model_weights_modified":False,
               "device":device, "batch_size":args.batch_size, **result}
    write_reports(output, metrics, mistakes, errors)
    print(json.dumps({"status":metrics["status"],"accuracy":metrics["accuracy"],
                      "macro_f1":metrics["macro_f1"], "errors":metrics["errors"],
                      "report_dir":str(output)}, indent=2), flush=True)
    return 0


def main() -> int:
    parser = build_parser()
    try:
        return run()
    except (ValueError, RuntimeError, OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        parser.exit(2, f"ATSC STEP 3.4 ERROR: {exc}\n")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
