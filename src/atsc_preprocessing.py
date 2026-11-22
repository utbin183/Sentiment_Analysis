"""Step 3.2 — ATSC preprocessing (gold review + target span -> DistilBERT pairs).

Pure validation/encoding helpers do not import Torch, Transformers, or Datasets.
The CLI imports tokenizer/dataset serialization utilities lazily; NEVER trains a model.

Input:  artifacts/atsc_audit/audit_report.json + pairs/{train,validation,test}.jsonl
Output: artifacts/atsc_preprocessing/tokenized_splits/ + tokenizer/ + manifest.json
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any, Mapping

SPLITS = ("train", "validation", "test")
POLARITY_ID_TO_NAME = {0: "negative", 1: "neutral", 2: "positive"}
DATASET_NAME = "thainq107/abte-restaurants"
TOKENIZER_NAME = "distilbert/distilbert-base-uncased"
TEXT_ORIGIN = "space_joined_source_tokens"
# These tokens are already present in the original BERT/DistilBERT vocabulary.
# Register them as *special* so the fast tokenizer keeps each one intact,
# without growing the embedding matrix. No weight initialization/training here.
TARGET_START_MARKER = "[unused1]"
TARGET_END_MARKER = "[unused2]"


def review_key_from_text(review: str) -> tuple[str, ...]:
    """Match ATE's case-insensitive, token-boundary-preserving review key.

    Step 3.1 review is exactly ' '.join(source Tokens). Do not use a hash or
    normalize text passed to the model; the key is ONLY for leakage checks.
    """
    return tuple(part.strip().casefold() for part in review.split(" "))


def _require_int(value: Any, name: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be an integer (bools not allowed)")
    return value


def read_audit_report(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"ATSC Step 3.1 audit report missing: {path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError("audit_report.json must be a JSON object")
    if report.get("status") != "AUDIT PASSED":
        raise ValueError("Step 3.1 audit status must be AUDIT PASSED")
    if report.get("dataset") != DATASET_NAME:
        raise ValueError("Dataset differs from ATE/ATSC source")
    expected = {str(k): v for k, v in POLARITY_ID_TO_NAME.items()}
    if report.get("polarity_mapping") != expected or report.get("unlabeled_polarity") != -1:
        raise ValueError("Step 3.1 polarity mapping mismatch")
    if report.get("review_overlap_between_splits") != 0:
        raise ValueError("Step 3.1 reports review leakage")
    if report.get("sentiment_train_validation_test_strategy") != "reuse_exact_ate_step_2_1_saved_review_splits":
        raise ValueError("Step 3.1 did not reuse persisted ATE splits")
    if report.get("ate_gold_not_predictions") is not True:
        raise ValueError("Step 3.1 inputs must be gold labels, not model predictions")
    if report.get("character_offsets") != "reconstructed_space_joined_tokens_not_original_raw_review_offsets":
        raise ValueError("Unexpected aspect offset convention")
    counts = report.get("pair_counts")
    distributions = report.get("class_distribution")
    uniques = report.get("unique_paired_reviews")
    if not all(isinstance(group, dict) and set(SPLITS).issubset(group) for group in (counts, distributions, uniques)):
        raise ValueError("Missing split counts, class distribution or unique review counts")
    for split in SPLITS:
        if _require_int(counts[split], f"pair_counts/{split}") < 1:
            raise ValueError(f"Empty or invalid ATSC split: {split}")
        if _require_int(uniques[split], f"unique_paired_reviews/{split}") < 1:
            raise ValueError(f"Invalid unique_paired_reviews/{split}")
        values = distributions[split]
        if not isinstance(values, dict) or set(values) != set(POLARITY_ID_TO_NAME.values()):
            raise ValueError(f"Invalid class_distribution/{split}")
        if any(_require_int(values[name], f"class_distribution/{split}/{name}") < 0 for name in values):
            raise ValueError(f"Negative class count in {split}")
        if sum(values.values()) != counts[split]:
            raise ValueError(f"Class distribution does not sum to pair count: {split}")
    return report


def validate_pair(row: Mapping[str, Any], *, split: str, index: int) -> dict[str, Any]:
    """Validate labels and ensure aspect character and word spans match the source.

    Character offsets are for the Step 3.1 reconstructed review, not raw text.
    """
    context = f"pairs/{split}[{index}]"
    if not isinstance(row, Mapping):
        raise ValueError(f"{context}: expected object")
    required = ("review", "aspect", "start", "end", "start_word", "end_word",
                "sentiment_id", "sentiment", "split", "text_origin")
    missing = [key for key in required if key not in row]
    if missing:
        raise ValueError(f"{context}: missing fields {missing}")
    review, aspect = row["review"], row["aspect"]
    if not isinstance(review, str) or not review.strip():
        raise ValueError(f"{context}: review must be non-empty text")
    if not isinstance(aspect, str) or not aspect.strip():
        raise ValueError(f"{context}: aspect must be non-empty text")
    if row["split"] != split:
        raise ValueError(f"{context}: row belongs to split {row['split']!r}, not {split}")
    if row["text_origin"] != TEXT_ORIGIN:
        raise ValueError(f"{context}: unsupported text_origin")
    sentiment_id = _require_int(row["sentiment_id"], f"{context}/sentiment_id")
    if sentiment_id not in POLARITY_ID_TO_NAME:
        raise ValueError(f"{context}: sentiment_id must be 0, 1 or 2; -1 is ignored")
    if row["sentiment"] != POLARITY_ID_TO_NAME[sentiment_id]:
        raise ValueError(f"{context}: sentiment text disagrees with sentiment_id")
    start = _require_int(row["start"], f"{context}/start")
    end = _require_int(row["end"], f"{context}/end")
    sw = _require_int(row["start_word"], f"{context}/start_word")
    ew = _require_int(row["end_word"], f"{context}/end_word")
    if not (0 <= start < end <= len(review)) or review[start:end] != aspect:
        raise ValueError(f"{context}: aspect character offsets do not match review")
    tokens = review.split(" ")
    if not (0 <= sw < ew <= len(tokens)) or any(not tok for tok in tokens):
        raise ValueError(f"{context}: malformed reconstructed review / word offsets")
    expected_start = len(" ".join(tokens[:sw])) + (1 if sw else 0)
    expected_end = expected_start + len(" ".join(tokens[sw:ew]))
    if (start, end) != (expected_start, expected_end):
        raise ValueError(f"{context}: character and word offsets disagree")
    return dict(row)


def load_pairs(pairs_dir: Path, report: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Read and validate all gold pairs, counts, class distributions and leakage."""
    splits: dict[str, list[dict[str, Any]]] = {}
    owner: dict[tuple[str,...], str] = {}
    for split in SPLITS:
        file = pairs_dir / f"{split}.jsonl"
        if not file.is_file():
            raise FileNotFoundError(f"Gold pairs missing: {file}")
        rows: list[dict[str, Any]] = []
        seen_spans: set[tuple[tuple[str,...], int, int]] = set()
        for line_number, line in enumerate(file.read_text(encoding="utf-8-sig").splitlines(), 1):
            if not line.strip():
                raise ValueError(f"{file}:{line_number}: blank line not allowed")
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{file}:{line_number}: invalid JSON") from exc
            row = validate_pair(raw, split=split, index=line_number)
            k = review_key_from_text(row["review"])
            previous = owner.setdefault(k, split)
            if previous != split:
                raise ValueError(f"Review leakage: same review in {previous} and {split}")
            span_key = (k, row["start_word"], row["end_word"])
            if span_key in seen_spans:
                raise ValueError(f"Duplicate gold span in {split}: {span_key}")
            seen_spans.add(span_key)
            # Keep distinct, labeled occurrences even when aspect text repeats.
            # mark_target_span() makes the exact occurrence part of model input.
            rows.append(row)
        if len(rows) != report["pair_counts"][split]:
            raise ValueError(f"{split} count mismatch: file={len(rows)}, audit={report['pair_counts'][split]}")
        counts = Counter(r["sentiment"] for r in rows)
        for cls in POLARITY_ID_TO_NAME.values():
            if counts[cls] != report["class_distribution"][split][cls]:
                raise ValueError(f"{split}: class_distribution mismatch for {cls}")
        n_reviews = len({review_key_from_text(row["review"]) for row in rows})
        if n_reviews != report["unique_paired_reviews"][split]:
            raise ValueError(f"{split}: unique review count does not match audit")
        splits[split] = rows
    return splits


def validate_ate_membership(
    pairs: Mapping[str, list[Mapping[str, Any]]], ate_raw: Mapping[str, Any],
) -> dict[str, int]:
    """Verify each Step 3.1 review maps to exactly the persisted Step 2.1 split.

    ATSC review counts may be smaller than ATE if a review has no labeled pair.
    """
    ate_owners: dict[tuple[str,...], str] = {}
    for split in SPLITS:
        if split not in ate_raw:
            raise ValueError(f"Missing ATE raw_splits/{split}")
        for index, record in enumerate(ate_raw[split]):
            tokens = record.get("Tokens")
            if not isinstance(tokens, list) or not tokens or any(not isinstance(t, str) for t in tokens):
                raise ValueError(f"Invalid ATE Tokens in {split}[{index}]")
            key = tuple(tok.strip().casefold() for tok in tokens)
            if key in ate_owners:
                raise ValueError("ATE saved splits contain duplicate reviews")
            ate_owners[key] = split
    counts = {}
    for split in SPLITS:
        keys = {review_key_from_text(r["review"]) for r in pairs[split]}
        for key in keys:
            if ate_owners.get(key) != split:
                raise ValueError(f"ATSC review does not belong to ATE saved split {split}")
        counts[split] = len(keys)
    return counts


def check_tokenizer(tokenizer: Any, *, max_length: int) -> None:
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("DistilBERT fast tokenizer required")
    capacity = getattr(tokenizer, "model_max_length", None)
    if type(max_length) is not int or max_length < 8 or not isinstance(capacity, int) or max_length > capacity:
        raise ValueError(f"max_length must be 8..{capacity}")
    if getattr(tokenizer, "pad_token_id", None) is None:
        raise ValueError("Tokenizer has no pad_token_id")
    if tokenizer.num_special_tokens_to_add(pair=True) != 3:
        raise ValueError("Expected BERT-style pair special tokens [CLS] review [SEP] aspect [SEP]")


def mark_target_span(review: str, start: int, end: int) -> str:
    """Mark *one* aspect occurrence, using half-open character offsets.

    This function must also be reused at future ATSC inference time with ATE's
    predicted start/end offsets. The gold review itself is never overwritten.
    """
    if not isinstance(review, str) or type(start) is not int or type(end) is not int:
        raise ValueError("mark_target_span expects review text and integer offsets")
    if not 0 <= start < end <= len(review) or not review[start:end].strip():
        raise ValueError("Invalid target span for aspect marker insertion")
    if any(marker.casefold() in review.casefold() for marker in
           (TARGET_START_MARKER, TARGET_END_MARKER)):
        raise ValueError("Review already contains a reserved target marker")
    # Change only whitespace immediately next to the inserted markers.
    left, target, right = review[:start].rstrip(), review[start:end], review[end:].lstrip()
    return " ".join(part for part in
                    (left, TARGET_START_MARKER, target, TARGET_END_MARKER, right) if part)


def register_target_markers(tokenizer: Any) -> dict[str, int]:
    """Register existing BERT [unused] IDs; refuse vocabulary resizing.

    This is important: the tokenizer must remain compatible with a standard
    pretrained DistilBERT checkpoint (same embedding matrix dimensions).
    """
    vocab = tokenizer.get_vocab()
    markers = (TARGET_START_MARKER, TARGET_END_MARKER)
    missing = [token for token in markers if token not in vocab]
    if missing:
        raise ValueError(f"Tokenizer vocabulary lacks target markers: {missing}")
    before_size = len(tokenizer)
    tokenizer.add_special_tokens({"additional_special_tokens": list(markers)})
    if len(tokenizer) != before_size:
        raise ValueError("Target markers would grow tokenizer vocabulary; refusing model incompatibility")
    ids = [tokenizer.convert_tokens_to_ids(token) for token in markers]
    if any(type(token_id) is not int or token_id == tokenizer.unk_token_id for token_id in ids):
        raise ValueError("Target markers must resolve to known token IDs")
    if ids[0] == ids[1]:
        raise ValueError("Target start/end markers must have distinct token IDs")
    return dict(zip(markers, ids))


def tokenize_gold_pairs(rows: list[Mapping[str, Any]], tokenizer: Any, *, max_length: int,
                        split: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Encode (marked review, aspect) as sentence pairs, without truncation.

    Refuse oversize samples instead of silently cutting review/aspect. Padding
    is intentionally deferred to the data collator of future Step 3.3.
    """
    check_tokenizer(tokenizer, max_length=max_length)
    encoded: list[dict[str, Any]] = []
    lengths: list[int] = []
    overlong: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        marked_review = mark_target_span(row["review"], row["start"], row["end"])
        sample = tokenizer(marked_review, row["aspect"], add_special_tokens=True,
                           padding=False, truncation=False, return_attention_mask=True)
        ids = sample["input_ids"]
        mask = sample["attention_mask"]
        if not isinstance(ids, list) or not ids or not isinstance(mask, list) or len(ids) != len(mask):
            raise ValueError(f"{split}[{i}]: invalid tokenizer output")
        if any(type(x) is not int or x < 0 for x in ids):
            raise ValueError(f"{split}[{i}]: invalid input_ids")
        if any(type(x) is not int or x != 1 for x in mask):
            raise ValueError(f"{split}[{i}]: attention_mask must be all ones before padding")
        # Confirm markers truly survived tokenization; otherwise a classifier
        # would again be unable to locate the targeted occurrence reliably.
        marker_ids = [tokenizer.convert_tokens_to_ids(t)
                      for t in (TARGET_START_MARKER, TARGET_END_MARKER)]
        if any(ids.count(marker_id) != 1 for marker_id in marker_ids):
            raise ValueError(f"{split}[{i}]: target marker missing/duplicated in encoded input")
        if ids.index(marker_ids[0]) >= ids.index(marker_ids[1]):
            raise ValueError(f"{split}[{i}]: target marker positions out of order")
        length = len(ids)
        lengths.append(length)
        if length > max_length:
            overlong.append({"row_index": i, "length": length, "aspect": row["aspect"]})
        encoded.append({"input_ids": ids, "attention_mask": mask, "labels": row["sentiment_id"]})
    stats = {"rows": len(rows), "over_max_length":len(overlong),
             "max_observed_length":max(lengths) if lengths else 0,
             "min_observed_length":min(lengths) if lengths else 0,
             "mean_token_length":round(sum(lengths)/len(lengths), 4) if lengths else None}
    if overlong:
        sample = overlong[:3]
        raise ValueError(f"{split}: {len(overlong)} pairs exceed max_length={max_length}; "
                         f"first examples: {sample}. No rows were truncated or saved. "
                         "Inspect examples before choosing a longer --max-length.")
    return encoded, stats


def class_imbalance_report(pairs: Mapping[str, list[Mapping[str, Any]]]) -> dict[str, Any]:
    dist: dict[str, dict[str, int]] = {}
    rates: dict[str, dict[str, float]] = {}
    for split in SPLITS:
        counts = Counter(r["sentiment_id"] for r in pairs[split])
        total = len(pairs[split])
        dist[split] = {POLARITY_ID_TO_NAME[k]:counts[k] for k in POLARITY_ID_TO_NAME}
        rates[split] = {POLARITY_ID_TO_NAME[k]:round(counts[k]/total, 6) if total else 0.0
                        for k in POLARITY_ID_TO_NAME}
    c = Counter(r["sentiment_id"] for r in pairs["train"])
    nonzero = [v for v in c.values() if v > 0]
    ratio = round(max(nonzero)/min(nonzero), 4) if nonzero else None
    n = len(pairs["train"])
    weights = {POLARITY_ID_TO_NAME[k]:round(n/(3*c[k]), 6) if c[k] > 0 else None
               for k in POLARITY_ID_TO_NAME}
    return {"class_counts": dist, "class_proportions": rates,
            "train_max_to_min_nonempty_class_ratio": ratio,
            "suggested_train_balanced_weights_NOT_APPLIED":weights,
            "weights_note":"Descriptive suggestion only; no loss weighting or training performed."}


def repeated_target_report(pairs: Mapping[str, list[Mapping[str, Any]]]) -> dict[str, Any]:
    """Count distinct gold occurrences sharing (review, aspect text).

    Repeated target text is valid, including when sentiment changes within the
    same review. We distinguish such occurrences by their gold offsets.
    """
    report: dict[str, Any] = {}
    for split in SPLITS:
        groups: dict[tuple[tuple[str, ...], str], list[Mapping[str, Any]]] = {}
        for row in pairs[split]:
            key = (review_key_from_text(row["review"]), row["aspect"].casefold())
            groups.setdefault(key, []).append(row)
        repeated = [rows for rows in groups.values() if len(rows) > 1]
        conflicting = [rows for rows in repeated if len({r["sentiment_id"] for r in rows}) > 1]
        report[split] = {
            "repeated_aspect_text_groups": len(repeated),
            "repeated_aspect_text_pairs": sum(len(rows) for rows in repeated),
            "conflicting_sentiment_repeated_text_groups": len(conflicting),
            "conflicting_sentiment_repeated_text_pairs": sum(len(rows) for rows in conflicting),
        }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ATSC Step 3.2: tokenize GOLD review/aspect pairs, NO TRAINING")
    parser.add_argument("--audit", default="artifacts/atsc_audit", help="Step 3.1 output directory")
    parser.add_argument("--ate-data", default="artifacts/ate_preprocessing", help="Step 2.1 output directory")
    parser.add_argument("--output", default="artifacts/atsc_preprocessing", help="New output directory")
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args(argv)
    audit_dir = Path(args.audit)
    ate_dir = Path(args.ate_data)
    output = Path(args.output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error(f"Output directory is not empty or not a directory: {output}")
    try:
        audit = read_audit_report(audit_dir / "audit_report.json")
        pairs = load_pairs(audit_dir / "pairs", audit)
        if not (ate_dir / "raw_splits").is_dir() or not (ate_dir / "tokenizer").is_dir():
            raise FileNotFoundError(f"Missing saved ATE raw_splits/tokenizer: {ate_dir}")
        ate_manifest_path = ate_dir / "manifest.json"
        if not ate_manifest_path.is_file():
            raise FileNotFoundError(f"ATE manifest missing: {ate_manifest_path}")
        ate_manifest = json.loads(ate_manifest_path.read_text(encoding="utf-8"))
        if ate_manifest.get("dataset") != DATASET_NAME or ate_manifest.get("tokenizer") != TOKENIZER_NAME:
            raise ValueError("ATE manifest dataset/tokenizer mismatch")
        if ate_manifest.get("review_overlap_between_final_splits") != 0:
            raise ValueError("ATE manifest reports review leakage")
        try:
            from datasets import Dataset, DatasetDict, load_from_disk
            from transformers import AutoTokenizer
        except (ImportError, ValueError) as exc:
            raise RuntimeError(f"Missing/incompatible dependencies: {exc}. See requirements_step_3_2.txt") from exc
        print("Loading ATE saved splits for membership verification (no predictions)...", flush=True)
        ate = load_from_disk(str(ate_dir / "raw_splits"))
        if any(len(ate[s]) != ate_manifest.get("split_counts", {}).get(s) for s in SPLITS):
            raise ValueError("Saved ATE raw split sizes disagree with manifest")
        checked_reviews = validate_ate_membership(pairs, ate)
        print("Loading local DistilBERT fast tokenizer (NOT model weights)...", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(str(ate_dir / "tokenizer"),
                                                   local_files_only=True, use_fast=True)
        check_tokenizer(tokenizer, max_length=args.max_length)
        marker_ids = register_target_markers(tokenizer)
        encoded_splits: dict[str, list[dict[str, Any]]] = {}
        lengths: dict[str, Any] = {}
        for split in SPLITS:
            encoded_splits[split], lengths[split] = tokenize_gold_pairs(
                pairs[split], tokenizer, max_length=args.max_length, split=split)
        stats = class_imbalance_report(pairs)
        repeated_targets = repeated_target_report(pairs)
        manifest = {
            "status":"PREPROCESSING FINISHED", "source_audit":str(audit_dir / "audit_report.json"),
            "source_dataset":DATASET_NAME,
            "gold_pairs_only":True, "ate_predictions_used":False,
            "tokenizer_name":TOKENIZER_NAME, "tokenizer_source":str(ate_dir / "tokenizer"),
            "input_format":"[CLS] review with [unused1] target [unused2] [SEP] aspect [SEP]",
            "target_disambiguation":"span_markers_from_gold_start_end_offsets",
            "target_markers":marker_ids,
            "vocabulary_resized":False,
            "max_length":args.max_length, "truncation":False, "padding":"deferred_to_training_collator",
            "label_mapping":{str(k):v for k,v in POLARITY_ID_TO_NAME.items()},
            "pair_counts":{s:len(pairs[s]) for s in SPLITS},
            "review_counts":checked_reviews,
            "review_overlap_between_splits":0,
            "encoded_token_lengths":lengths,
            "imbalance":stats,
            "repeated_target_occurrences":repeated_targets,
            "held_out_test_kept_separate":True, "training_performed":False,
            "model_weights_loaded":False, "model_predictions":None, "model_metrics":None,
        }
        # We do not create/modify output until ALL checks and encoding pass.
        output.mkdir(parents=True, exist_ok=True)
        DatasetDict({s:Dataset.from_list(encoded_splits[s]) for s in SPLITS}).save_to_disk(
            str(output / "tokenized_splits"))
        tokenizer.save_pretrained(str(output / "tokenizer"))
        (output / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
        print(f"ATSC PREPROCESSING FINISHED (NO MODEL TRAINED) -> {output.resolve()}")
        return 0
    except (ValueError, FileNotFoundError, RuntimeError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
