

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
from typing import Any, Iterable, Mapping


DATASET_NAME = "thainq107/abte-restaurants"
TOKENIZER_NAME = "distilbert/distilbert-base-uncased"
ID2LABEL = {0: "O", 1: "B-Term", 2: "I-Term"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}
ALLOWED_TAGS = set(ID2LABEL)


def review_key(tokens: list[str]) -> tuple[str, ...]:
    """A case-insensitive, word-boundary-preserving key for duplicate detection."""
    return tuple(token.strip().casefold() for token in tokens)


def validate_row(row: Mapping[str, Any], *, source: str, index: int) -> tuple[list[str], list[int]]:
    """Check original ATE schema; ignore the Polarities field deliberately."""
    if "Tokens" not in row or "Tags" not in row:
        raise ValueError(f"{source}[{index}]: missing Tokens or Tags")
    tokens, raw_tags = row["Tokens"], row["Tags"]
    if not isinstance(tokens, list) or not isinstance(raw_tags, list):
        raise ValueError(f"{source}[{index}]: Tokens/Tags must be lists")
    if not tokens or len(tokens) != len(raw_tags):
        raise ValueError(
            f"{source}[{index}]: non-empty Tokens and equal-length Tags required "
            f"(got {len(tokens)}, {len(raw_tags)})"
        )
    if any(not isinstance(token, str) or not token.strip() for token in tokens):
        raise ValueError(f"{source}[{index}]: empty/non-string token")
    try:
        tags = [int(t) for t in raw_tags]
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{source}[{index}]: non-integer BIO tag") from exc
    if any(tag not in ALLOWED_TAGS for tag in tags):
        raise ValueError(f"{source}[{index}]: labels must be one of 0/1/2")
    bio_spans(tags, source=f"{source}[{index}]")  # Fail early on invalid BIO.
    return list(tokens), tags


def bio_spans(tags: list[int], *, source: str = "row") -> list[tuple[int, int]]:
    """Return 0-based half-open word spans for valid BIO tags (strict)."""
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for idx, tag in enumerate(tags + [0]):  # sentinel closes the final span
        if tag == 1:
            if start is not None:
                spans.append((start, idx))
            start = idx
        elif tag == 2:
            if start is None:
                raise ValueError(f"{source}: I-Term at word {idx} without B-Term")
        elif tag == 0:
            if start is not None:
                spans.append((start, idx))
                start = None
        else:
            raise ValueError(f"{source}: unexpected tag {tag}")
    return spans


def tags_from_spans(length: int, spans: Iterable[tuple[int, int]]) -> list[int]:
    """Construct BIO sequence, refusing overlapping aspect boundaries."""
    tags = [0] * length
    for start, end in sorted(set(spans)):
        if not (0 <= start < end <= length):
            raise ValueError(f"Invalid aspect span ({start}, {end}) for {length} words")
        if any(tags[i] != 0 for i in range(start, end)):
            raise ValueError(f"Overlapping/conflicting aspect spans near ({start}, {end})")
        tags[start] = 1
        for index in range(start + 1, end):
            tags[index] = 2
    return tags


def group_ate_rows(rows: Iterable[Mapping[str, Any]], *, source: str) -> tuple[dict[tuple[str, ...], dict[str, Any]], dict[str, int]]:
    """Merge aspect annotations from duplicate reviews, ignoring Polarities.

    The public ABTE dataset has repeated identical Tokens/Tags rows for distinct
    aspect polarities. It is *not* safe to split its rows independently.
    """
    grouped: dict[tuple[str, ...], dict[str, Any]] = {}
    total_rows = 0
    duplicate_rows = 0
    merged_extra_spans = 0
    for idx, row in enumerate(rows):
        total_rows += 1
        tokens, tags = validate_row(row, source=source, index=idx)
        key = review_key(tokens)
        spans = set(bio_spans(tags))
        if key not in grouped:
            grouped[key] = {"Tokens": tokens, "spans": spans, "source_rows": 1}
        else:
            duplicate_rows += 1
            item = grouped[key]
            existing = item["spans"]
            union = existing | spans
            # Raises rather than silently altering conflicting annotations.
            tags_from_spans(len(tokens), union)
            merged_extra_spans += len(union) - len(existing)
            item["spans"] = union
            item["source_rows"] += 1
    for item in grouped.values():
        item["Tags"] = tags_from_spans(len(item["Tokens"]), item.pop("spans"))
    return grouped, {
        "source_rows": total_rows,
        "unique_reviews": len(grouped),
        "duplicate_rows": duplicate_rows,
        "merged_additional_aspect_spans": merged_extra_spans,
    }


def prepare_splits(
    train_rows: Iterable[Mapping[str, Any]],
    test_rows: Iterable[Mapping[str, Any]],
    *,
    validation_ratio: float = 0.10,
    seed: int = 42,
    overlap_policy: str = "drop-train",
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Preserve official test; split *unique training reviews* into train/val.

    If source train and test overlap, default removes colliding training reviews
    and reports the count; test entries remain held out. Use 'error' instead to
    stop and investigate overlaps. No group appears in more than one split.
    """
    if not (0.0 < validation_ratio < 1.0):
        raise ValueError("validation_ratio must be strictly between 0 and 1")
    if overlap_policy not in {"drop-train", "error"}:
        raise ValueError("overlap_policy must be 'drop-train' or 'error'")

    train, train_stats = group_ate_rows(train_rows, source="train")
    test, test_stats = group_ate_rows(test_rows, source="test")

    collisions = set(train) & set(test)
    if collisions and overlap_policy == "error":
        raise ValueError(f"{len(collisions)} unique review(s) overlap train and test; stop to prevent leakage")
    # Preserve official test, discard *training* collisions (report count).
    colliding_source_rows = sum(train[key]["source_rows"] for key in collisions)
    for key in collisions:
        del train[key]

    keys = sorted(train)  # stable across original row ordering
    if len(keys) < 2:
        raise ValueError("Need at least two unique training reviews after deduplication")
    random.Random(seed).shuffle(keys)
    n_val = max(1, min(len(keys) - 1, round(len(keys) * validation_ratio)))
    val_keys = set(keys[:n_val])
    train_keys = set(keys[n_val:])
    test_keys = set(test)

    assert not (train_keys & val_keys or train_keys & test_keys or val_keys & test_keys)
    result = {
        "train": [dict(train[key]) for key in sorted(train_keys)],
        "validation": [dict(train[key]) for key in sorted(val_keys)],
        "test": [dict(test[key]) for key in sorted(test_keys)],
    }
    for split in result:
        for record in result[split]:
            record["Tags"] = list(record["Tags"])

    stats = {
        "source_train": train_stats,
        "source_test": test_stats,
        "cross_source_duplicate_review_groups_removed_from_train": len(collisions),
        "cross_source_duplicate_training_rows_removed": colliding_source_rows,
        "overlap_policy": overlap_policy,
        "split_counts": {name: len(records) for name, records in result.items()},
        "seed": seed,
        "validation_ratio": validation_ratio,
        "review_overlap_between_final_splits": 0,
        "merged_spans_are_review_level": True,
    }
    return result, stats


def tokenize_and_align_labels(
    examples: Mapping[str, list[list[Any]]],
    tokenizer: Any,
    *,
    max_length: int = 128,
) -> dict[str, Any]:
    """Preserve BIO boundaries when splitting a source word into subwords.

    CLS/SEP/PAD labels are -100. For the *first* wordpiece use the original BIO
    tag; for continuation pieces: B-Term->I-Term, I-Term->I-Term, O->O.
    Unlike manual tokenization, fast tokenizer returns special tokens and
    attention_mask and maps each piece back to its original word index.
    """
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("A fast tokenizer with word_ids() is required")
    if max_length < 3:
        raise ValueError("max_length must be >= 3")
    all_tokens = examples["Tokens"]
    all_tags = examples["Tags"]
    if len(all_tokens) != len(all_tags):
        raise ValueError("Tokens and Tags batches differ in size")
    for i, (tokens, tags) in enumerate(zip(all_tokens, all_tags)):
        validate_row({"Tokens": tokens, "Tags": tags}, source="batch", index=i)

    encoded = tokenizer(
        all_tokens,
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_attention_mask=True,
        add_special_tokens=True,
    )
    # A second non-truncated pass detects *partially cut wordpieces* as well
    # as completely omitted words (important at the max_length boundary).
    untruncated = tokenizer(
        all_tokens,
        is_split_into_words=True,
        truncation=False,
        max_length=None,
        padding=False,
        return_attention_mask=False,
        add_special_tokens=True,
    )
    aligned: list[list[int]] = []
    truncation_count = 0
    truncated_span_count = 0
    for index, original_tags in enumerate(all_tags):
        tags = [int(t) for t in original_tags]
        word_ids = encoded.word_ids(batch_index=index)
        previous_word = None
        label_sequence: list[int] = []
        for word_id in word_ids:
            if word_id is None:
                label_sequence.append(-100)
            else:
                if word_id < 0 or word_id >= len(tags):
                    raise ValueError(f"Tokenizer returned invalid word id {word_id}")
                label = tags[word_id]
                if word_id == previous_word and label == 1:
                    label = 2  # continuation of B-Term, not another B-Term
                label_sequence.append(label)
            previous_word = word_id
        if len(label_sequence) != len(encoded["input_ids"][index]):
            raise ValueError("word_ids / input_ids length mismatch")
        covered = {word_id for word_id in word_ids if word_id is not None}
        full_wordpiece_counts = Counter(word_id for word_id in untruncated.word_ids(batch_index=index) if word_id is not None)
        kept_wordpiece_counts = Counter(word_id for word_id in word_ids if word_id is not None)
        if len(encoded["input_ids"][index]) < len(untruncated["input_ids"][index]):
            truncation_count += 1
            truncated_span_count += sum(
                any(kept_wordpiece_counts.get(w, 0) < full_wordpiece_counts.get(w, 0) for w in range(start, end))
                for start, end in bio_spans(tags)
            )
        if not covered:
            raise ValueError("Tokenizer did not retain any original words; increase max_length")
        aligned.append(label_sequence)

    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "labels": aligned,
        "_truncated_review_count": truncation_count,
        "_truncated_aspect_span_count": truncated_span_count,
    }


def encode_split(records: list[dict[str, Any]], tokenizer: Any, *, max_length: int) -> tuple[list[dict[str, list[int]]], dict[str, int]]:
    """Encode records in memory. Deliberately small dataset; no model loaded."""
    batch = {"Tokens": [r["Tokens"] for r in records], "Tags": [r["Tags"] for r in records]}
    out = tokenize_and_align_labels(batch, tokenizer, max_length=max_length)
    stats = {
        "truncated_reviews": out.pop("_truncated_review_count"),
        "truncated_aspect_spans": out.pop("_truncated_aspect_span_count"),
    }
    encoded_rows = [{key: value[i] for key, value in out.items()} for i in range(len(records))]
    return encoded_rows, stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare ATE BIO data; does NOT train a model")
    parser.add_argument("--dataset", default=DATASET_NAME, help="HF dataset identifier")
    parser.add_argument("--tokenizer", default=TOKENIZER_NAME, help="HF tokenizer identifier or local path")
    parser.add_argument("--output", default="artifacts/ate_preprocessing", help="New output directory")
    parser.add_argument("--validation-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--overlap-policy", choices=["drop-train", "error"], default="drop-train")
    args = parser.parse_args(argv)

    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        parser.error(f"Output already contains files: {output}. Use a fresh --output path.")

    try:
        from datasets import Dataset, DatasetDict, load_dataset
        from transformers import AutoTokenizer
    except ImportError as exc:
        parser.error(f"Missing dependencies: {exc}. Install using pip -r requirements_ate_preprocessing.txt")

    print(f"Loading dataset: {args.dataset}", flush=True)
    raw = load_dataset(args.dataset)
    if "train" not in raw or "test" not in raw:
        raise ValueError("Expected source dataset to contain 'train' and 'test' splits")

    splits, stats = prepare_splits(
        raw["train"], raw["test"],
        validation_ratio=args.validation_ratio, seed=args.seed,
        overlap_policy=args.overlap_policy,
    )
    print(f"Loading tokenizer (not model): {args.tokenizer}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    if not tokenizer.is_fast:
        raise ValueError("A fast tokenizer is required for word_ids()")
    if args.max_length > tokenizer.model_max_length:
        raise ValueError(f"max_length exceeds tokenizer limit {tokenizer.model_max_length}")

    encoded_splits: dict[str, list[dict[str, list[int]]]] = {}
    tokenization_stats: dict[str, dict[str, int]] = {}
    for split, records in splits.items():
        encoded_splits[split], tokenization_stats[split] = encode_split(records, tokenizer, max_length=args.max_length)
    stats["tokenization"] = tokenization_stats
    stats.update({
        "dataset": args.dataset,
        "tokenizer": args.tokenizer,
        "max_length": args.max_length,
        "id2label": ID2LABEL,
        "training_performed": False,
        "model_metrics": None,
    })

    output.mkdir(parents=True, exist_ok=True)
    DatasetDict({k: Dataset.from_list(v) for k, v in splits.items()}).save_to_disk(str(output / "raw_splits"))
    DatasetDict({k: Dataset.from_list(v) for k, v in encoded_splits.items()}).save_to_disk(str(output / "tokenized_splits"))
    tokenizer.save_pretrained(str(output / "tokenizer"))
    (output / "manifest.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"PREPROCESSING FINISHED (no model trained) -> {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
