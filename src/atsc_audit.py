"""Step 3.1: audit source ATSC labels and prepare review/aspect/gold-polarity pairs.

No training, inference, model import, prediction, or modification of existing ATE splits.
Uses the ATE Step 2.1 review-grouping key and BIO validation exactly.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from ate_preprocessing import (
    DATASET_NAME,
    bio_spans,
    group_ate_rows,
    review_key,
    validate_row,
)

# This mapping comes from the published ABSA dataset documentation; it is NOT
# inferred from numerical ordering or from prediction results.
POLARITY_ID_TO_NAME = {0: "negative", 1: "neutral", 2: "positive"}
NOT_LABELED = -1
SPLITS = ("train", "validation", "test")


def integer_labels(values: Any, *, source: str) -> list[int]:
    """Reject bool, floats and unexpected/non-annotated sentiment codes."""
    if not isinstance(values, list):
        raise ValueError(f"{source}: Polarities must be a list")
    labels: list[int] = []
    for i, value in enumerate(values):
        if isinstance(value, bool) or not (isinstance(value, int) or
            isinstance(value, str) and value.strip().lstrip("-").isdigit()):
            raise ValueError(f"{source}: non-integer Polarities[{i}]={value!r}")
        label = int(value)
        if label not in {*POLARITY_ID_TO_NAME, NOT_LABELED}:
            raise ValueError(f"{source}: unknown Polarities[{i}]={label}; stop rather than guessing")
        labels.append(label)
    return labels


def parse_atsc_row(row: Mapping[str, Any], *, source: str, index: int) -> tuple[list[str], list[tuple[int,int,int]], int]:
    """Extract ONLY explicitly annotated gold aspect spans.

    Other BIO spans with -1 on every token are NOT negative/neutral examples.
    For multi-word aspects every word must share the aspect's polarity.
    """
    name = f"{source}[{index}]"
    tokens, tags = validate_row(row, source=source, index=index)
    if "Polarities" not in row:
        raise ValueError(f"{name}: missing Polarities")
    polarities = integer_labels(row["Polarities"], source=name)
    if len(tokens) != len(polarities):
        raise ValueError(f"{name}: Tokens/Tags/Polarities length mismatch")
    spans = bio_spans(tags, source=name)
    covered = {word for start, end in spans for word in range(start, end)}
    for i, polarity in enumerate(polarities):
        if i not in covered and polarity != NOT_LABELED:
            raise ValueError(f"{name}: polarity {polarity} outside BIO aspect at word {i}")
    result: list[tuple[int,int,int]] = []
    unlabeled_spans = 0
    for start, end in spans:
        values = set(polarities[start:end])
        if values == {NOT_LABELED}:
            unlabeled_spans += 1
            continue
        if NOT_LABELED in values:
            raise ValueError(f"{name}: partially annotated polarity in aspect [{start},{end})")
        if len(values) != 1:
            raise ValueError(f"{name}: conflicting token polarities in aspect [{start},{end}): {sorted(values)}")
        result.append((start, end, next(iter(values))))
    return tokens, result, unlabeled_spans


def make_text_and_offsets(tokens: list[str], start: int, end: int) -> tuple[str,str,int,int]:
    """Offsets refer to a *space-joined token reconstruction*, not raw source text."""
    if not 0 <= start < end <= len(tokens):
        raise ValueError("invalid aspect word indices")
    text = " ".join(tokens)
    char_start = len(" ".join(tokens[:start])) + (1 if start else 0)
    char_end = char_start + len(" ".join(tokens[start:end]))
    aspect = text[char_start:char_end]
    if aspect != " ".join(tokens[start:end]):
        raise AssertionError("reconstructed character span mismatch")
    return text, aspect, char_start, char_end


def key_as_jsonable(key: tuple[str,...]) -> str:
    return json.dumps(key, ensure_ascii=False, separators=(",", ":"))


def build_gold_pairs(
    source_train: Iterable[Mapping[str,Any]],
    source_test: Iterable[Mapping[str,Any]],
    ate_splits: Mapping[str, Iterable[Mapping[str,Any]]],
) -> tuple[dict[str,list[dict[str,Any]]], dict[str,Any]]:
    """Honor EXACT Step 2.1 ATE split membership; never split ATSC pairs at row level.

    Raises on corrupted gold labels, inconsistent duplicates, unexpected reviews,
    mismatched ATE aspect inventory, or any cross-split review overlap.
    """
    source_train = list(source_train)
    source_test = list(source_test)
    ate_splits = {name: list(ate_splits[name]) for name in SPLITS}
    ownership: dict[tuple[str,...],str] = {}
    canonical: dict[tuple[str,...],list[str]] = {}
    ate_aspects: dict[tuple[str,...],set[tuple[int,int]]] = {}
    for split in SPLITS:
        for ix, row in enumerate(ate_splits[split]):
            tokens, tags = validate_row(row, source=f"ate/{split}", index=ix)
            key = review_key(tokens)
            if key in ownership:
                raise ValueError(f"Duplicate review in ATE split(s): {ownership[key]} and {split}")
            ownership[key] = split
            canonical[key] = tokens
            ate_aspects[key] = set(bio_spans(tags))

    # Reconcile source ATE annotations with the persisted ATE gold inventory.
    train_groups, src_train_stats = group_ate_rows(source_train, source="source/train")
    test_groups, src_test_stats = group_ate_rows(source_test, source="source/test")
    test_keys = set(test_groups)
    expected = {
        "train": set(train_groups) - test_keys,
        "test": test_keys,
    }
    actual_train = {key for key, split in ownership.items() if split in ("train", "validation")}
    actual_test = {key for key, split in ownership.items() if split == "test"}
    if actual_train != expected["train"] or actual_test != expected["test"]:
        raise ValueError("ATE saved splits do not correspond to source reviews after dedup/drop-train")
    for key, split in ownership.items():
        src_group = test_groups[key] if split == "test" else train_groups[key]
        if set(bio_spans(src_group["Tags"])) != ate_aspects[key]:
            raise ValueError(f"ATE gold spans differ from source BIO annotations for a {split} review")

    pair_labels: dict[tuple[str,...],dict[tuple[int,int],int]] = defaultdict(dict)
    occurrences = Counter()
    source_stats: dict[str,Any] = {"train":src_train_stats, "test":src_test_stats}
    unlabeled = Counter()
    source_rows_without_polarity = Counter()
    raw_label_counts: dict[str, Counter[str]] = {split: Counter() for split in SPLITS}
    dropped_cross_source = 0
    for source_name, rows in (("train", source_train), ("test", source_test)):
        for idx, row in enumerate(rows):
            tokens, annotated, missing = parse_atsc_row(row, source=source_name, index=idx)
            key = review_key(tokens)
            if source_name == "train" and key in test_keys:
                # ATE Step 2.1 already excluded the colliding training review.
                dropped_cross_source += 1
                continue
            if key not in ownership:
                raise ValueError(f"Unassigned review in source {source_name}[{idx}]")
            split = ownership[key]
            if source_name == "test" and split != "test":
                raise ValueError(f"Test review incorrectly assigned to {split}")
            if source_name == "train" and split == "test":
                raise ValueError("Source train annotation leaked into held-out test")
            unlabeled[split] += missing
            if not annotated:
                source_rows_without_polarity[split] += 1
            for start, end, polarity in annotated:
                raw_label_counts[split][POLARITY_ID_TO_NAME[polarity]] += 1
                span = (start, end)
                if span not in ate_aspects[key]:
                    raise ValueError(f"Gold polarity on span {span} missing from saved ATE BIO annotations")
                previous = pair_labels[key].get(span)
                if previous is not None and previous != polarity:
                    raise ValueError(
                        f"Conflicting polarity labels for one review/aspect {span}: "
                        f"{previous} vs {polarity}; source {source_name}[{idx}]"
                    )
                pair_labels[key][span] = polarity
                occurrences[split] += 1

    result: dict[str,list[dict[str,Any]]] = {split:[] for split in SPLITS}
    for key, spans in pair_labels.items():
        split = ownership[key]
        tokens = canonical[key]
        for (start, end), polarity in sorted(spans.items()):
            review, aspect, char_start, char_end = make_text_and_offsets(tokens,start,end)
            result[split].append({
                "review":review,
                "aspect":aspect,
                "start_word":start,
                "end_word":end,
                "start":char_start,
                "end":char_end,
                "sentiment_id":polarity,
                "sentiment":POLARITY_ID_TO_NAME[polarity],
                "split":split,
                # Original dataset supplies tokens, not raw review strings.
                "text_origin":"space_joined_source_tokens",
            })
    for split in SPLITS:
        result[split].sort(key=lambda r:(r["review"].casefold(),r["start_word"],r["end_word"],r["sentiment_id"]))

    dist = {split: {name:sum(r["sentiment"]==name for r in result[split])
                    for name in POLARITY_ID_TO_NAME.values()} for split in SPLITS}
    stats: dict[str,Any] = {
        "status":"AUDIT PASSED",
        "dataset":DATASET_NAME,
        "polarity_mapping": {str(k):v for k,v in POLARITY_ID_TO_NAME.items()},
        "unlabeled_polarity":NOT_LABELED,
        "source_rows": {"train":len(source_train), "test":len(source_test)},
        "source_statistics": source_stats,
        "ate_review_splits": {split:len(ate_splits[split]) for split in SPLITS},
        "source_cross_split_duplicate_rows_omitted":dropped_cross_source,
        "unique_paired_reviews": {split:sum(ownership[k]==split and bool(v) for k,v in pair_labels.items()) for split in SPLITS},
        "pair_counts": {split:len(result[split]) for split in SPLITS},
        "class_distribution":dist,
        "duplicate_polarity_occurrences_deduplicated": {split:occurrences[split]-len(result[split]) for split in SPLITS},
        "raw_labeled_polarity_occurrences_before_dedup": {split:dict(raw_label_counts[split]) for split in SPLITS},
        "source_rows_without_labeled_polarity": dict(source_rows_without_polarity),
        "unlabeled_aspect_spans_in_source_rows":dict(unlabeled),
        "reviews_without_atsc_gold_pair": {split:len(ate_splits[split])-sum(ownership[k]==split and bool(v) for k,v in pair_labels.items()) for split in SPLITS},
        "review_overlap_between_splits":0,
        "sentiment_train_validation_test_strategy":"reuse_exact_ate_step_2_1_saved_review_splits",
        "character_offsets":"reconstructed_space_joined_tokens_not_original_raw_review_offsets",
        "ate_gold_not_predictions":True,
        "model_trained":False,
        "model_predictions":None,
        "model_metrics":None,
    }
    return result,stats


def write_jsonl(path:Path, rows:list[dict[str,Any]]) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row,ensure_ascii=False) + "\n")


def main(argv:list[str]|None=None)->int:
    parser=argparse.ArgumentParser(description="ATSC Step 3.1 gold-label audit; NO TRAINING")
    parser.add_argument("--dataset",default=DATASET_NAME)
    parser.add_argument("--ate-data",default="artifacts/ate_preprocessing")
    parser.add_argument("--output",default="artifacts/atsc_audit")
    args=parser.parse_args(argv)
    if args.dataset != DATASET_NAME:
        parser.error(f"This audit is scoped to {DATASET_NAME}; received {args.dataset!r}")
    output=Path(args.output)
    if output.exists() and any(output.iterdir()):
        parser.error(f"Output directory not empty: {output}; use a fresh path to avoid overwrites")
    ate_dir=Path(args.ate_data)
    manifest_path=ate_dir/"manifest.json"
    if not manifest_path.is_file() or not (ate_dir/"raw_splits").is_dir():
        parser.error(f"ATE Step 2.1 artifacts missing: {ate_dir}")
    manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dataset") != args.dataset:
        parser.error("Dataset does not match ATE preprocessing manifest")
    if manifest.get("review_overlap_between_final_splits") != 0:
        parser.error("ATE preprocessing manifest indicates review overlap")
    try:
        from datasets import load_dataset, load_from_disk
    except ImportError as exc:
        parser.error(f"Missing datasets dependency: {exc}")
    print(f"Loading source dataset with Polarities: {args.dataset}",flush=True)
    raw=load_dataset(args.dataset)
    if not {"train","test"}.issubset(raw.keys()):
        parser.error("Source dataset requires train and test")
    print("Loading saved ATE gold review splits (NOT predictions)...",flush=True)
    ate=load_from_disk(str(ate_dir/"raw_splits"))
    if not all(split in ate for split in SPLITS):
        parser.error("ATE raw_splits must have train/validation/test")
    if any(len(ate[split])!=manifest.get("split_counts",{}).get(split) for split in SPLITS):
        parser.error("ATE saved split lengths differ from manifest")
    pairs,report=build_gold_pairs(raw["train"],raw["test"],ate)
    report["ate_preprocessing_manifest"] = str(manifest_path)
    report["source_dataset_split_fingerprints"] = {
        split:getattr(raw[split],"_fingerprint",None) for split in ("train","test")
    }
    output.mkdir(parents=True,exist_ok=True)
    for split in SPLITS:
        write_jsonl(output/"pairs"/f"{split}.jsonl",pairs[split])
    (output/"audit_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2))
    print(f"ATSC AUDIT FINISHED (NO MODEL TRAINED) -> {output.resolve()}")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
