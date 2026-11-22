from __future__ import annotations

import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping
from transformers import AutoTokenizer
import atsc_inference as atsc
from atsc_preprocessing import mark_target_span

DEFAULT_PREPROCESSING = "artifacts/atsc_preprocessing"
REQUIRED_MAX_LENGTH = 128


def require_review(review: Any) -> str:
    if not isinstance(review, str) or not review.strip():
        raise ValueError("review must be a non-empty string; offsets refer to exactly this text")
    return review


def require_payload(payload: Any) -> tuple[str, list[Any]]:
    """Validate the top-level ATE JSON without changing review or input order."""
    if not isinstance(payload, dict):
        raise ValueError("ATE result must be a JSON object with review and aspects")
    if "review" not in payload or "aspects" not in payload:
        raise ValueError("ATE result requires both 'review' and 'aspects'")
    review = require_review(payload["review"])
    aspects = payload["aspects"]
    if not isinstance(aspects, list):
        raise ValueError("ATE result 'aspects' must be a list (empty is allowed)")
    return review, aspects


def validate_ate_record(review: str, raw: Any) -> dict[str, Any]:
    """ATE Step 2.4 contract: literal text/offsets, uncalibrated confidence."""
    if not isinstance(raw, dict):
        raise AdapterError("INVALID_ASPECT_RECORD", "Each ATE aspect must be a JSON object")
    for key in ("text", "start", "end", "confidence"):
        if key not in raw:
            raise AdapterError("INVALID_ASPECT_RECORD", f"ATE aspect is missing {key!r}")
    text, start, end, confidence = (raw[k] for k in ("text", "start", "end", "confidence"))
    try:
        atsc.validate_span(review, text, start, end)
    except (ValueError, TypeError) as exc:
        raise AdapterError("INVALID_SPAN", str(exc)) from exc
    if type(confidence) not in (float, int) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise AdapterError("INVALID_ATE_CONFIDENCE", "ATE confidence must be a finite number in [0,1], not boolean")
    return {"text": text, "start": start, "end": end, "ate_confidence_uncalibrated": float(confidence)}


class AdapterError(ValueError):
    def __init__(self, code: str, message: str, *, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.details = details or {}


def _failure(index: int, raw: Any, problem: AdapterError, *, aspect: dict[str, Any] | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {"index": index, "status": "ERROR", "error": {
        "code": problem.code, "message": str(problem), "details": problem.details,
    }}
    if aspect is None:
        result["source_aspect"] = raw
    else:
        result["aspect"] = aspect
    return result


def _token_count(tokenizer: Any, review: str, aspect: dict[str, Any]) -> int:
    """Measure the exact ATSC pair without truncation, preserving original review."""
    try:
        marked = mark_target_span(review, aspect["start"], aspect["end"])
    except (TypeError, ValueError) as exc:
        raise AdapterError("MARKER_COLLISION_OR_SPAN", str(exc)) from exc
    try:
        encoded = tokenizer(marked, aspect["text"], add_special_tokens=True,
                            padding=False, truncation=False, return_attention_mask=True)
        ids = encoded["input_ids"]
    except (TypeError, ValueError, KeyError) as exc:
        raise AdapterError("TOKENIZATION_FAILED", str(exc)) from exc
    if not isinstance(ids, list) or any(type(x) is not int for x in ids):
        raise AdapterError("TOKENIZATION_FAILED", "Tokenizer returned malformed input_ids")
    return len(ids)


def adapt_ate_output(payload: Any, tokenizer: Any, *, max_length: int = REQUIRED_MAX_LENGTH) -> dict[str, Any]:
    """Prepare supplied ATE predictions, preserving every input index.

    This is NOT an end-to-end model pipeline. It never calls extract() or predict().
    Use saved Step 3.2 tokenizer and max_length=128, not a freshly trained model.
    """
    review, raw_aspects = require_payload(payload)
    if type(max_length) is not int or max_length != REQUIRED_MAX_LENGTH:
        raise ValueError("Adapter must retain the trained ATSC max_length=128; no silent changes")
    atsc.validate_tokenizer(tokenizer, max_length)

    results: list[dict[str, Any]] = []
    seen: list[tuple[int, int]] = []
    for index, raw in enumerate(raw_aspects):
        try:
            aspect = validate_ate_record(review, raw)
        except AdapterError as problem:
            results.append(_failure(index, raw, problem))
            continue
        start, end = aspect["start"], aspect["end"]
        if (start, end) in seen:
            results.append(_failure(index, raw, AdapterError(
                "DUPLICATE_SPAN", "The same exact aspect occurrence appeared more than once",
                details={"start": start, "end": end}), aspect=aspect))
            continue
        if any(start < prev_end and prev_start < end for prev_start, prev_end in seen):
            results.append(_failure(index, raw, AdapterError(
                "OVERLAPPING_SPAN", "ATE predicted overlapping aspect occurrences",
                details={"start": start, "end": end}), aspect=aspect))
            continue
        # Reserve an occurrence even when it overflows, to make duplicates visible.
        seen.append((start, end))
        try:
            count = _token_count(tokenizer, review, aspect)
            if count > max_length:
                raise AdapterError(
                    "ATSC_TOKEN_OVERFLOW",
                    f"ATSC needs {count} tokens, over the trained limit {max_length}; NOT truncated",
                    details={"required_tokens": count, "max_length": max_length,
                             "excess_tokens": count - max_length, "truncation": False})
            
            encoded = atsc.encode_pair(tokenizer, review, aspect["text"], start, end,
                                       max_length=max_length)
            if len(encoded["input_ids"]) != count:
                raise AdapterError("TOKENIZATION_UNSTABLE", "Tokenizer yielded different lengths across two identical encodings")
        except AdapterError as problem:
            results.append(_failure(index, raw, problem, aspect=aspect))
            continue
        except (TypeError, ValueError, KeyError) as exc:
            results.append(_failure(index, raw, AdapterError("ATSC_ENCODING_FAILED", str(exc)), aspect=aspect))
            continue
        results.append({
            "index": index, "status": "READY_FOR_ATSC", "aspect": aspect,
            "atsc_input": {"aspect": aspect["text"], "start": start, "end": end},
            "token_budget": {"used": count, "limit": max_length, "truncated": False},
        })

    ready = sum(row["status"] == "READY_FOR_ATSC" for row in results)
    failed = len(results) - ready
    if not results:
        status = "NO_ASPECTS"
    elif failed == 0:
        status = "READY"
    elif ready == 0:
        status = "REJECTED"
    else:
        status = "PARTIAL"
    return {
        "step": "4.2", "status": status, "review": review,
        "offset_contract": "python_unicode_codepoints_half_open", "review_modified": False,
        "atsc_input_format": "[CLS] review with [unused1] target [unused2] [SEP] aspect [SEP]",
        "max_atsc_tokens": max_length, "overflow_policy": "per_aspect_structured_error_no_truncation",
        "summary": {"input_aspects": len(raw_aspects), "ready": ready, "errors": failed},
        "aspects": results, "ate_extraction_performed": False,
        "atsc_predictions_performed": False, "training_performed": False,
        "gold_test_evaluation_performed": False, "pipeline_created": False,
    }


def load_tokenizer(preprocessing_dir: str | Path) -> tuple[Any, int]:
    """Load ONLY the tokenizer and manifest from Step 3.2 (not model weights)."""
    max_length = atsc.read_preprocessing_manifest(preprocessing_dir)
    if max_length != REQUIRED_MAX_LENGTH:
        raise ValueError(f"ATSC was trained with max_length={max_length}; this adapter expects exactly 128")
    folder = Path(preprocessing_dir) / "tokenizer"
    if not folder.is_dir():
        raise FileNotFoundError(f"Missing saved ATSC tokenizer folder: {folder}")
    
    tokenizer = AutoTokenizer.from_pretrained(str(folder), local_files_only=True, use_fast=True)
    atsc.validate_tokenizer(tokenizer, max_length)
    return tokenizer, max_length


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Phase 4.2 ATE-output -> ATSC-input adapter; NO models/inference/training/pipeline")
    p.add_argument("--ate-json", required=True, type=Path,
                   help="JSON output from src/ate_inference.py, with review and aspects")
    p.add_argument("--preprocessing", default=DEFAULT_PREPROCESSING,
                   help="Saved Step 3.2 directory (manifest and tokenizer only)")
    p.add_argument("--output", type=Path, help="Optional new JSON output file; no overwrite")
    p.add_argument("--dry-run", action="store_true", help="Prepare and print, never write output")
    p.add_argument("--strict", action="store_true", help="Exit nonzero when any aspect is rejected")
    return p


def main(argv: list[str] | None = None) -> int:
    p = build_parser()
    args = p.parse_args(argv)
    if args.output is not None and args.output.exists() and not args.dry_run:
        p.error(f"Output already exists; refusing overwrite: {args.output}")
    try:
        payload = json.loads(args.ate_json.read_text(encoding="utf-8-sig"))
        require_payload(payload)  # fail before loading tokenizer when malformed
        tokenizer, max_length = load_tokenizer(args.preprocessing)
        result = adapt_ate_output(payload, tokenizer, max_length=max_length)
        rendered = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        if args.output is not None and not args.dry_run:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as f:
                f.write(rendered)
        print(rendered, end="")
        return 2 if args.strict and result["summary"]["errors"] else 0
    except (ValueError, TypeError, KeyError, FileNotFoundError, OSError, UnicodeError, ImportError) as exc:
        p.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
