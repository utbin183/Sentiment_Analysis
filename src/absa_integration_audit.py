"""Phase 4.1 — read-only compatibility audit of ATE -> ATSC contracts.

This is NOT an ATE->ATSC prediction pipeline. It never feeds ATE predictions
into ATSC and never reads train/validation/test datasets or trains models.

Default: inspect existing code and local saved metadata, exercise deterministic
text/span fixtures. Optional --check-tokenizers / --check-model-loading do
read-only tokenizer/model checks and separate *manual* inference smoke tests.
"""
from __future__ import annotations

import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

import argparse
from collections import Counter
import hashlib
import inspect
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import ate_inference as ate
import atsc_inference as atsc
import atsc_preprocessing as pre

LABELS = {0: "negative", 1: "neutral", 2: "positive"}
BIO_LABELS = {0: "O", 1: "B-Term", 2: "I-Term"}
INPUT_FORMAT = "[CLS] review with [unused1] target [unused2] [SEP] aspect [SEP]"
MARKERS = {"[unused1]": 2, "[unused2]": 3}


def _finding(level: str, code: str, detail: str) -> dict[str, str]:
    if level not in ("INFO", "RISK", "BLOCKER"):
        raise ValueError("Invalid audit finding severity")
    return {"level": level, "code": code, "detail": detail}


def check_interface_contracts() -> dict[str, Any]:
    """Importable Python API checks, not a prediction pipeline."""
    required = {
        "ate.extract": (ate.AspectExtractor.extract, ("self", "review")),
        "atsc.predict": (atsc.ATSCClassifier.predict, ("self", "review", "aspect", "start", "end")),
        "atsc.validate_span": (atsc.validate_span, ("review", "aspect", "start", "end")),
        "atsc.encode_pair": (atsc.encode_pair, ("tokenizer", "review", "aspect", "start", "end")),
    }
    for name, (func, expected) in required.items():
        actual = tuple(inspect.signature(func).parameters)
        if actual[:len(expected)] != expected:
            raise ValueError(f"Incompatible function signature for {name}: {actual}")
    if ate.ID2LABEL != BIO_LABELS:
        raise ValueError("ATE BIO mapping changed")
    if atsc.ID2LABEL != LABELS or atsc.MARKERS != MARKERS:
        raise ValueError("ATSC sentiment mapping or target marker IDs changed")
    if atsc.mark_target_span is not pre.mark_target_span:
        raise ValueError("ATSC must reuse the Step 3.2 mark_target_span() function")
    if (ate.DEFAULT_MAX_LENGTH != 128 or atsc.DEFAULT_DATA != "artifacts/atsc_preprocessing"
            or not callable(ate.load_extractor) or not callable(atsc.load_classifier)):
        raise ValueError("Unexpected model loading defaults or ATE token budget")
    return {
        "status": "PASS",
        "ate_python_api": "AspectExtractor.extract(review) -> list[aspect]",
        "atsc_python_api": "ATSCClassifier.predict(review, aspect, start, end) -> result",
        "ate_output_schema": ["text", "start", "end", "confidence"],
        "atsc_required_input": ["review", "aspect", "start", "end"],
        "offset_unit": "zero-based Python Unicode character indexes; half-open [start,end)",
        "marker_helper_identity": True,
        "ate_bio_classes": BIO_LABELS,
        "atsc_sentiment_classes": LABELS,
        "shared_base_backbone": "distilbert-base-uncased",
        "heads": {"ate": "token_classification", "atsc": "sequence_classification"},
        "model_loading": {"ate": "load_extractor() once", "atsc": "load_classifier() once"},
        "inference_uses_test_or_training_data": False,
    }


def validate_ate_aspects(review: str, aspects: Any) -> dict[str, Any]:
    """Validate prospective ATE return values WITHOUT passing them to ATSC."""
    if not isinstance(review, str) or not review.strip():
        raise ValueError("Expected nonempty literal review")
    if not isinstance(aspects, list):
        raise ValueError("ATE output must be list[dict]")
    previous_end = 0
    checked: list[dict[str, Any]] = []
    for index, aspect in enumerate(aspects):
        if not isinstance(aspect, Mapping) or not {"text", "start", "end", "confidence"}.issubset(aspect):
            raise ValueError(f"ATE aspect[{index}] has wrong schema")
        start, end = aspect["start"], aspect["end"]
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(review):
            raise ValueError(f"ATE aspect[{index}] invalid half-open offsets")
        if start < previous_end:
            raise ValueError("ATE aspects must be ordered and non-overlapping")
        if not isinstance(aspect["text"], str) or review[start:end] != aspect["text"]:
            raise ValueError(f"ATE aspect[{index}] text does not match original review")
        conf = aspect["confidence"]
        if type(conf) not in (float, int) or not math.isfinite(conf) or not 0 <= conf <= 1:
            raise ValueError(f"ATE aspect[{index}] invalid uncalibrated confidence")
        checked.append({"text": aspect["text"], "start": start, "end": end})
        previous_end = end
    return {"status": "PASS", "aspect_count": len(checked), "verified_spans": checked}


def validate_atsc_result(review: str, aspect: str, start: int, end: int, output: Any) -> dict[str, Any]:
    """Validate an independent ATSC prediction JSON schema (no model call)."""
    atsc.validate_span(review, aspect, start, end)
    if not isinstance(output, Mapping):
        raise ValueError("ATSC output must be dict")
    expected = {"review", "aspect", "sentiment", "label_id",
                "probabilities_softmax_uncalibrated", "predicted_probability_uncalibrated"}
    if not expected.issubset(output):
        raise ValueError("ATSC output missing required fields")
    if output["review"] != review or output["aspect"] != {"text": aspect, "start": start, "end": end}:
        raise ValueError("ATSC returned a review/aspect/offset mismatch")
    label = output["label_id"]
    if type(label) is not int or label not in LABELS or output["sentiment"] != LABELS[label]:
        raise ValueError("ATSC label ID and sentiment name disagree")
    probabilities = output["probabilities_softmax_uncalibrated"]
    if not isinstance(probabilities, Mapping) or set(probabilities) != set(LABELS.values()):
        raise ValueError("ATSC must return probabilities for negative/neutral/positive")
    values = [probabilities[LABELS[k]] for k in LABELS]
    winner, checked = atsc.validate_probabilities(values)
    if winner != label:
        raise ValueError("ATSC predicted label differs from probability argmax")
    p = output["predicted_probability_uncalibrated"]
    if type(p) not in (float, int) or not math.isfinite(p) or abs(p - checked[LABELS[label]]) > 1e-5:
        raise ValueError("ATSC predicted_probability does not match winning class")
    return {"status": "PASS", "predicted_class": LABELS[label], "softmax_is_calibrated": False}


def span_fixture_cases() -> list[dict[str, Any]]:
    """Synthetic hand-written input contracts only; no gold dataset or model predictions."""
    specs = [
        ("punctuation", "The food was delicious, but the service was terrible.", "food", "first"),
        ("punctuation_second", "The food was delicious, but the service was terrible.", "service", "first"),
        ("repeated_first", "The service was great, but later the service was terrible.", "service", "first"),
        ("repeated_last", "The service was great, but later the service was terrible.", "service", "last"),
        ("multi_word", "The dining room was pleasant.", "dining room", "first"),
        ("unicode", "Café food was excellent.", "food", "first"),
        ("extra_whitespace", "  The food\twas great.  ", "food", "first"),
        ("newline", "The food\nwas excellent.", "food", "first"),
        ("apostrophe", "Chef's food was great.", "food", "first"),
    ]
    cases = []
    for name, text, aspect, occurrence in specs:
        start = text.index(aspect) if occurrence == "first" else text.rindex(aspect)
        cases.append({"name": name, "review": text, "aspect": aspect,
                      "start": start, "end": start + len(aspect)})
    return cases


def audit_span_fixtures() -> dict[str, Any]:
    summary: list[dict[str, Any]] = []
    reconstructed_differences = 0
    for case in span_fixture_cases():
        review, aspect, start, end = (case[k] for k in ("review", "aspect", "start", "end"))
        words, offsets = ate.split_words_with_offsets(review)
        if not words or any(review[s:e] != word for word, (s, e) in zip(words, offsets)):
            raise ValueError("ATE tokenization modifies raw text span or loses order")
        if any(a[1] > b[0] for a, b in zip(offsets, offsets[1:])):
            raise ValueError("ATE token spans overlap")
        atsc.validate_span(review, aspect, start, end)
        reconstructed_differences += (" ".join(words) != review)
        marked = pre.mark_target_span(review, start, end)
        if marked.count(pre.TARGET_START_MARKER) != 1 or marked.count(pre.TARGET_END_MARKER) != 1:
            raise ValueError("ATSC target marker count is invalid")
        if marked.index(pre.TARGET_START_MARKER) >= marked.index(pre.TARGET_END_MARKER):
            raise ValueError("ATSC target markers reversed")
        summary.append({"case": case["name"], "aspect": aspect, "start": start, "end": end,
                        "ate_word_count": len(words), "marked_text": marked})
    pair = {x["case"]: x["marked_text"] for x in summary}
    if pair["repeated_first"] == pair["repeated_last"]:
        raise ValueError("Two occurrences of the same aspect must have distinct ATSC marked inputs")
    return {"status": "PASS", "synthetic_cases": len(summary), "repeated_occurrences_distinguished": True,
            "raw_text_differs_from_space_joined_ate_tokens_in_cases": reconstructed_differences,
            "cases": summary}


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required artifact missing: {path}")
    info = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(info, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return info


def _config_labels(config: Mapping[str, Any]) -> dict[int, str]:
    raw = config.get("id2label")
    if not isinstance(raw, Mapping):
        raise ValueError("Saved model config lacks id2label")
    try:
        return {int(k): v for k, v in raw.items()}
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid model config label IDs") from exc


def check_saved_artifacts(ate_model: Path, atsc_model: Path, atsc_data: Path,
                          training_dir: Path) -> dict[str, Any]:
    """Read ONLY manifests/configs/file metadata. No model weights or dataset rows."""
    info = _load_json(atsc_data / "manifest.json")
    max_length = atsc.read_preprocessing_manifest(atsc_data)
    if info.get("held_out_test_kept_separate") is not True or info.get("ate_predictions_used") is not False:
        raise ValueError("ATSC preprocessing provenance is not gold-only or held-out")
    train = _load_json(training_dir / "run_summary.json")
    if (train.get("status") != "TRAINING FINISHED" or train.get("training_performed") is not True
            or train.get("test_used") is not False or train.get("selection_metric") != "eval_macro_f1"):
        raise ValueError("ATSC training was not validated correctly, or test was used")
    ate.require_local_model(ate_model)
    atsc.check_local_model_dir(atsc_model)
    ate_conf = _load_json(ate_model / "config.json")
    atsc_conf = _load_json(atsc_model / "config.json")
    for name, conf, required_labels, architecture in (
        ("ATE", ate_conf, BIO_LABELS, "DistilBertForTokenClassification"),
        ("ATSC", atsc_conf, LABELS, "DistilBertForSequenceClassification"),
    ):
        if conf.get("model_type") != "distilbert" or _config_labels(conf) != required_labels:
            raise ValueError(f"{name}: wrong model type or label mapping")
        architectures = conf.get("architectures", [])
        if architectures and architecture not in architectures:
            raise ValueError(f"{name}: incorrect model head architecture {architectures!r}")
        if conf.get("vocab_size") != 30522:
            raise ValueError(f"{name}: unexpected embedding vocabulary size")
    if atsc_conf.get("label2id") != {v: k for k, v in LABELS.items()}:
        raise ValueError("ATSC label2id mapping differs from Step 3.3")
    if str(train.get("best_model_dir", "")).replace("\\", "/").split("/")[-1] != "best_model":
        raise ValueError("Training summary does not point at best_model")
    best_ckpt = str(train.get("best_checkpoint", "")).replace("\\", "/").split("/")[-1]
    if not re.fullmatch(r"checkpoint-\d+", best_ckpt):
        raise ValueError("Invalid best checkpoint name in training summary")
    return {"status": "PASS", "ate_model_dir": str(ate_model), "atsc_model_dir": str(atsc_model),
            "ate_head": "token_classification", "atsc_head": "sequence_classification",
            "shared_vocab_size": 30522, "atsc_max_length": max_length,
            "best_checkpoint": best_ckpt, "checkpoint_selected_by": "validation_macro_f1",
            "test_used_for_selection": False, "weights_loaded": False}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_selected_checkpoint(atsc_model: Path, training_dir: Path) -> dict[str, Any]:
    """Read-only byte comparison to selected original checkpoint. No changes."""
    summary = _load_json(training_dir / "run_summary.json")
    checkpoint = str(summary.get("best_checkpoint", "")).replace("\\", "/").split("/")[-1]
    if not re.fullmatch(r"checkpoint-\d+", checkpoint):
        raise ValueError("Training summary has invalid checkpoint")
    weight = atsc_model / "model.safetensors"
    original = training_dir / "checkpoints" / checkpoint / "model.safetensors"
    if not weight.is_file() or not original.is_file():
        raise FileNotFoundError(f"Selected checkpoint or best_model weights missing: {weight} / {original}")
    file_a, file_b = sha256_file(weight), sha256_file(original)
    if file_a != file_b:
        raise ValueError("BLOCKER: best_model weights differ from the selected validation checkpoint")
    # Note: equal hashes verify identity; HF loader or safetensors must still load them.
    return {"status": "PASS", "checkpoint": checkpoint, "identical": True,
            "sha256": file_a, "bytes": weight.stat().st_size,
            "weights_modified": False}


def check_tokenizer_pair(ate_model: Path, atsc_model: Path, atsc_data: Path) -> dict[str, Any]:
    """Read tokenizers only; inspect sentence-pair budget, no model weights."""
    from transformers import AutoTokenizer
    ate_tok = AutoTokenizer.from_pretrained(str(ate_model), local_files_only=True, use_fast=True)
    atsc_tok = AutoTokenizer.from_pretrained(str(atsc_model), local_files_only=True, use_fast=True)
    max_length = atsc.read_preprocessing_manifest(atsc_data)
    if getattr(ate_tok, "is_fast", False) is not True:
        raise ValueError("ATE tokenizer is not fast")
    atsc.validate_tokenizer(atsc_tok, max_length)
    if len(ate_tok) != 30522 or len(atsc_tok) != 30522:
        raise ValueError("ATE/ATSC tokenizer vocabulary size differs")

    scenarios = span_fixture_cases()
    # This intentionally tests the 128-token boundary; ATE can fit while ATSC
    # may not because ATSC additionally encodes the aspect and target markers.
    long_review = "food " + "nice " * 122 + "good"
    scenarios.append({"name": "boundary_stress", "review": long_review, "aspect": "food", "start": 0, "end": 4})
    results = []
    for case in scenarios:
        words, _ = ate.split_words_with_offsets(case["review"])
        ids_ate = ate_tok([words], is_split_into_words=True, add_special_tokens=True,
                          truncation=False, padding=False)["input_ids"][0]
        marked = pre.mark_target_span(case["review"], case["start"], case["end"])
        raw = atsc_tok(marked, case["aspect"], add_special_tokens=True,
                       padding=False, truncation=False, return_attention_mask=True)
        ids_atsc = raw["input_ids"]
        if raw["attention_mask"] != [1] * len(ids_atsc):
            raise ValueError("ATSC tokenizer produced non-unit mask in unpadded sample")
        if ids_atsc.count(2) != 1 or ids_atsc.count(3) != 1 or ids_atsc.index(2) >= ids_atsc.index(3):
            raise ValueError("ATSC tokenizer has missing/reversed occurrence marker IDs")
        results.append({"case": case["name"], "ate_tokens": len(ids_ate),
                        "atsc_tokens": len(ids_atsc), "ate_within_128": len(ids_ate) <= 128,
                        "atsc_within_preprocessing_limit": len(ids_atsc) <= max_length})
    return {"status": "PASS", "ate_tokenizer": "fast pretrained DistilBERT", "atsc_tokenizer": "fast DistilBERT with registered [unused] markers",
            "model_weights_loaded": False, "scenarios": results,
            "budget_divergence_cases": [x["case"] for x in results if x["ate_within_128"] and not x["atsc_within_preprocessing_limit"]]}


def smoke_test_models(ate_model: Path, atsc_model: Path, atsc_data: Path, device: str) -> dict[str, Any]:
    """Independent read-only inference smoke tests: NOT an integrated pipeline.

    ATE output is never passed into ATSC. ATSC receives manually supplied
    synthetic aspect offsets, so this cannot be an end-to-end evaluation.
    """
    if device not in ("cpu", "cuda", "auto"):
        raise ValueError("device must be cpu/cuda/auto")
    extractor = ate.load_extractor(model_dir=ate_model, device=device)
    classifier = atsc.load_classifier(model_dir=atsc_model, preprocessing_dir=atsc_data, device=device)
    review = "The food was delicious, but the service was terrible."
    extracted = extractor.extract(review)
    ate_check = validate_ate_aspects(review, extracted)
    manual = [("food", 4, 8), ("service", 32, 39)]
    atsc_checks = []
    for aspect, start, end in manual:
        output = classifier.predict(review, aspect, start, end)
        atsc_checks.append({"aspect": aspect, **validate_atsc_result(review, aspect, start, end, output)})
    return {"status": "PASS", "ate_independent_inference": ate_check,
            "atsc_independent_manual_inference": atsc_checks,
            "ate_predictions_passed_to_atsc": False, "gold_test_data_used": False,
            "trained_weights_updated": False}


def build_report(*, ate_model: Path, atsc_model: Path, atsc_data: Path,
                 training_dir: Path, check_tokenizers: bool = False,
                 check_models: bool = False, verify_checkpoint: bool = False,
                 device: str = "cpu", strict_artifacts: bool = True) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    findings = [
        _finding("RISK", "TOKENIZATION_DISTRIBUTION_SHIFT",
                 "ATE inference uses regex word/punctuation splitting of raw text; training used source Tokens. "
                 "ATSC training used space-joined source Tokens, but runtime can receive raw punctuation/whitespace. "
                 "The exact same original review must be passed to both modules, without normalizing/rebuilding it."),
        _finding("RISK", "TOKEN_BUDGET_MISMATCH",
                 "ATE accepts up to 128 model tokens; ATSC uses a 128-token review+target pair including 2 extra markers. "
                 "An ATE-accepted review can still exceed the ATSC budget; do not truncate or silently drop an aspect."),
        _finding("RISK", "MODEL_DOMAIN_LIMITATION",
                 "Separate gold-aspect ATSC test metrics cannot be used as end-to-end ATE->ATSC performance; "
                 "no pipeline performance was measured in this audit."),
        _finding("INFO", "PYTHON_OFFSETS",
                 "Both Python modules use Unicode code point indexes, not UTF-8 byte offsets or JavaScript UTF-16 indices."),
        _finding("INFO", "MODEL_INSTANCES",
                 "ATE and ATSC require different task heads; load each model once and keep raw review unchanged."),
    ]
    checks["interfaces"] = check_interface_contracts()
    checks["text_and_offsets"] = audit_span_fixtures()
    try:
        checks["saved_artifacts"] = check_saved_artifacts(ate_model, atsc_model, atsc_data, training_dir)
    except (ValueError, FileNotFoundError, KeyError, TypeError, json.JSONDecodeError) as exc:
        if strict_artifacts:
            findings.append(_finding("BLOCKER", "LOCAL_ARTIFACTS", str(exc)))
        else:
            checks["saved_artifacts"] = {"status": "NOT_CHECKED", "reason": str(exc)}
    if verify_checkpoint:
        try:
            checks["checkpoint_integrity"] = verify_selected_checkpoint(atsc_model, training_dir)
        except (ValueError, FileNotFoundError, OSError) as exc:
            findings.append(_finding("BLOCKER", "CHECKPOINT_INTEGRITY", str(exc)))
    if check_tokenizers:
        try:
            checks["tokenizers"] = check_tokenizer_pair(ate_model, atsc_model, atsc_data)
            if checks["tokenizers"]["budget_divergence_cases"]:
                findings.append(_finding("RISK", "TOKEN_BUDGET_DIVERGENCE_REPRODUCED",
                                         "See tokenizers.budget_divergence_cases: ATE fits but ATSC would refuse."))
        except Exception as exc:
            findings.append(_finding("BLOCKER", "TOKENIZER_CHECK", f"{type(exc).__name__}: {exc}"))
    if check_models:
        try:
            checks["model_smoke"] = smoke_test_models(ate_model, atsc_model, atsc_data, device)
        except Exception as exc:
            findings.append(_finding("BLOCKER", "MODEL_LOADING_OR_SMOKE", f"{type(exc).__name__}: {exc}"))
    blocked = any(x["level"] == "BLOCKER" for x in findings)
    return {
        "phase": "4.1", "status": "AUDIT BLOCKED" if blocked else "AUDIT READY WITH RISKS",
        "audit_only": True, "pipeline_created": False, "training_performed": False,
        "test_evaluation_performed": False, "weights_modified": False,
        "coverage": {
            "static_interfaces_and_synthetic_offsets": True,
            "saved_model_metadata": checks.get("saved_artifacts", {}).get("status") == "PASS",
            "local_tokenizers_checked": checks.get("tokenizers", {}).get("status") == "PASS",
            "selected_checkpoint_sha256_verified": checks.get("checkpoint_integrity", {}).get("status") == "PASS",
            "both_models_loaded_and_independently_smoke_tested": checks.get("model_smoke", {}).get("status") == "PASS",
            "pipeline_end_to_end_tested": False,
        },
        "checks": checks, "findings": findings,
        "next_step": "Design offset-preserving adapter and overflow policy before implementing Stage 4 pipeline"
                     if not blocked else "Resolve blockers, rerun audit before Stage 4 pipeline",
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Phase 4.1 read-only ATE/ATSC integration audit; NOT the pipeline")
    p.add_argument("--ate-model", type=Path, default=Path("artifacts/ate_training/best_model"))
    p.add_argument("--atsc-model", type=Path, default=Path("artifacts/atsc_training/best_model"))
    p.add_argument("--atsc-data", type=Path, default=Path("artifacts/atsc_preprocessing"))
    p.add_argument("--training", type=Path, default=Path("artifacts/atsc_training"))
    p.add_argument("--output", type=Path, default=Path("artifacts/integration_audit/audit_report.json"))
    p.add_argument("--dry-run", action="store_true", help="Print report, do not write output or load weights")
    p.add_argument("--check-tokenizers", action="store_true", help="Read real local tokenizers; no weights")
    p.add_argument("--verify-checkpoint", action="store_true", help="SHA-256 selected checkpoint vs best_model, read-only")
    p.add_argument("--check-model-loading", action="store_true", help="Load both trained models; run separate manual smoke tests, NO pipeline")
    p.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    p.add_argument("--allow-missing-artifacts", action="store_true", help="For portable unit-test/demo only, not acceptance on Windows")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.dry_run and args.check_model_loading:
        raise SystemExit("--dry-run cannot be combined with --check-model-loading; omit --dry-run for model loading")
    if not args.dry_run and args.output.exists():
        raise SystemExit(f"Refusing to overwrite audit report: {args.output}")
    report = build_report(ate_model=args.ate_model, atsc_model=args.atsc_model,
                          atsc_data=args.atsc_data, training_dir=args.training,
                          check_tokenizers=args.check_tokenizers,
                          check_models=args.check_model_loading,
                          verify_checkpoint=args.verify_checkpoint,
                          device=args.device, strict_artifacts=not args.allow_missing_artifacts)
    report_text = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    print(report_text, end="")
    if not args.dry_run:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(report_text)
        print(f"READ-ONLY AUDIT REPORT -> {args.output}")
    return 2 if report["status"] == "AUDIT BLOCKED" else 0


if __name__ == "__main__":
    raise SystemExit(main())
