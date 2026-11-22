from __future__ import annotations

import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

import ate_inference as ate
import atsc_inference as atsc
import absa_adapter as adapter

DEFAULT_ATE_MODEL = ate.DEFAULT_MODEL
DEFAULT_ATSC_MODEL = atsc.DEFAULT_MODEL
DEFAULT_PREPROCESSING = atsc.DEFAULT_DATA
OFFSET_CONTRACT = "python_unicode_codepoints_half_open"


def _error(code: str, message: str, *, exception_type: str | None = None) -> dict[str, Any]:
    value: dict[str, Any] = {"code": code, "message": str(message)}
    if exception_type is not None:
        value["details"] = {"exception_type": exception_type}
    else:
        value["details"] = {}
    return value


def validate_sentiment_output(result: Any, review: str, expected: Mapping[str, Any]) -> dict[str, Any]:
    """Refuse wrong occurrence/labels/non-finite probabilities from ATSC.

    The raw model's softmax is not calibrated; no confidence threshold, label
    remapping, or probability adjustment is applied here.
    """
    if not isinstance(result, Mapping) or result.get("review") != review:
        raise ValueError("ATSC result must contain the unmodified original review")
    record = result.get("aspect")
    if not isinstance(record, Mapping) or any(record.get(key) != value for key, value in
        (("text", expected["text"]), ("start", expected["start"]), ("end", expected["end"]))):
        raise ValueError("ATSC result aspect text/offsets changed")
    label_id = result.get("label_id")
    if type(label_id) is not int or label_id not in atsc.ID2LABEL:
        raise ValueError("ATSC label_id must be exactly 0, 1 or 2")
    sentiment = result.get("sentiment")
    if sentiment != atsc.ID2LABEL[label_id]:
        raise ValueError("ATSC sentiment label disagrees with label_id")
    values = result.get("probabilities_softmax_uncalibrated")
    if not isinstance(values, Mapping) or set(values) != set(atsc.LABEL2ID):
        raise ValueError("ATSC probabilities must contain negative/neutral/positive")
    probs = [values[atsc.ID2LABEL[idx]] for idx in range(3)]
    if any(type(p) not in (int, float) or not math.isfinite(p) or not 0.0 <= p <= 1.0 for p in probs):
        raise ValueError("ATSC probabilities contain invalid/non-finite values")
    if abs(sum(probs) - 1.0) > 1e-4:
        raise ValueError("ATSC probabilities do not sum to one")
    winner = max(range(3), key=probs.__getitem__)
    if label_id != winner:
        raise ValueError("ATSC label_id does not match probability argmax")
    winning_probability = result.get("predicted_probability_uncalibrated")
    if (type(winning_probability) not in (int, float) or
            not math.isfinite(winning_probability) or
            abs(winning_probability - probs[label_id]) > 1e-5):
        raise ValueError("ATSC reported winner probability is inconsistent")
    return {"sentiment": sentiment, "label_id": label_id,
            "probabilities_softmax_uncalibrated": {atsc.ID2LABEL[i]: float(probs[i]) for i in range(3)},
            "predicted_probability_uncalibrated": float(winning_probability)}


def _base(review: str) -> dict[str, Any]:
    return {
        "phase": "4.3", "review": review, "review_modified": False,
        "offset_contract": OFFSET_CONTRACT,
        "max_atsc_tokens": adapter.REQUIRED_MAX_LENGTH,
        "overflow_policy": "per_aspect_structured_error_no_truncation",
        "aspects": [],
        "ate_extraction_performed": False,
        "atsc_predictions_performed": False,
        "training_performed": False,
        "gold_test_evaluation_performed": False,
        "weights_modified": False,
    }


def _global_error(review: str, code: str, exc: Exception) -> dict[str, Any]:
    report = _base(review)
    report["status"] = "ATE_ERROR" if code.startswith("ATE_") else "PIPELINE_ERROR"
    report["summary"] = {"extracted": 0, "classified": 0, "errors": 0, "global_errors": 1}
    report["error"] = _error(code, str(exc), exception_type=type(exc).__name__)
    return report


def _failed_item(item: Mapping[str, Any], error: Mapping[str, Any], *, keep_budget: bool = False) -> dict[str, Any]:
    r = {"index": item["index"], "status": "ERROR",
         "sentiment": None, "label_id": None,
         "probabilities_softmax_uncalibrated": None,
         "predicted_probability_uncalibrated": None,
         "error": dict(error)}
    for key in ("aspect", "source_aspect"):
        if key in item:
            r[key] = item[key]
    if keep_budget and "token_budget" in item:
        r["token_budget"] = item["token_budget"]
    elif "token_budget" not in item:
        r["token_budget"] = None
    return r


class ABSAInferencePipeline:
    """Hold one instance of each pretrained model for any number of reviews."""

    def __init__(self, extractor: Any, classifier: Any):
        if not callable(getattr(extractor, "extract", None)):
            raise TypeError("ATE extractor must implement extract(review)")
        if not callable(getattr(classifier, "predict", None)):
            raise TypeError("ATSC classifier must implement predict(review, aspect, start, end)")
        tokenizer = getattr(classifier, "tokenizer", None)
        max_length = getattr(classifier, "max_length", None)
        if tokenizer is None or max_length != adapter.REQUIRED_MAX_LENGTH or type(max_length) is not int:
            raise ValueError("ATSC classifier must expose saved tokenizer and trained max_length=128")
        atsc.validate_tokenizer(tokenizer, max_length)
        self.extractor = extractor
        self.classifier = classifier
        self.tokenizer = tokenizer
        self.max_length = max_length

    def predict(self, review: str) -> dict[str, Any]:
        """Run one review; preserve ATE ordering and every input aspect index.

        Top-level invalid review is a programmer input error. Model/adapter
        failures are returned as structured results. Each ATSC prediction error
        affects only its own aspect.
        """
        review = adapter.require_review(review)
        try:
            aspects = self.extractor.extract(review)
        except Exception as exc:
            code = "ATE_TOKEN_OVERFLOW" if (isinstance(exc, ValueError) and
                    "No input was truncated" in str(exc)) else "ATE_INFERENCE_FAILED"
            return _global_error(review, code, exc)
        if not isinstance(aspects, list):
            return _global_error(review, "ATE_OUTPUT_INVALID", TypeError("ATE extract() must return a list"))
        try:
            prepared = adapter.adapt_ate_output({"review": review, "aspects": aspects},
                                                self.tokenizer, max_length=self.max_length)
        except Exception as exc:
            report = _global_error(review, "ADAPTER_FAILED", exc)
            report["ate_extraction_performed"] = True
            report["summary"]["extracted"] = len(aspects)
            return report
        if prepared.get("review") != review or len(prepared.get("aspects", [])) != len(aspects):
            report = _global_error(review, "ADAPTER_OUTPUT_INVALID",
                                   ValueError("Adapter changed review or number of aspects"))
            report["ate_extraction_performed"] = True
            report["summary"]["extracted"] = len(aspects)
            return report

        rows: list[dict[str, Any]] = []
        classified = 0
        for item in prepared["aspects"]:
            if item["status"] == "ERROR":
                rows.append(_failed_item(item, item["error"]))
                continue
            if item["status"] != "READY_FOR_ATSC":
                rows.append(_failed_item(item, _error("ADAPTER_OUTPUT_INVALID", "Unexpected adapter status"),
                                         keep_budget=True))
                continue
            aspect = item["aspect"]
            try:
                prediction = self.classifier.predict(review, aspect["text"], aspect["start"], aspect["end"])
            except Exception as exc:
                rows.append(_failed_item(item, _error("ATSC_INFERENCE_FAILED", str(exc),
                                                   exception_type=type(exc).__name__), keep_budget=True))
                continue
            try:
                values = validate_sentiment_output(prediction, review, aspect)
            except (ValueError, TypeError, KeyError) as exc:
                rows.append(_failed_item(item, _error("ATSC_OUTPUT_INVALID", str(exc),
                                                   exception_type=type(exc).__name__), keep_budget=True))
                continue
            rows.append({"index": item["index"], "status": "CLASSIFIED", "aspect": aspect,
                         "token_budget": item["token_budget"], **values, "error": None})
            classified += 1
        errors = len(rows) - classified
        if not rows:
            status = "NO_ASPECTS"
        elif errors == 0:
            status = "COMPLETE"
        elif classified:
            status = "PARTIAL"
        else:
            status = "FAILED"
        report = _base(review)
        report.update({"status": status,
                       "summary": {"extracted": len(aspects), "classified": classified,
                                   "errors": errors, "global_errors": 0},
                       "aspects": rows,
                       "ate_extraction_performed": True,
                       "atsc_predictions_performed": bool(classified)})
        return report


def load_pipeline(*, ate_model_dir: str | Path = DEFAULT_ATE_MODEL,
                  atsc_model_dir: str | Path = DEFAULT_ATSC_MODEL,
                  preprocessing_dir: str | Path = DEFAULT_PREPROCESSING,
                  device: str = "cpu") -> ABSAInferencePipeline:
    """Load exactly one ATE model and one ATSC model, from local files only."""
    if device not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be cpu, cuda or auto")
    extractor = ate.load_extractor(model_dir=ate_model_dir, device=device, max_length=128)
    classifier = atsc.load_classifier(model_dir=atsc_model_dir,
                                      preprocessing_dir=preprocessing_dir, device=device)
    return ABSAInferencePipeline(extractor, classifier)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Step 4.3: inference-only ATE -> adapter -> ATSC; NO training/test/Streamlit")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--review", help="Exactly one raw English review")
    source.add_argument("--input-file", type=Path, help="UTF-8 raw review file; no normalization")
    parser.add_argument("--ate-model", default=DEFAULT_ATE_MODEL)
    parser.add_argument("--atsc-model", default=DEFAULT_ATSC_MODEL)
    parser.add_argument("--preprocessing", default=DEFAULT_PREPROCESSING)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, help="New JSON output file; refuses overwrite")
    parser.add_argument("--strict", action="store_true", help="Exit 2 when any aspect/global error; still emits JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.output is not None and args.output.exists():
        parser.error(f"Refusing overwrite: output exists: {args.output}")
    try:
        review = args.review if args.review is not None else args.input_file.read_text(encoding="utf-8-sig")
        adapter.require_review(review)
        pipeline = load_pipeline(ate_model_dir=args.ate_model, atsc_model_dir=args.atsc_model,
                                 preprocessing_dir=args.preprocessing, device=args.device)
        result = pipeline.predict(review)
        rendered = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as out:
                out.write(rendered)
        print(rendered, end="")
        return 2 if args.strict and (result["summary"]["errors"] or result["summary"]["global_errors"]) else 0
    except (ValueError, TypeError, FileNotFoundError, RuntimeError, OSError, UnicodeError, ImportError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
