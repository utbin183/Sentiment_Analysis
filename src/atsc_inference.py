"""Step 3.5 — local, inference-only ATSC for a *specified aspect occurrence*.

Inputs: original review string, exact aspect substring, Python character [start,end).
Encoding MUST match Step 3.2: ([unused1] target [unused2] marked review, aspect).
Loads only the locally saved Step 3.3 best_model and Step 3.2 manifest; it
never loads train/validation/test splits, never trains and never saves weights.
"""
from __future__ import annotations

# Set before any Transformers import (Windows Anaconda with TensorFlow/Keras 3).
import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

# Reuse exactly the target-marking function from Step 3.2, not a reimplementation.
from atsc_preprocessing import (
    mark_target_span,
    TARGET_START_MARKER,
    TARGET_END_MARKER,
)

MODEL_NAME = "distilbert/distilbert-base-uncased"
DEFAULT_MODEL = "artifacts/atsc_training/best_model"
DEFAULT_DATA = "artifacts/atsc_preprocessing"
ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}
MARKERS = {TARGET_START_MARKER: 2, TARGET_END_MARKER: 3}
VOCAB_SIZE = 30522


def validate_span(review: str, aspect: str, start: int, end: int) -> None:
    """Exact, case-sensitive, half-open Python string offsets (not bytes)."""
    if not isinstance(review, str) or not review.strip():
        raise ValueError("review must be a non-empty string")
    if not isinstance(aspect, str) or not aspect.strip():
        raise ValueError("aspect must be a non-empty string")
    if type(start) is not int or type(end) is not int:
        raise ValueError("start/end must be integers (not booleans)")
    if not (0 <= start < end <= len(review)):
        raise ValueError(f"Invalid half-open aspect offsets [{start}, {end}) for review length {len(review)}")
    if review[start:end] != aspect:
        raise ValueError(
            f"aspect does not exactly match review[start:end]={review[start:end]!r}; "
            "use offsets for the specific occurrence (end is exclusive)"
        )
    # The shared Step 3.2 helper separately rejects marker collisions.


def read_preprocessing_manifest(data_dir: str | Path) -> int:
    path = Path(data_dir) / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing Step 3.2 preprocessing manifest: {path}")
    info = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(info, Mapping) or info.get("status") != "PREPROCESSING FINISHED":
        raise ValueError("Step 3.2 preprocessing manifest is not marked finished")
    if info.get("tokenizer_name") != MODEL_NAME or info.get("vocabulary_resized") is not False:
        raise ValueError("Unexpected tokenizer or resized vocabulary; cannot reproduce Step 3.2 inputs")
    if info.get("target_markers") != MARKERS or info.get("target_disambiguation") != "span_markers_from_gold_start_end_offsets":
        raise ValueError("Missing/incorrect Step 3.2 aspect occurrence markers")
    if info.get("input_format") != "[CLS] review with [unused1] target [unused2] [SEP] aspect [SEP]":
        raise ValueError("ATSC input format differs from training")
    if info.get("label_mapping") != {str(k): v for k, v in ID2LABEL.items()}:
        raise ValueError("ATSC class mapping differs from training")
    if info.get("truncation") is not False or info.get("gold_pairs_only") is not True:
        raise ValueError("Step 3.2 preprocessing integrity flags are invalid")
    max_length = info.get("max_length")
    if type(max_length) is not int or not (8 <= max_length <= 512):
        raise ValueError("Invalid preprocessing max_length")
    return max_length


def validate_model_config(config: Any) -> None:
    if getattr(config, "model_type", None) != "distilbert" or getattr(config, "num_labels", None) != 3:
        raise ValueError("Expected a DistilBERT 3-class ATSC checkpoint")
    try:
        observed = {int(k): v for k, v in config.id2label.items()}
    except (TypeError, AttributeError, ValueError) as exc:
        raise ValueError("Checkpoint has invalid id2label mapping") from exc
    if observed != ID2LABEL:
        raise ValueError(f"Checkpoint labels {observed!r} do not match {ID2LABEL!r}")
    observed_reverse = getattr(config, "label2id", None)
    if observed_reverse != LABEL2ID:
        raise ValueError("Checkpoint label2id mapping differs from Step 3.3")
    vocab_size = getattr(config, "vocab_size", None)
    if vocab_size != VOCAB_SIZE:
        raise ValueError("Checkpoint vocabulary differs from DistilBERT base")


def validate_tokenizer(tokenizer: Any, max_length: int) -> None:
    if getattr(tokenizer, "is_fast", False) is not True:
        raise ValueError("Step 3.2 uses a fast DistilBERT tokenizer")
    if len(tokenizer) != VOCAB_SIZE:
        raise ValueError("Tokenizer vocabulary was resized or is incompatible")
    if getattr(tokenizer, "pad_token_id", None) is None:
        raise ValueError("Tokenizer lacks pad token")
    if getattr(tokenizer, "padding_side", None) != "right":
        raise ValueError("Tokenizer must use right padding")
    capacity = getattr(tokenizer, "model_max_length", None)
    if not isinstance(capacity, int) or max_length > capacity:
        raise ValueError("Saved tokenizer cannot represent max_length")
    if tokenizer.num_special_tokens_to_add(pair=True) != 3:
        raise ValueError("Expected BERT sequence pair: [CLS] review [SEP] aspect [SEP]")
    if not all(marker in tokenizer.additional_special_tokens for marker in MARKERS):
        raise ValueError("The two existing [unused] markers are not registered as special tokens")
    for marker, idx in MARKERS.items():
        if tokenizer.convert_tokens_to_ids(marker) != idx:
            raise ValueError(f"Marker {marker} must have pretrained ID {idx}")


def choose_device(device: str, torch_module: Any) -> str:
    if device not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be cpu, cuda, or auto")
    if device == "cpu":
        return "cpu"
    available = bool(torch_module.cuda.is_available())
    if device == "cuda" and not available:
        raise ValueError("CUDA not available; use --device cpu")
    return "cuda" if available else "cpu"


def check_local_model_dir(path: Path) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"ATSC best_model directory missing: {path}")
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors"):
        if not (path / name).is_file():
            raise FileNotFoundError(f"Missing saved ATSC model artifact: {path / name}")


def encode_pair(tokenizer: Any, review: str, aspect: str, start: int, end: int,
                *, max_length: int) -> dict[str, list[int]]:
    """Exact Step 3.2 encoding with no truncation and no padding."""
    validate_span(review, aspect, start, end)
    marked_review = mark_target_span(review, start, end)
    encoded = tokenizer(marked_review, aspect, add_special_tokens=True,
                        padding=False, truncation=False, return_attention_mask=True)
    ids, mask = encoded["input_ids"], encoded["attention_mask"]
    if (not isinstance(ids, list) or not isinstance(mask, list) or
            len(ids) != len(mask) or len(ids) < 8):
        raise ValueError("Malformed tokenizer output: input_ids/attention_mask")
    if any(type(t) is not int or not 0 <= t < VOCAB_SIZE for t in ids):
        raise ValueError("Tokenizer produced an invalid vocabulary ID")
    if any(type(m) is not int or m != 1 for m in mask):
        raise ValueError("Unpadded inputs must have all-one attention_mask")
    a, b = MARKERS.values()
    if ids.count(a) != 1 or ids.count(b) != 1 or ids.index(a) >= ids.index(b):
        raise ValueError("Exactly one ordered pair of aspect markers must survive tokenization")
    sep = tokenizer.sep_token_id
    if (ids[0] != tokenizer.cls_token_id or ids[-1] != sep or
            ids.count(sep) != 2 or ids.index(b) >= ids.index(sep)):
        raise ValueError("Expected [CLS] marked-review [SEP] aspect [SEP] in this order")
    if type(max_length) is not int or not 8 <= max_length <= 512:
        raise ValueError("max_length must be 8..512")
    if len(ids) > max_length:
        raise ValueError(
            f"ATSC pair needs {len(ids)} tokens, but Step 3.2 limit is {max_length}; "
            "input was NOT truncated. Provide a shorter review (with correct offsets)."
        )
    return {"input_ids": ids, "attention_mask": mask}


def validate_probabilities(values: Any) -> tuple[int, dict[str, float]]:
    """Validate softmax vector, retain full float precision, never claim calibration."""
    if not isinstance(values, (list, tuple)) or len(values) != 3:
        raise ValueError("Expected exactly three sentiment probabilities")
    if any(type(p) not in (float, int) or not math.isfinite(p) or not 0 <= p <= 1 for p in values):
        raise ValueError("Invalid / non-finite softmax probabilities")
    if abs(sum(values) - 1) > 1e-4:
        raise ValueError("Softmax probabilities do not sum to one")
    winner = max(range(3), key=lambda i: values[i])
    return winner, {ID2LABEL[i]: float(values[i]) for i in range(3)}


class ATSCClassifier:
    """Reuse one loaded model to classify multiple manually specified aspects."""

    def __init__(self, model: Any, tokenizer: Any, *, max_length: int, device: str = "cpu", torch_module: Any = None):
        if torch_module is None:
            import torch as torch_module
        validate_model_config(model.config)
        validate_tokenizer(tokenizer, max_length)
        if model.get_input_embeddings().num_embeddings != len(tokenizer):
            raise ValueError("Model embeddings and tokenizer vocabulary differ")
        self.device = choose_device(device, torch_module)
        self.model = model.to(self.device)
        self.model.eval()
        if self.model.training:
            raise RuntimeError("ATSC model cannot be set to evaluation mode")
        self.torch = torch_module
        self.tokenizer = tokenizer
        self.max_length = max_length

    def predict(self, review: str, aspect: str, start: int, end: int) -> dict[str, Any]:
        """One aspect occurrence, 0-based [start,end) relative to *this* review."""
        sample = encode_pair(self.tokenizer, review, aspect, start, end, max_length=self.max_length)
        torch = self.torch
        inputs = {
            "input_ids": torch.tensor([sample["input_ids"]], dtype=torch.long, device=self.device),
            "attention_mask": torch.tensor([sample["attention_mask"]], dtype=torch.long, device=self.device),
        }
        self.model.eval()
        with torch.inference_mode():
            logits = self.model(**inputs).logits
            if tuple(logits.shape) != (1, 3) or not bool(torch.isfinite(logits).all().item()):
                raise ValueError("ATSC model returned invalid logits (expected [1,3] finite)")
            probabilities = torch.softmax(logits.float(), dim=-1)[0].cpu().tolist()
        label_id, probabilities_by_name = validate_probabilities(probabilities)
        return {
            "review": review,
            "aspect": {"text": aspect, "start": start, "end": end},
            "sentiment": ID2LABEL[label_id],
            "label_id": label_id,
            "probabilities_softmax_uncalibrated": probabilities_by_name,
            "predicted_probability_uncalibrated": probabilities_by_name[ID2LABEL[label_id]],
        }


def load_classifier(*, model_dir: str | Path = DEFAULT_MODEL,
                    preprocessing_dir: str | Path = DEFAULT_DATA,
                    device: str = "cpu") -> ATSCClassifier:
    """Load *local* best_model once; never access gold/test data or train."""
    max_length = read_preprocessing_manifest(preprocessing_dir)
    path = Path(model_dir)
    check_local_model_dir(path)
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    resolved_device = choose_device(device, torch)
    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True, use_fast=True)
    # Validate tokenizer/labels before touching potentially large weight file.
    validate_tokenizer(tokenizer, max_length)
    try:
        model = AutoModelForSequenceClassification.from_pretrained(str(path), local_files_only=True)
    except Exception as exc:
        raise RuntimeError(
            f"Cannot load ATSC weights from {path}: {exc}. "
            "Check model.safetensors validity; do not retrain or replace the selected checkpoint "
            "as a side effect of inference."
        ) from exc
    return ATSCClassifier(model, tokenizer, device=resolved_device,
                          max_length=max_length, torch_module=torch)


def predict_sentiment(review: str, aspect: str, start: int, end: int, *,
                      model_dir: str | Path = DEFAULT_MODEL,
                      preprocessing_dir: str | Path = DEFAULT_DATA,
                      device: str = "cpu") -> dict[str, Any]:
    """Convenience API. For many aspects, load_classifier() once instead."""
    validate_span(review, aspect, start, end)
    return load_classifier(model_dir=model_dir, preprocessing_dir=preprocessing_dir,
                           device=device).predict(review, aspect, start, end)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Step 3.5: offline ATSC inference on ONE specified aspect occurrence; NO training/test/ATE/Streamlit")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--review", help="Literal English review (Python character offsets)")
    source.add_argument("--input-file", type=Path, help="UTF-8 file containing the literal review")
    parser.add_argument("--aspect", required=True, help="Exact aspect text, case-sensitive")
    parser.add_argument("--start", type=int, required=True, help="Zero-based inclusive character start")
    parser.add_argument("--end", type=int, required=True, help="Zero-based exclusive character end")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Local Step 3.3 best_model folder")
    parser.add_argument("--preprocessing", default=DEFAULT_DATA, help="Step 3.2 folder, manifest ONLY")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, help="Optional new JSON file; refuses overwrite")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.output is not None and args.output.exists():
        parser.error(f"Output already exists, refusing overwrite: {args.output}")
    try:
        review = args.review if args.review is not None else args.input_file.read_text(encoding="utf-8-sig")
        validate_span(review, args.aspect, args.start, args.end)
        classifier = load_classifier(model_dir=args.model, preprocessing_dir=args.preprocessing, device=args.device)
        result = classifier.predict(review, args.aspect, args.start, args.end)
        rendered = json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered, encoding="utf-8")
        print(rendered, end="")
        return 0
    except (ValueError, TypeError, RuntimeError, FileNotFoundError, UnicodeError, OSError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
