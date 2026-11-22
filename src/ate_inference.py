"""Step 2.4 — ATE inference on fresh English reviews; NO training.

Loads the Step 2.2 locally saved DistilBERT token-classification model.
BIO decoding matches Step 2.2/2.3: first subword per original word; a stray
predicted I-Term starts an entity. Character offsets are [start, end) in the
exact input review, with no normalization of returned aspect text.

No dataset, gold labels, test split, Trainer, or optimizer is used.
"""
from __future__ import annotations

# Prevent installed TensorFlow / Keras 3 from interfering with PyTorch-only
# Transformers imports. These must be set before importing transformers.
import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

import argparse
import json
from pathlib import Path
import re
from typing import Any, Sequence

ID2LABEL = {0: "O", 1: "B-Term", 2: "I-Term"}
DEFAULT_MODEL = "artifacts/ate_training/best_model"
DEFAULT_MAX_LENGTH = 128  # matches Step 2.1 preprocessing, NOT a learned cutoff

# Keep punctuation separate and preserve original character boundaries. The
# training source supplies separate 'Tokens'; raw input requires this heuristic.
# In particular, don't normalize whitespace or reconstruct aspect text by join.
WORD_PATTERN = re.compile(r"\w+(?:['\u2019]\w+)*|[^\w\s]", re.UNICODE)


def split_words_with_offsets(text: str) -> tuple[list[str], list[tuple[int, int]]]:
    """Find words/punctuation without changing text or character positions."""
    if not isinstance(text, str):
        raise TypeError("Review must be a string")
    matches = list(WORD_PATTERN.finditer(text))
    return [m.group(0) for m in matches], [m.span() for m in matches]


def predicted_bio_spans(tags: Sequence[int]) -> list[tuple[int, int]]:
    """Decode BIO as in ate_train.bio_entity_spans(strict_gold=False).

    A predicted I-Term with no active B-Term begins a new span; a new B-Term
    closes the previous span. Return sorted [start_word, end_word) ranges.
    """
    spans: list[tuple[int, int]] = []
    start = None
    for index, tag in enumerate([*tags, 0]):
        if tag not in ID2LABEL:
            raise ValueError(f"Invalid BIO prediction id: {tag}")
        if tag == 0:
            if start is not None:
                spans.append((start, index))
                start = None
        elif tag == 1:
            if start is not None:
                spans.append((start, index))
            start = index
        elif start is None:  # a stray predicted I-Term
            start = index
    return spans


def model_label_check(config: Any) -> None:
    if getattr(config, "num_labels", None) != len(ID2LABEL):
        raise ValueError("Loaded checkpoint must have exactly three BIO labels")
    raw_labels = getattr(config, "id2label", {})
    try:
        observed = {int(key): value for key, value in raw_labels.items()}
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError("Checkpoint has invalid id2label mapping") from exc
    if observed != ID2LABEL:
        raise ValueError(f"Checkpoint labels {observed!r} do not match {ID2LABEL!r}")


def choose_device(requested: str, torch_module: Any) -> str:
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    if requested == "cpu":
        return "cpu"
    available = bool(torch_module.cuda.is_available())
    if requested == "cuda" and not available:
        raise ValueError("CUDA unavailable; choose --device cpu")
    return "cuda" if requested == "cuda" or (requested == "auto" and available) else "cpu"


def first_subword_positions(word_ids: Sequence[int | None], num_words: int) -> list[int]:
    """Require one or more wordpieces for every supplied source word."""
    positions: list[int] = []
    previous_word = None
    encountered: list[int] = []
    for position, word_id in enumerate(word_ids):
        if word_id is not None:
            if not isinstance(word_id, int) or not (0 <= word_id < num_words):
                raise ValueError("Tokenizer returned an invalid word ID")
            if word_id != previous_word:
                positions.append(position)
                encountered.append(word_id)
        previous_word = word_id
    if encountered != list(range(num_words)):
        raise ValueError("Tokenizer lost or reordered words; refusing incomplete inference")
    return positions


def build_aspects(review: str, offsets: Sequence[tuple[int, int]],
                  tags: Sequence[int], word_confidence: Sequence[float]) -> list[dict[str, Any]]:
    """Convert predicted word BIO + scores to literal string spans.

    Confidence = mean softmax probability of the *predicted* BIO class across
    first wordpieces in the span; not a calibrated aspect correctness score.
    """
    if len(offsets) != len(tags) or len(tags) != len(word_confidence):
        raise ValueError("Word offsets, predicted tags and confidence lengths disagree")
    aspects: list[dict[str, Any]] = []
    for start_word, end_word in predicted_bio_spans(tags):
        start = offsets[start_word][0]
        end = offsets[end_word - 1][1]
        if not (0 <= start < end <= len(review)):
            raise ValueError("Invalid character offset from tokenizer word mapping")
        confidence = sum(word_confidence[start_word:end_word]) / (end_word - start_word)
        if not (0.0 <= confidence <= 1.0):
            raise ValueError("Invalid model probability")
        aspects.append({
            "text": review[start:end],
            "start": start,
            "end": end,
            "confidence": round(float(confidence), 6),
        })
    return aspects


class AspectExtractor:
    """Reusable, loaded-once local ATE inference (no training code)."""

    def __init__(self, model: Any, tokenizer: Any, *, device: str = "cpu",
                 max_length: int = DEFAULT_MAX_LENGTH, torch_module: Any = None):
        if not getattr(tokenizer, "is_fast", False):
            raise ValueError("ATE requires a Hugging Face fast tokenizer with word_ids()")
        if getattr(tokenizer, "padding_side", "right") != "right":
            raise ValueError("Expected right-padding tokenizer from Step 2.2")
        if not isinstance(max_length, int) or max_length < 3:
            raise ValueError("max_length must be >= 3")
        model_limit = getattr(model.config, "max_position_embeddings", None)
        if isinstance(model_limit, int) and max_length > model_limit:
            raise ValueError(f"max_length {max_length} exceeds model capacity {model_limit}")
        tokenizer_limit = getattr(tokenizer, "model_max_length", None)
        if isinstance(tokenizer_limit, int) and max_length > tokenizer_limit:
            raise ValueError(f"max_length {max_length} exceeds tokenizer capacity {tokenizer_limit}")
        model_label_check(model.config)
        if torch_module is None:
            import torch as torch_module
        self.torch = torch_module
        self.device = choose_device(device, torch_module)
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.model = model.to(self.device)
        self.model.eval()

    def extract(self, review: str) -> list[dict[str, Any]]:
        words, offsets = split_words_with_offsets(review)
        if not words:
            raise ValueError("Review must contain at least one non-whitespace character")
        # As in Step 2.1: pretokenized source words -> WordPiece alignment.
        # We do NOT truncate: long inputs must fail explicitly instead of
        # dropping aspects. This is a conscious, documented limitation.
        encoded = self.tokenizer(
            [words], is_split_into_words=True, truncation=False,
            padding=False, return_attention_mask=True, add_special_tokens=True,
        )
        ids = encoded["input_ids"][0]
        attention_mask = encoded["attention_mask"][0]
        if len(ids) != len(attention_mask) or not ids:
            raise ValueError("Tokenizer returned malformed model input")
        if len(ids) > self.max_length:
            raise ValueError(
                f"Review requires {len(ids)} model tokens (limit {self.max_length}); "
                "split into shorter reviews, or explicitly increase --max-length "
                "within the model's position limit. No input was truncated."
            )
        word_ids = encoded.word_ids(batch_index=0)
        if len(word_ids) != len(ids):
            raise ValueError("Tokenizer word_ids and input_ids lengths disagree")
        first_positions = first_subword_positions(word_ids, len(words))
        torch = self.torch
        inputs = {
            "input_ids": torch.tensor([ids], dtype=torch.long, device=self.device),
            "attention_mask": torch.tensor([attention_mask], dtype=torch.long, device=self.device),
        }
        # Reaffirm eval mode: a third-party caller might have toggled it.
        self.model.eval()
        with torch.inference_mode():
            logits = self.model(**inputs).logits
            if tuple(logits.shape) != (1, len(ids), len(ID2LABEL)):
                raise ValueError(f"Model returned unexpected logits shape: {tuple(logits.shape)}")
            if not bool(torch.isfinite(logits).all()):
                raise ValueError("Model returned non-finite logits")
            probs = torch.softmax(logits.float(), dim=-1)[0]
            predicted_token_ids = probs.argmax(dim=-1)
            tags = [int(predicted_token_ids[position].item()) for position in first_positions]
            confidences = [float(probs[position, tags[index]].item())
                           for index, position in enumerate(first_positions)]
        return build_aspects(review, offsets, tags, confidences)


def require_local_model(model_dir: Path) -> None:
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Missing trained model directory: {model_dir}")
    for filename in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        if not (model_dir / filename).is_file():
            raise FileNotFoundError(f"Missing saved model/tokenizer artifact: {model_dir / filename}")
    if not any((model_dir / filename).is_file() for filename in ("model.safetensors", "pytorch_model.bin")):
        raise FileNotFoundError(f"Missing local model weights in {model_dir}")


def load_extractor(*, model_dir: str | Path = DEFAULT_MODEL, device: str = "auto",
                   max_length: int = DEFAULT_MAX_LENGTH) -> AspectExtractor:
    """Offline model load, once; never load source training or test data."""
    path = Path(model_dir)
    require_local_model(path)
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer
    device = choose_device(device, torch)
    tokenizer = AutoTokenizer.from_pretrained(str(path), use_fast=True, local_files_only=True)
    model = AutoModelForTokenClassification.from_pretrained(str(path), local_files_only=True)
    return AspectExtractor(model=model, tokenizer=tokenizer, device=device,
                           max_length=max_length, torch_module=torch)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Step 2.4 ATE inference: saved model only; NO training, ATSC or Streamlit")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", help="English review as a command-line string")
    source.add_argument("--input-file", type=Path, help="UTF-8 text file containing one review")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Local Step 2.2 best_model directory")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH,
                        help="Maximum input tokens including special tokens (default matches Step 2.1: 128)")
    parser.add_argument("--output", type=Path, help="Optional JSON report file (refuses overwrite)")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.output is not None and args.output.exists():
        parser.error(f"Output already exists: {args.output}. Choose a new output path.")
    try:
        review = args.text if args.text is not None else args.input_file.read_text(encoding="utf-8-sig")
        extractor = load_extractor(model_dir=args.model, device=args.device, max_length=args.max_length)
        aspects = extractor.extract(review)
    except (FileNotFoundError, ValueError, UnicodeError) as exc:
        parser.error(str(exc))
    result = {"review": review, "aspects": aspects}
    report_json = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report_json, encoding="utf-8")
    print(report_json, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
