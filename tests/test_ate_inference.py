"""Step 2.4 inference tests: deterministic synthetic logits; NO real model metrics.

All fake labels/scores in this file are test fixtures, NOT hardcoded inference
in ate_inference.py. No dataset loading, training, or remote model downloads.
"""
from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import torch

from src.ate_inference import (
    AspectExtractor, build_aspects, choose_device, first_subword_positions,
    load_extractor, main, model_label_check, predicted_bio_spans,
    require_local_model, split_words_with_offsets,
)


class StubEncoding(dict):
    def __init__(self, ids, word_ids):
        super().__init__(input_ids=[ids], attention_mask=[[1] * len(ids)])
        self._word_ids = word_ids

    def word_ids(self, batch_index=0):
        assert batch_index == 0
        return self._word_ids


class StubFastTokenizer:
    is_fast = True
    padding_side = "right"
    model_max_length = 512

    def __init__(self):
        self.calls = []

    def __call__(self, words_batch, **kwargs):
        assert len(words_batch) == 1 and isinstance(words_batch[0], list)
        self.calls.append(kwargs)
        words = words_batch[0]
        ids, word_ids = [101], [None]
        for i, word in enumerate(words):
            # Force one subword continuation for alignment tests.
            pieces = 2 if word.lower() == "unaffordable" else 1
            for p in range(pieces):
                ids.append(1000 + 4 * i + p)
                word_ids.append(i)
        ids.append(102)
        word_ids.append(None)
        return StubEncoding(ids, word_ids)


class StubModel(torch.nn.Module):
    def __init__(self, predictions: list[int], *, labels=None, score=5.0):
        super().__init__()
        self.config = types.SimpleNamespace(
            num_labels=3,
            id2label=labels if labels is not None else {0: "O", 1: "B-Term", 2: "I-Term"},
            max_position_embeddings=512,
        )
        self.predictions = predictions
        self.score = score
        self.calls = 0
        self.forward_training_states = []
        self.kwarg_keys = []

    def forward(self, **kwargs):
        self.calls += 1
        self.forward_training_states.append((self.training, torch.is_inference_mode_enabled()))
        self.kwarg_keys.append(set(kwargs))
        input_ids = kwargs["input_ids"]
        assert len(input_ids.shape) == 2 and input_ids.shape[0] == 1
        length = input_ids.shape[1]
        if len(self.predictions) != length:
            raise AssertionError(f"StubModel expected {len(self.predictions)} positions, got {length}")
        logits = torch.full((1, length, 3), -5.0, dtype=torch.float32)
        for pos, tag in enumerate(self.predictions):
            logits[0, pos, tag] = self.score
        return types.SimpleNamespace(logits=logits)


def extractor_for(words: list[str], tags_by_word: list[int], *, subword_override=None, max_length=128):
    assert len(words) == len(tags_by_word)
    tokenizer = StubFastTokenizer()
    encoding = tokenizer([words], is_split_into_words=True)
    positions = first_subword_positions(encoding.word_ids(), len(words))
    preds = [0] * len(encoding["input_ids"][0])
    for i, pos in enumerate(positions):
        preds[pos] = tags_by_word[i]
    if subword_override is not None:
        preds[subword_override[0]] = subword_override[1]
    model = StubModel(preds)
    return AspectExtractor(model, tokenizer, device="cpu", max_length=max_length), model


class TestATEInference(unittest.TestCase):
    def test_offsets_retained_with_repeated_spaces(self):
        text = "  Food,   service!  "
        words, offsets = split_words_with_offsets(text)
        self.assertEqual(words, ["Food", ",", "service", "!"])
        self.assertEqual([text[s:e] for s, e in offsets], words)
        self.assertEqual(offsets, [(2, 6), (6, 7), (10, 17), (17, 18)])

    def test_offsets_for_unicode_punctuation_and_apostrophe(self):
        text = "It’s amazing — café!"
        words, offsets = split_words_with_offsets(text)
        self.assertEqual(words, ["It’s", "amazing", "—", "café", "!"])
        self.assertEqual([text[s:e] for s, e in offsets], words)

    def test_invalid_input_type_rejected(self):
        with self.assertRaises(TypeError):
            split_words_with_offsets(None)

    def test_bio_policy_matches_step22(self):
        from src.ate_train import bio_entity_spans
        for sequence in ([0, 1, 2, 0], [2, 2, 0], [1, 1, 2], [0, 0], [2, 1, 2, 0, 2]):
            self.assertEqual(predicted_bio_spans(sequence), sorted(bio_entity_spans(sequence, strict_gold=False)))

    def test_illegal_bio_id_rejected(self):
        with self.assertRaisesRegex(ValueError, "Invalid BIO"):
            predicted_bio_spans([1, 3])

    def test_char_span_is_exact_original_substring(self):
        review = "Great  customer service!"
        words, offsets = split_words_with_offsets(review)
        self.assertEqual(words, ["Great", "customer", "service", "!"])
        aspects = build_aspects(review, offsets, [0, 1, 2, 0], [0.9, 0.8, 0.6, 0.4])
        self.assertEqual(aspects, [{"text": "customer service", "start": 7, "end": 23, "confidence": 0.7}])
        self.assertEqual(review[aspects[0]["start"]:aspects[0]["end"]], aspects[0]["text"])

    def test_confidence_mean_is_not_multiplied(self):
        out = build_aspects("a b c", [(0, 1), (2, 3), (4, 5)], [1, 2, 0], [0.4, 0.8, 0.9])
        self.assertEqual(out[0]["confidence"], 0.6)

    def test_confidence_length_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError, "lengths disagree"):
            build_aspects("a", [(0, 1)], [1], [])

    def test_first_wordpiece_aligns_only_first_of_two_subwords(self):
        self.assertEqual(first_subword_positions([None, 0, 0, 1, None], 2), [1, 3])

    def test_missing_wordpiece_rejected(self):
        with self.assertRaisesRegex(ValueError, "lost or reordered"):
            first_subword_positions([None, 0, None], 2)

    def test_out_of_order_wordpiece_rejected(self):
        with self.assertRaisesRegex(ValueError, "lost or reordered"):
            first_subword_positions([None, 1, 0, None], 2)

    def test_model_config_requires_exact_bio_mapping(self):
        with self.assertRaisesRegex(ValueError, "do not match"):
            model_label_check(types.SimpleNamespace(num_labels=3, id2label={0: "O", 1: "NEG", 2: "POS"}))
        with self.assertRaisesRegex(ValueError, "three BIO"):
            model_label_check(types.SimpleNamespace(num_labels=2, id2label={0: "O", 1: "B-Term"}))

    def test_pdf_example_uses_fake_logits_not_predictions(self):
        # Predicted labels are explicitly synthetic fixture; not learned.
        text = "The food was delicious, but the service was terrible."
        words, _ = split_words_with_offsets(text)
        tags = [1 if w.lower() in {"food", "service"} else 0 for w in words]
        extractor, model = extractor_for(words, tags)
        result = extractor.extract(text)
        self.assertEqual([a["text"] for a in result], ["food", "service"])
        for a in result:
            self.assertEqual(text[a["start"]:a["end"]], a["text"])
            self.assertGreater(a["confidence"], 0.98)
        self.assertEqual(model.calls, 1)
        self.assertEqual(model.kwarg_keys, [{"input_ids", "attention_mask"}])
        self.assertEqual(model.forward_training_states, [(False, True)])

    def test_subword_continuation_is_not_new_entity(self):
        text = "unaffordable service"
        words, _ = split_words_with_offsets(text)
        extractor, _ = extractor_for(words, [1, 0], subword_override=(2, 1))
        result = extractor.extract(text)
        self.assertEqual([a["text"] for a in result], ["unaffordable"])

    def test_predicted_stray_i_starts_span(self):
        text = "food service"
        words, _ = split_words_with_offsets(text)
        extractor, _ = extractor_for(words, [2, 2])
        self.assertEqual([a["text"] for a in extractor.extract(text)], ["food service"])

    def test_multiple_identical_aspects_keep_distinct_offsets(self):
        text = "food, food!"
        words, _ = split_words_with_offsets(text)
        extractor, _ = extractor_for(words, [1, 0, 1, 0])
        got = extractor.extract(text)
        self.assertEqual([(a["start"], a["end"]) for a in got], [(0, 4), (6, 10)])

    def test_no_aspects_is_empty_list(self):
        text = "Everything was fine."
        words, _ = split_words_with_offsets(text)
        extractor, _ = extractor_for(words, [0] * len(words))
        self.assertEqual(extractor.extract(text), [])

    def test_empty_review_is_explicit_error(self):
        extractor, model = extractor_for(["hello"], [0])
        with self.assertRaisesRegex(ValueError, "non-whitespace"):
            extractor.extract(" \t \n ")
        self.assertEqual(model.calls, 0)

    def test_overlong_review_is_refused_without_truncation(self):
        text = "nice restaurant food"
        words, _ = split_words_with_offsets(text)
        extractor, model = extractor_for(words, [0, 1, 0], max_length=4)
        with self.assertRaisesRegex(ValueError, "No input was truncated"):
            extractor.extract(text)
        self.assertEqual(model.calls, 0)

    def test_max_length_model_capacity_validation(self):
        tok = StubFastTokenizer()
        model = StubModel([0, 0, 0])
        with self.assertRaisesRegex(ValueError, "model capacity"):
            AspectExtractor(model, tok, max_length=600)

    def test_cuda_unavailable_is_rejected(self):
        fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
        self.assertEqual(choose_device("auto", fake_torch), "cpu")
        self.assertEqual(choose_device("cpu", fake_torch), "cpu")
        with self.assertRaisesRegex(ValueError, "CUDA unavailable"):
            choose_device("cuda", fake_torch)

    def test_nonfast_tokenizer_rejected(self):
        tok = StubFastTokenizer()
        tok.is_fast = False
        with self.assertRaisesRegex(ValueError, "fast tokenizer"):
            AspectExtractor(StubModel([0, 0]), tok)

    def test_local_checkpoint_missing_is_explicit_error(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(FileNotFoundError):
                require_local_model(Path(td))

    def test_local_model_loader_uses_local_files_only(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors"):
                (root / name).write_bytes(b"fixture")
            text = "food"
            tok = StubFastTokenizer()
            model = StubModel([0, 1, 0])
            transformers = types.SimpleNamespace(
                AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a, **kw: (
                    self.assertTrue(kw["local_files_only"]) or self.assertTrue(kw["use_fast"]) or tok
                )),
                AutoModelForTokenClassification=types.SimpleNamespace(from_pretrained=lambda *a, **kw: (
                    self.assertTrue(kw["local_files_only"]) or model
                )),
            )
            with patch.dict("sys.modules", {"transformers": transformers}):
                extractor = load_extractor(model_dir=root, device="cpu")
            self.assertEqual(extractor.extract(text)[0]["text"], "food")

    def test_cli_inference_e2e_fake_model(self):
        text = "The food was great."
        words, _ = split_words_with_offsets(text)
        extractor, fake_model = extractor_for(words, [1 if w == "food" else 0 for w in words])
        with tempfile.TemporaryDirectory() as td:
            output = Path(td) / "results.json"
            stdout = io.StringIO()
            with patch(f"{main.__module__}.load_extractor", return_value=extractor) as loader, redirect_stdout(stdout):
                rc = main(["--text", text, "--output", str(output), "--device", "cpu"])
            result = json.loads(stdout.getvalue())
            self.assertEqual(rc, 0)
            self.assertEqual(result, json.loads(output.read_text(encoding="utf-8")))
            self.assertEqual([x["text"] for x in result["aspects"]], ["food"])
            self.assertEqual(fake_model.calls, 1)
            loader.assert_called_once()
            with patch(f"{main.__module__}.load_extractor", return_value=extractor):
                with redirect_stdout(io.StringIO()), patch("sys.stderr", new_callable=io.StringIO):
                    with self.assertRaises(SystemExit):
                        main(["--text", text, "--output", str(output)])

    def test_cli_utf8_file_input_works(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "review.txt"
            text = "The  café service is great."
            path.write_text(text, encoding="utf-8")
            words, _ = split_words_with_offsets(text)
            extractor, _ = extractor_for(words, [1 if w == "service" else 0 for w in words])
            stdout = io.StringIO()
            with patch(f"{main.__module__}.load_extractor", return_value=extractor), redirect_stdout(stdout):
                main(["--input-file", str(path)])
            result = json.loads(stdout.getvalue())
            self.assertEqual([x["text"] for x in result["aspects"]], ["service"])

    def test_no_dataset_or_training_import_from_cli(self):
        # This is structural: the module does not depend on Trainer/datasets.
        import inspect
        import src.ate_inference as ate_inference
        src = inspect.getsource(ate_inference)
        self.assertNotIn("from datasets import", src)
        self.assertNotIn("from transformers import Trainer", src)
        self.assertNotIn(".train()", src)


if __name__ == "__main__":
    unittest.main()
