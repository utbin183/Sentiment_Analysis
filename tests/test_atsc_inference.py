"""Step 3.5 tests: synthetic text, fake tokenizer and fake Torch module.

Never accesses the held-out test split or the real ATSC model weights.
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import torch

import atsc_inference as inf
from atsc_preprocessing import mark_target_span


class FakeTokenizer:
    is_fast = True
    pad_token_id = 0
    padding_side = "right"
    model_max_length = 512
    cls_token_id = 101
    sep_token_id = 102
    additional_special_tokens = ["[unused1]", "[unused2]"]

    def __init__(self):
        self.vocab = {"[unused1]": 2, "[unused2]": 3}
        self.last_args = None
        self.forced_ids = None

    def __len__(self):
        return 30522

    def num_special_tokens_to_add(self, pair=False):
        return 3 if pair else 2

    def convert_tokens_to_ids(self, t):
        return self.vocab.get(t, 100)

    def __call__(self, a, b, **kwargs):
        self.last_args = (a, b, kwargs)
        if self.forced_ids is not None:
            ids = list(self.forced_ids)
        else:
            def ids_for(text):
                tokens = text.split()
                result = []
                for tok in tokens:
                    if tok not in self.vocab:
                        self.vocab[tok] = len(self.vocab) + 200
                    result.append(self.vocab[tok])
                return result
            ids = [101] + ids_for(a) + [102] + ids_for(b) + [102]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


class FakeModel(torch.nn.Module):
    def __init__(self, logits=(0.1, 0.2, 3.0)):
        super().__init__()
        self.config = types.SimpleNamespace(
            model_type="distilbert", num_labels=3,
            id2label=dict(inf.ID2LABEL), label2id=dict(inf.LABEL2ID), vocab_size=30522,
        )
        self.embedding = torch.nn.Embedding(30522, 2)
        self.logits = list(logits)
        self.forward_calls = 0
        self.no_grad_on_forward = None

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, input_ids, attention_mask):
        self.forward_calls += 1
        self.no_grad_on_forward = not torch.is_grad_enabled()
        self.last_ids = input_ids.cpu().tolist()[0]
        self.last_mask = attention_mask.cpu().tolist()[0]
        return types.SimpleNamespace(logits=torch.tensor([self.logits], device=input_ids.device))


class ValidationTests(unittest.TestCase):
    def test_good_span(self):
        inf.validate_span("The food was good", "food", 4, 8)

    def test_span_at_start(self):
        inf.validate_span("service was good", "service", 0, 7)

    def test_span_at_end(self):
        inf.validate_span("Great food", "food", 6, 10)

    def test_empty_review(self):
        with self.assertRaisesRegex(ValueError, "review"):
            inf.validate_span("   ", "x", 0, 1)

    def test_nonstr_review(self):
        with self.assertRaises(ValueError):
            inf.validate_span(None, "x", 0, 1)

    def test_empty_aspect(self):
        with self.assertRaisesRegex(ValueError, "aspect"):
            inf.validate_span("food", "", 0, 4)

    def test_aspect_whitespace_only(self):
        with self.assertRaises(ValueError):
            inf.validate_span("food ", " ", 4, 5)

    def test_aspect_not_string(self):
        with self.assertRaises(ValueError):
            inf.validate_span("food", ["food"], 0, 4)

    def test_start_negative(self):
        with self.assertRaises(ValueError):
            inf.validate_span("food", "food", -1, 4)

    def test_end_exceeds_length(self):
        with self.assertRaises(ValueError):
            inf.validate_span("food", "food", 0, 5)

    def test_start_equals_end(self):
        with self.assertRaises(ValueError):
            inf.validate_span("food", "food", 1, 1)

    def test_start_greater_than_end(self):
        with self.assertRaises(ValueError):
            inf.validate_span("food", "food", 3, 1)

    def test_bool_start_rejected(self):
        with self.assertRaises(ValueError):
            inf.validate_span("food", "food", False, 4)

    def test_bool_end_rejected(self):
        with self.assertRaises(ValueError):
            inf.validate_span("food", "food", 0, True)

    def test_float_offset_rejected(self):
        with self.assertRaises(ValueError):
            inf.validate_span("food", "food", 0.0, 4)

    def test_case_sensitive_text(self):
        with self.assertRaisesRegex(ValueError, "exactly"):
            inf.validate_span("Great Food", "food", 6, 10)

    def test_wrong_occurrence_offsets(self):
        with self.assertRaises(ValueError):
            inf.validate_span("food and service", "service", 0, 4)

    def test_unicode_python_character_offsets(self):
        review = "Café food is good"
        inf.validate_span(review, "food", 5, 9)

    def test_marking_same_occurrence_helper(self):
        review = "The service was good, but the service was slow."
        second = review.rfind("service")
        got = mark_target_span(review, second, second + 7)
        self.assertEqual(got, "The service was good, but the [unused1] service [unused2] was slow.")

    def test_marker_collision_rejected(self):
        with self.assertRaisesRegex(ValueError, "marker"):
            inf.encode_pair(FakeTokenizer(), "[unused1] food", "food", 10, 14, max_length=128)


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)
        self.manifest = {
            "status": "PREPROCESSING FINISHED",
            "tokenizer_name": inf.MODEL_NAME,
            "vocabulary_resized": False,
            "target_markers": dict(inf.MARKERS),
            "target_disambiguation": "span_markers_from_gold_start_end_offsets",
            "input_format": "[CLS] review with [unused1] target [unused2] [SEP] aspect [SEP]",
            "label_mapping": {str(k): v for k, v in inf.ID2LABEL.items()},
            "truncation": False, "gold_pairs_only": True, "max_length": 128,
        }
        self.write()

    def write(self):
        (self.folder / "manifest.json").write_text(json.dumps(self.manifest), encoding="utf-8")

    def test_read_manifest_128(self):
        self.assertEqual(inf.read_preprocessing_manifest(self.folder), 128)

    def test_missing_manifest(self):
        with self.assertRaises(FileNotFoundError):
            inf.read_preprocessing_manifest(self.folder / "missing")

    def test_bad_status(self):
        self.manifest["status"] = "IN PROGRESS"
        self.write()
        with self.assertRaises(ValueError):
            inf.read_preprocessing_manifest(self.folder)

    def test_marker_ids_mismatch(self):
        self.manifest["target_markers"]["[unused1]"] = 42
        self.write()
        with self.assertRaises(ValueError):
            inf.read_preprocessing_manifest(self.folder)

    def test_label_mapping_mismatch(self):
        self.manifest["label_mapping"]["0"] = "positive"
        self.write()
        with self.assertRaises(ValueError):
            inf.read_preprocessing_manifest(self.folder)

    def test_bad_input_format(self):
        self.manifest["input_format"] = "review only"
        self.write()
        with self.assertRaises(ValueError):
            inf.read_preprocessing_manifest(self.folder)

    def test_long_length_rejected(self):
        self.manifest["max_length"] = 513
        self.write()
        with self.assertRaises(ValueError):
            inf.read_preprocessing_manifest(self.folder)

    def test_bool_length_rejected(self):
        self.manifest["max_length"] = True
        self.write()
        with self.assertRaises(ValueError):
            inf.read_preprocessing_manifest(self.folder)

    def test_truncation_rejected(self):
        self.manifest["truncation"] = True
        self.write()
        with self.assertRaises(ValueError):
            inf.read_preprocessing_manifest(self.folder)

    def test_vocab_resize_rejected(self):
        self.manifest["vocabulary_resized"] = True
        self.write()
        with self.assertRaises(ValueError):
            inf.read_preprocessing_manifest(self.folder)

    def test_only_gold_expected(self):
        self.manifest["gold_pairs_only"] = False
        self.write()
        with self.assertRaises(ValueError):
            inf.read_preprocessing_manifest(self.folder)


class TokenizerAndModelTests(unittest.TestCase):
    def test_good_tokenizer(self):
        inf.validate_tokenizer(FakeTokenizer(), 128)

    def test_slow_tokenizer_rejected(self):
        tok = FakeTokenizer(); tok.is_fast = False
        with self.assertRaises(ValueError):
            inf.validate_tokenizer(tok, 128)

    def test_markers_not_special_rejected(self):
        tok = FakeTokenizer(); tok.additional_special_tokens = []
        with self.assertRaises(ValueError):
            inf.validate_tokenizer(tok, 128)

    def test_marker_wrong_id(self):
        tok = FakeTokenizer(); tok.vocab["[unused1]"] = 12
        with self.assertRaises(ValueError):
            inf.validate_tokenizer(tok, 128)

    def test_malformed_pair_special_count(self):
        tok = FakeTokenizer(); tok.num_special_tokens_to_add = lambda pair: 2
        with self.assertRaises(ValueError):
            inf.validate_tokenizer(tok, 128)

    def test_bad_padding_side(self):
        tok = FakeTokenizer(); tok.padding_side = "left"
        with self.assertRaises(ValueError):
            inf.validate_tokenizer(tok, 128)

    def test_bad_capacity(self):
        tok = FakeTokenizer(); tok.model_max_length = 64
        with self.assertRaises(ValueError):
            inf.validate_tokenizer(tok, 128)

    def test_good_config(self):
        inf.validate_model_config(FakeModel().config)

    def test_config_wrong_label_order(self):
        m = FakeModel(); m.config.id2label = {0: "positive", 1: "neutral", 2: "negative"}
        with self.assertRaises(ValueError):
            inf.validate_model_config(m.config)

    def test_config_wrong_num_labels(self):
        m = FakeModel(); m.config.num_labels = 2
        with self.assertRaises(ValueError):
            inf.validate_model_config(m.config)

    def test_config_wrong_reverse_mapping(self):
        m = FakeModel(); m.config.label2id = {"negative": 1}
        with self.assertRaises(ValueError):
            inf.validate_model_config(m.config)

    def test_config_wrong_model_type(self):
        m = FakeModel(); m.config.model_type = "bert"
        with self.assertRaises(ValueError):
            inf.validate_model_config(m.config)

    def test_cuda_unavailable(self):
        with patch.object(torch.cuda, "is_available", return_value=False):
            with self.assertRaises(ValueError):
                inf.choose_device("cuda", torch)

    def test_cpu_selected(self):
        self.assertEqual(inf.choose_device("cpu", torch), "cpu")

    def test_auto_cpu_selected(self):
        with patch.object(torch.cuda, "is_available", return_value=False):
            self.assertEqual(inf.choose_device("auto", torch), "cpu")

    def test_unknown_device_rejected(self):
        with self.assertRaises(ValueError):
            inf.choose_device("tpu", torch)


class EncodeTests(unittest.TestCase):
    def test_review_aspect_pair_exact_order(self):
        tok = FakeTokenizer()
        review = "The food is good"
        encoded = inf.encode_pair(tok, review, "food", 4, 8, max_length=128)
        a, b, kwargs = tok.last_args
        self.assertEqual(a, "The [unused1] food [unused2] is good")
        self.assertEqual(b, "food")
        self.assertEqual(kwargs["truncation"], False)
        self.assertEqual(kwargs["padding"], False)
        self.assertEqual(encoded["input_ids"][0], tok.cls_token_id)
        self.assertEqual(encoded["input_ids"][-1], tok.sep_token_id)
        self.assertEqual(encoded["input_ids"].count(tok.sep_token_id), 2)

    def test_repeated_aspect_uses_specific_offsets(self):
        tok = FakeTokenizer()
        review = "The service was fast but the service was rude"
        first = review.find("service")
        second = review.rfind("service")
        x = inf.encode_pair(tok, review, "service", first, first + 7, max_length=128)
        y = inf.encode_pair(tok, review, "service", second, second + 7, max_length=128)
        self.assertNotEqual(x["input_ids"], y["input_ids"])

    def test_aspect_multiword(self):
        tok = FakeTokenizer()
        review = "The customer service was kind"
        inf.encode_pair(tok, review, "customer service", 4, 20, max_length=128)
        self.assertIn("[unused1] customer service [unused2]", tok.last_args[0])

    def test_max_length_refusal(self):
        review = " ".join(["food"] * 130)
        with self.assertRaisesRegex(ValueError, "NOT truncated"):
            inf.encode_pair(FakeTokenizer(), review, "food", 0, 4, max_length=128)

    def test_missing_marker_in_tokenizer_output(self):
        tok = FakeTokenizer(); tok.forced_ids = [101, 200, 102, 201, 102]
        with self.assertRaises(ValueError):
            inf.encode_pair(tok, "food is nice", "food", 0, 4, max_length=128)

    def test_marker_order_reversed(self):
        tok = FakeTokenizer(); tok.forced_ids = [101, 3, 200, 2, 102, 200, 102, 201]
        with self.assertRaises(ValueError):
            inf.encode_pair(tok, "food is nice", "food", 0, 4, max_length=128)

    def test_malformed_sep_tokens(self):
        tok = FakeTokenizer(); tok.forced_ids = [101, 2, 200, 3, 102, 200, 201, 201]
        with self.assertRaises(ValueError):
            inf.encode_pair(tok, "food is nice", "food", 0, 4, max_length=128)

    def test_nonunit_attention_mask(self):
        tok = FakeTokenizer()
        original_call = tok.__call__
        # monkeypatch class method because special methods are looked up on type
        with patch.object(FakeTokenizer, "__call__", autospec=True, side_effect=lambda self, a, b, **kw: {
                **original_call(a, b, **kw), "attention_mask": [0] * len(original_call(a,b,**kw)["input_ids"]) }):
            with self.assertRaises(ValueError):
                inf.encode_pair(tok, "food is nice", "food", 0, 4, max_length=128)

    def test_probabilities_all_positive(self):
        label, probs = inf.validate_probabilities([0.0, 0.0, 1.0])
        self.assertEqual(label, 2)
        self.assertEqual(probs["positive"], 1.0)

    def test_probs_invalid_sum(self):
        with self.assertRaises(ValueError):
            inf.validate_probabilities([0.2, 0.2, 0.2])

    def test_probs_nan_rejected(self):
        with self.assertRaises(ValueError):
            inf.validate_probabilities([float("nan"), 0.5, 0.5])

    def test_probs_inf_rejected(self):
        with self.assertRaises(ValueError):
            inf.validate_probabilities([float("inf"), 0.0, 0.0])

    def test_probs_bool_rejected(self):
        with self.assertRaises(ValueError):
            inf.validate_probabilities([True, 0, 0])


class InferenceTests(unittest.TestCase):
    def test_predict_positive_and_no_grad(self):
        fake = FakeModel((0.0, 0.0, 3.0))
        engine = inf.ATSCClassifier(fake, FakeTokenizer(), max_length=128, device="cpu")
        result = engine.predict("The food is nice", "food", 4, 8)
        self.assertEqual(result["sentiment"], "positive")
        self.assertEqual(result["label_id"], 2)
        self.assertEqual(result["aspect"], {"text": "food", "start": 4, "end": 8})
        self.assertEqual(result["review"], "The food is nice")
        self.assertAlmostEqual(sum(result["probabilities_softmax_uncalibrated"].values()), 1.0, places=6)
        self.assertFalse(fake.training)
        self.assertTrue(fake.no_grad_on_forward)
        self.assertEqual(fake.forward_calls, 1)
        self.assertEqual(fake.last_mask, [1] * len(fake.last_ids))

    def test_predict_negative(self):
        engine = inf.ATSCClassifier(FakeModel((3, 0, 0)), FakeTokenizer(), max_length=128)
        self.assertEqual(engine.predict("Bad food", "food", 4, 8)["sentiment"], "negative")

    def test_predict_neutral(self):
        engine = inf.ATSCClassifier(FakeModel((0, 3, 0)), FakeTokenizer(), max_length=128)
        self.assertEqual(engine.predict("Fine food", "food", 5, 9)["sentiment"], "neutral")

    def test_model_reused_many_aspects(self):
        fake = FakeModel((0,0,1))
        engine = inf.ATSCClassifier(fake, FakeTokenizer(), max_length=128)
        text = "The pizza is tasty and the service is slow"
        engine.predict(text, "pizza", 4, 9)
        engine.predict(text, "service", 27, 34)
        self.assertEqual(fake.forward_calls, 2)

    def test_model_rejects_nonfinite_logits(self):
        engine = inf.ATSCClassifier(FakeModel((float("nan"), 0, 0)), FakeTokenizer(), max_length=128)
        with self.assertRaisesRegex(ValueError, "logits"):
            engine.predict("Good food", "food", 5, 9)

    def test_model_rejects_wrong_logits_shape(self):
        model = FakeModel((0, 1))
        engine = inf.ATSCClassifier(model, FakeTokenizer(), max_length=128)
        with self.assertRaisesRegex(ValueError, "logits"):
            engine.predict("Good food", "food", 5, 9)

    def test_model_rejects_wrong_embeddings(self):
        m = FakeModel(); m.embedding = torch.nn.Embedding(30000, 2)
        with self.assertRaises(ValueError):
            inf.ATSCClassifier(m, FakeTokenizer(), max_length=128)

    def test_inference_requires_valid_span_before_model_load(self):
        with patch.object(inf, "load_classifier", side_effect=AssertionError("should not load")):
            with self.assertRaises(ValueError):
                inf.predict_sentiment("food", "service", 0, 4)

    def test_local_model_checks_missing_dir(self):
        with self.assertRaises(FileNotFoundError):
            inf.check_local_model_dir(Path("/__model_does_not_exist__"))

    def test_no_trainer_or_dataset_imports(self):
        source = Path(inf.__file__).read_text(encoding="utf-8")
        self.assertNotIn("from datasets import", source)
        self.assertNotIn("Trainer(", source)
        self.assertNotIn(".train()", source)
        self.assertNotIn('tokenized_splits/test', source)

    def test_cli_parser_requires_review_and_aspect_offsets(self):
        parser = inf.build_parser()
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args(["--review", "hello"])

    def test_cli_synthetic_output(self):
        fake = inf.ATSCClassifier(FakeModel((0, 0, 3)), FakeTokenizer(), max_length=128)
        buf = io.StringIO()
        with patch.object(inf, "load_classifier", return_value=fake), contextlib.redirect_stdout(buf):
            code = inf.main(["--review", "The food is great", "--aspect", "food", "--start", "4", "--end", "8"])
        self.assertEqual(code, 0)
        output = json.loads(buf.getvalue())
        self.assertEqual(output["sentiment"], "positive")
        self.assertIn("probabilities_softmax_uncalibrated", output)

    def test_cli_optional_output_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "existing.json"
            path.write_text("existing", encoding="utf-8")
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as caught:
                    inf.main(["--review", "food", "--aspect", "food", "--start", "0", "--end", "4", "--output", str(path)])
            self.assertEqual(caught.exception.code, 2)
            self.assertEqual(path.read_text(), "existing")

    def test_cli_input_file_utf8(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "input.txt"
            p.write_text("The food is great", encoding="utf-8")
            engine = inf.ATSCClassifier(FakeModel((0, 3, 0)), FakeTokenizer(), max_length=128)
            buf = io.StringIO()
            with patch.object(inf, "load_classifier", return_value=engine), contextlib.redirect_stdout(buf):
                inf.main(["--input-file", str(p), "--aspect", "food", "--start", "4", "--end", "8"])
            self.assertEqual(json.loads(buf.getvalue())["sentiment"], "neutral")

    def test_cli_invalid_span_does_not_load_model(self):
        with patch.object(inf, "load_classifier", side_effect=AssertionError("unexpected")):
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as e:
                    inf.main(["--review", "The food is great", "--aspect", "service", "--start", "4", "--end", "8"])
            self.assertEqual(e.exception.code, 2)


class LoaderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name) / "best_model"
        self.folder.mkdir()
        for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors"):
            (self.folder / name).write_text("synthetic-not-real-model", encoding="utf-8")

    def test_local_model_present(self):
        inf.check_local_model_dir(self.folder)

    def test_local_model_missing_config(self):
        (self.folder / "config.json").unlink()
        with self.assertRaises(FileNotFoundError):
            inf.check_local_model_dir(self.folder)

    def test_local_model_missing_weights(self):
        (self.folder / "model.safetensors").unlink()
        with self.assertRaises(FileNotFoundError):
            inf.check_local_model_dir(self.folder)

    def test_load_classifier_offline_and_no_training(self):
        calls = []
        tokenizer = FakeTokenizer()
        model = FakeModel()

        class AutoTokenizer:
            @staticmethod
            def from_pretrained(path, *, local_files_only, use_fast):
                calls.append(("tok", path, local_files_only, use_fast))
                return tokenizer

        class AutoModelForSequenceClassification:
            @staticmethod
            def from_pretrained(path, *, local_files_only):
                calls.append(("model", path, local_files_only))
                return model

        module = types.ModuleType("transformers")
        module.AutoTokenizer = AutoTokenizer
        module.AutoModelForSequenceClassification = AutoModelForSequenceClassification
        with patch.object(inf, "read_preprocessing_manifest", return_value=128), patch.dict(sys.modules, {"transformers": module}):
            engine = inf.load_classifier(model_dir=self.folder, preprocessing_dir="unused", device="cpu")
        self.assertIsInstance(engine, inf.ATSCClassifier)
        self.assertEqual(calls, [
            ("tok", str(self.folder), True, True),
            ("model", str(self.folder), True),
        ])
        self.assertFalse(model.training)
        self.assertEqual(model.forward_calls, 0)
        self.assertEqual(engine.max_length, 128)

    def test_load_rejects_missing_model_before_import(self):
        with patch.object(inf, "read_preprocessing_manifest", return_value=128):
            with self.assertRaises(FileNotFoundError):
                inf.load_classifier(model_dir=self.folder / "missing")

    def test_cli_output_file_new_only(self):
        with tempfile.TemporaryDirectory() as output_folder:
            output_path = Path(output_folder) / "pred.json"
            fake = inf.ATSCClassifier(FakeModel((2, 0, 0)), FakeTokenizer(), max_length=128)
            with patch.object(inf, "load_classifier", return_value=fake), contextlib.redirect_stdout(io.StringIO()):
                code = inf.main(["--review", "Good food", "--aspect", "food", "--start", "5", "--end", "9", "--output", str(output_path)])
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(output_path.read_text(encoding="utf-8"))["sentiment"], "negative")

    def test_result_rejects_conflicting_marker_in_review(self):
        fake = inf.ATSCClassifier(FakeModel(), FakeTokenizer(), max_length=128)
        review = "[unused2] food is fine"
        with self.assertRaises(ValueError):
            fake.predict(review, "food", review.index("food"), review.index("food") + 4)


if __name__ == "__main__":
    unittest.main()
