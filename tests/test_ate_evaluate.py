"""Step 2.3 tests — all model results are SYNTHETIC, not real F1 scores.

Tests are designed to run offline without downloading Transformers or datasets.
Torch is used only for a tiny mocked token classifier forward pass.
"""
from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from src.ate_evaluate import (
    EXPECTED_SPLIT, build_parser, check_model_labels, choose_device,
    display_spans, iter_model_batches, main, make_error_examples, make_report,
    require_artifact_files, verify_test_splits, word_tags_from_token_predictions,
)
from src.ate_train import exact_span_scores, first_wordpiece_positions


TRAIN = [{"Tokens": ["Food"], "Tags": [1]}]
VAL = [{"Tokens": ["service"], "Tags": [1]}]
TEST = [
    {"Tokens": ["The", "pizza", "was", "delicious"], "Tags": [0, 1, 0, 0]},
    {"Tokens": ["Nice", "service"], "Tags": [0, 1]},
]
ENCODED = [
    {"input_ids": [101, 10, 11, 12, 13, 102],
     "attention_mask": [1, 1, 1, 1, 1, 1], "labels": [-100, 0, 1, 0, 0, -100]},
    {"input_ids": [101, 20, 21, 102],
     "attention_mask": [1, 1, 1, 1], "labels": [-100, 0, 1, -100]},
]


def manifest():
    return {
        "dataset": "thainq107/abte-restaurants",
        "tokenizer": "distilbert/distilbert-base-uncased",
        "training_performed": False,
        "id2label": {"0": "O", "1": "B-Term", "2": "I-Term"},
        "review_overlap_between_final_splits": 0,
        "split_counts": {"train": 1, "validation": 1, "test": 2},
        "tokenization": {
            name: {"truncated_reviews": 0, "truncated_aspect_spans": 0}
            for name in ("train", "validation", "test")
        },
        "max_length": 128, "seed": 42,
    }


def training_summary():
    return {
        "step": "2.2", "model": "distilbert/distilbert-base-uncased",
        "dataset": "thainq107/abte-restaurants", "training_performed": True,
        "test_used": False, "train_examples": 1, "validation_examples": 1,
        "best_checkpoint": "checkpoints/checkpoint-10", "best_validation_span_f1": 0.6,
        "training_config": {"metric_for_best_model": "eval_span_f1", "load_best_model_at_end": True},
    }


class FakeBatchEncoding(dict):
    def __init__(self, *args, wordids=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._wordids = wordids

    def word_ids(self, batch_index=0):
        return self._wordids[batch_index]


class FakeTokenizer:
    is_fast = True
    padding_side = "right"
    pad_token_id = 0

    def __call__(self, words_batches, **kwargs):
        assert kwargs["is_split_into_words"] is True
        ids = []
        wordids = []
        for words in words_batches:
            if words == TEST[0]["Tokens"]:
                ids.append(ENCODED[0]["input_ids"])
            elif words == TEST[1]["Tokens"]:
                ids.append(ENCODED[1]["input_ids"])
            else:
                raise AssertionError("Unexpected tokenizer input")
            wordids.append([None, *range(len(words)), None])
        return FakeBatchEncoding(input_ids=ids, wordids=wordids)


class FakeModel:
    """No training capability; predicts pizza as B, misses service in review 2."""
    def __init__(self):
        self.eval_calls = 0
        self.forward_calls = 0
        self.config = SimpleNamespace(
            num_labels=3, id2label={0: "O", 1: "B-Term", 2: "I-Term"}
        )

    def eval(self):
        self.eval_calls += 1
        return self

    def to(self, device):
        self.device = device
        return self

    def train(self, *args, **kwargs):
        raise AssertionError("TEST EVALUATION MUST NEVER CALL train()")

    def __call__(self, *, input_ids, attention_mask):
        self.forward_calls += 1
        assert input_ids.shape == attention_mask.shape
        logits = torch.zeros((*input_ids.shape, 3), device=input_ids.device)
        logits[:, :, 0] = 5
        logits[:, :, 1][input_ids == 11] = 10
        return SimpleNamespace(logits=logits)


class Step23Tests(unittest.TestCase):
    def create_artifacts(self, root: Path):
        data = root / "ate_preprocessing"
        model = root / "best_model"
        summary_path = root / "run_summary.json"
        (data / "raw_splits" / "test").mkdir(parents=True)
        (data / "raw_splits" / "train").mkdir(parents=True)
        (data / "raw_splits" / "validation").mkdir(parents=True)
        (data / "tokenized_splits" / "test").mkdir(parents=True)
        model.mkdir(parents=True)
        for file in ("config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors"):
            (model / file).write_text("fake model placeholder", encoding="utf-8")
        (data / "manifest.json").write_text(json.dumps(manifest()), encoding="utf-8")
        summary_path.write_text(json.dumps(training_summary()), encoding="utf-8")
        return data, model, summary_path

    def test_preflight_metadata_only_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            data, model, summary = self.create_artifacts(Path(tmp))
            result = require_artifact_files(data, model, summary, Path(tmp) / "report", manifest())
            self.assertEqual(result["preflight_status"], "PASSED")
            self.assertFalse(result["test_opened"])
            self.assertFalse(result["model_weights_loaded"])

    def test_preflight_rejects_missing_weight(self):
        with tempfile.TemporaryDirectory() as tmp:
            data, model, summary = self.create_artifacts(Path(tmp))
            (model / "model.safetensors").unlink()
            with self.assertRaisesRegex(ValueError, "Missing model weights"):
                require_artifact_files(data, model, summary, Path(tmp) / "report", manifest())

    def test_preflight_rejects_training_with_test_used(self):
        with tempfile.TemporaryDirectory() as tmp:
            data, model, summary = self.create_artifacts(Path(tmp))
            incorrect = training_summary()
            incorrect["test_used"] = True
            summary.write_text(json.dumps(incorrect), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "test-free training"):
                require_artifact_files(data, model, summary, Path(tmp) / "report", manifest())

    def test_preflight_rejects_wrong_metric_selected(self):
        with tempfile.TemporaryDirectory() as tmp:
            data, model, summary = self.create_artifacts(Path(tmp))
            incorrect = training_summary()
            incorrect["training_config"]["metric_for_best_model"] = "eval_accuracy"
            summary.write_text(json.dumps(incorrect), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "validation span-F1"):
                require_artifact_files(data, model, summary, Path(tmp) / "report", manifest())

    def test_preflight_rejects_truncated_gold(self):
        with tempfile.TemporaryDirectory() as tmp:
            data, model, summary = self.create_artifacts(Path(tmp))
            m = manifest()
            m["tokenization"]["test"]["truncated_aspect_spans"] = 1
            with self.assertRaisesRegex(ValueError, "truncated"):
                require_artifact_files(data, model, summary, Path(tmp) / "report", m)

    def test_preflight_guards_existing_report_even_if_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            data, model, summary = self.create_artifacts(Path(tmp))
            output = Path(tmp) / "report"
            output.mkdir()
            with self.assertRaisesRegex(ValueError, "already exists"):
                require_artifact_files(data, model, summary, output, manifest())

    def test_preflight_requires_test_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            data, model, summary = self.create_artifacts(Path(tmp))
            (data / "raw_splits" / "test").rmdir()
            with self.assertRaisesRegex(ValueError, "Missing held-out test"):
                require_artifact_files(data, model, summary, Path(tmp) / "report", manifest())

    def test_verify_test_nonoverlap_and_count(self):
        result = verify_test_splits(TEST, ENCODED, TRAIN, VAL, manifest())
        self.assertEqual(result["test_examples"], 2)
        self.assertEqual(result["gold_aspect_count"], 2)
        self.assertEqual(result["test_review_overlap_with_train_validation"], 0)

    def test_verify_test_rejects_train_overlap(self):
        bad = [{"Tokens": ["The", "pizza", "was", "delicious"], "Tags": [1, 0, 0, 0]}]
        with self.assertRaisesRegex(ValueError, "LEAKAGE"):
            verify_test_splits(TEST, ENCODED, bad, VAL, manifest())

    def test_verify_test_rejects_duplicate_test_reviews(self):
        dup = [TEST[0], TEST[0]]
        with self.assertRaisesRegex(ValueError, "Duplicate normalized test"):
            verify_test_splits(dup, ENCODED, TRAIN, VAL, manifest())

    def test_verify_test_rejects_invalid_gold(self):
        bad = [{"Tokens": ["The", "pizza", "was", "delicious"], "Tags": [2, 1, 0, 0]}, TEST[1]]
        with self.assertRaisesRegex(ValueError, "Gold BIO"):
            verify_test_splits(bad, ENCODED, TRAIN, VAL, manifest())

    def test_verify_test_rejects_invalid_attention_mask(self):
        wrong = [dict(ENCODED[0], attention_mask=[0, 1, 1, 1, 1, 1]), ENCODED[1]]
        with self.assertRaisesRegex(ValueError, "attention mask"):
            verify_test_splits(TEST, wrong, TRAIN, VAL, manifest())

    def test_model_mapping_check(self):
        check_model_labels(FakeModel().config)
        with self.assertRaisesRegex(ValueError, "Unexpected"):
            check_model_labels(SimpleNamespace(num_labels=3, id2label={0: "A", 1: "B", 2: "C"}))

    def test_tokenizer_first_subword_positions(self):
        expected = [[1, 2, 3, 4], [1, 2]]
        self.assertEqual(first_wordpiece_positions(FakeTokenizer(), TEST, ENCODED, 128), expected)

    def test_batch_inference_uses_eval_and_no_labels(self):
        model = FakeModel()
        pred_batches = list(iter_model_batches(ENCODED, batch_size=2, pad_token_id=0, device="cpu", model=model))
        self.assertEqual(model.eval_calls, 1)
        self.assertEqual(model.forward_calls, 1)
        self.assertEqual(len(pred_batches), 1)
        self.assertEqual([len(x) for x in pred_batches[0]], [6, 4])
        self.assertEqual(pred_batches[0][0][2], 1)
        self.assertEqual(pred_batches[0][1][2], 0)

    def test_batch_inference_checks_positive_batch(self):
        with self.assertRaisesRegex(ValueError, "batch_size"):
            list(iter_model_batches(ENCODED, batch_size=0, pad_token_id=0, device="cpu", model=FakeModel()))

    def test_word_projection_preserves_original_offsets(self):
        result = word_tags_from_token_predictions(
            TEST, ENCODED, [[1, 2, 3, 4], [1, 2]],
            [[0, 0, 1, 0, 0, 0], [0, 0, 0, 0]],
        )
        gold, pred = result
        self.assertEqual(gold, [[0, 1, 0, 0], [0, 1]])
        self.assertEqual(pred, [[0, 1, 0, 0], [0, 0]])
        self.assertAlmostEqual(exact_span_scores(gold, pred)["span_f1"], 2/3)

    def test_word_projection_rejects_mismatched_predictions(self):
        with self.assertRaisesRegex(ValueError, "alignment size"):
            word_tags_from_token_predictions(TEST, ENCODED, [[1, 2, 3, 4], [1, 2]], [[0, 1], [0, 1, 0, 0]])

    def test_error_example_explicit_missed_aspect(self):
        pred = [[0, 1, 0, 0], [0, 0]]
        gold = [[0, 1, 0, 0], [0, 1]]
        examples, counts = make_error_examples(TEST, gold, pred, 25)
        self.assertEqual(counts["test_reviews_with_any_span_error"], 1)
        self.assertEqual(examples[0]["missed_gold_spans"][0]["term"], "service")
        self.assertEqual(examples[0]["test_index"], 1)

    def test_error_example_bounds_and_false_positive(self):
        pred = [[0, 1, 2, 0], [0, 1]]
        gold = [[0, 1, 0, 0], [0, 1]]
        examples, counts = make_error_examples(TEST, gold, pred, 25)
        self.assertEqual(counts["test_reviews_with_false_positive"], 1)
        self.assertEqual(counts["test_reviews_with_missed_gold"], 1)
        self.assertEqual(examples[0]["false_positive_spans"][0]["term"], "pizza was")
        self.assertEqual(examples[0]["missed_gold_spans"][0]["term"], "pizza")

    def test_device_cpu_override(self):
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))
        self.assertEqual(choose_device("cpu", fake_torch), "cpu")
        self.assertEqual(choose_device("auto", fake_torch), "cuda")
        fake_torch.cuda.is_available = lambda: False
        with self.assertRaisesRegex(ValueError, "cuda"):
            choose_device("cuda", fake_torch)

    def test_parser_default_split(self):
        self.assertEqual(EXPECTED_SPLIT, "test")
        args = build_parser().parse_args([])
        self.assertEqual(args.model, "artifacts/ate_training/best_model")
        self.assertEqual(args.device, "auto")

    def test_report_distinguishes_test_and_validation(self):
        scores = exact_span_scores([[1, 0]], [[1, 0]])
        report = make_report(manifest(), training_summary(), {"test_examples": 2}, scores,
                             {"stored_error_examples": 0}, model_dir=Path("model"),
                             test_examples=2, batch_size=8, device="cpu")
        self.assertEqual(report["best_validation_span_f1_prior_to_test"], 0.6)
        self.assertEqual(report["test_metrics"]["span_f1"], 1.0)
        self.assertTrue(report["test_used"])
        self.assertFalse(report["model_trained_during_evaluation"])

    def test_preflight_cli_never_loads_test_or_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            data, model, summary = self.create_artifacts(Path(tmp))
            output = Path(tmp) / "report"
            capture = StringIO()
            with redirect_stdout(capture):
                rc = main(["--data", str(data), "--model", str(model),
                           "--training-summary", str(summary), "--output", str(output), "--preflight"])
            self.assertEqual(rc, 0)
            self.assertFalse(output.exists())
            self.assertIn("PREFLIGHT FINISHED", capture.getvalue())

    def test_full_cli_synthetic_end_to_end_inference(self):
        """Simulates I/O and model: resulting F1 is synthetic and NOT a real test metric."""
        with tempfile.TemporaryDirectory() as tmp:
            data, model, summary = self.create_artifacts(Path(tmp))
            output = Path(tmp) / "report"
            loaded = []
            def loader(path):
                name, split = Path(path).parts[-2:]
                loaded.append((name, split))
                return {
                    ("raw_splits", "train"): TRAIN,
                    ("raw_splits", "validation"): VAL,
                    ("raw_splits", "test"): TEST,
                    ("tokenized_splits", "test"): ENCODED,
                }[(name, split)]
            fake_datasets = ModuleType("datasets")
            fake_datasets.load_from_disk = loader
            fake_transformers = ModuleType("transformers")
            class FakeAutoTokenizer:
                @staticmethod
                def from_pretrained(*args, **kwargs):
                    self.assertTrue(kwargs["local_files_only"])
                    return FakeTokenizer()
            class FakeAutoModel:
                @staticmethod
                def from_pretrained(*args, **kwargs):
                    self.assertTrue(kwargs["local_files_only"])
                    return FakeModel()
            fake_transformers.AutoTokenizer = FakeAutoTokenizer
            fake_transformers.AutoModelForTokenClassification = FakeAutoModel
            stdout = StringIO()
            with patch.dict(sys.modules, {"datasets": fake_datasets, "transformers": fake_transformers}):
                with redirect_stdout(stdout):
                    rc = main(["--data", str(data), "--model", str(model),
                               "--training-summary", str(summary), "--output", str(output), "--device", "cpu"])
            self.assertEqual(rc, 0)
            self.assertEqual(loaded, [
                ("raw_splits", "test"), ("tokenized_splits", "test"),
                ("raw_splits", "train"), ("raw_splits", "validation"),
            ])
            report = json.loads((output / "test_metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(report["test_examples"], 2)
            self.assertEqual(report["test_metrics"]["span_tp"], 1)
            self.assertEqual(report["test_metrics"]["span_fp"], 0)
            self.assertEqual(report["test_metrics"]["span_fn"], 1)
            self.assertAlmostEqual(report["test_metrics"]["span_f1"], 2/3)
            self.assertFalse(report["model_trained_during_evaluation"])
            self.assertIn("TEST EVALUATION FINISHED", stdout.getvalue())
            error_lines = (output / "error_examples.jsonl").read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(error_lines), 1)
            self.assertEqual(json.loads(error_lines[0])["missed_gold_spans"][0]["term"], "service")


if __name__ == "__main__":
    unittest.main()
