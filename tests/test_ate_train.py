"""Offline/unit tests for Step 2.2. NO model download/training/test-set loading.

Fixtures are synthetic; precision/recall/F1 in unit tests are NOT model metrics.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from io import StringIO
from contextlib import redirect_stdout
from unittest.mock import patch
import sys
import types

from src.ate_train import (
    DATASET_NAME, MODEL_NAME, ID2LABEL, SPLITS_FOR_TRAINING,
    bio_entity_spans, check_splits, exact_span_scores,
    first_wordpiece_positions, load_two_splits, make_compute_metrics,
    output_can_be_created, read_and_validate_manifest, training_plan, main,
)


class FakeEncoding(dict):
    def __init__(self, input_ids, words):
        super().__init__(input_ids=input_ids)
        self.words = words

    def word_ids(self, batch_index):
        return self.words[batch_index]


class FakeFastTokenizer:
    is_fast = True

    def __call__(self, token_batch, *, is_split_into_words, truncation, max_length,
                 padding, return_attention_mask, add_special_tokens):
        assert is_split_into_words and truncation and not padding and add_special_tokens
        tokens = token_batch[0]
        # Synthetic WordPiece: 'unaffordable' -> two pieces; never download tokenizer.
        ids = [101]
        word_ids = [None]
        for i, token in enumerate(tokens):
            for code in ([201, 202] if token == "unaffordable" else [203]):
                ids.append(code)
                word_ids.append(i)
        ids.append(102)
        word_ids.append(None)
        return FakeEncoding([ids], [word_ids])


def raw_row(word: str, aspect: bool = True):
    return {"Tokens": [word], "Tags": [1 if aspect else 0]}


def encoded_row(aspect: bool = True):
    return {"input_ids": [101, 203, 102], "attention_mask": [1, 1, 1], "labels": [-100, 1 if aspect else 0, -100]}


def valid_manifest():
    return {
        "dataset": DATASET_NAME,
        "tokenizer": MODEL_NAME,
        "id2label": {str(k): v for k, v in ID2LABEL.items()},
        "training_performed": False,
        "review_overlap_between_final_splits": 0,
        "split_counts": {"train": 2, "validation": 1, "test": 4},
        "max_length": 128,
        "seed": 42,
        "tokenization": {
            "train": {"truncated_reviews": 0, "truncated_aspect_spans": 0},
            "validation": {"truncated_reviews": 0, "truncated_aspect_spans": 0},
            "test": {"truncated_reviews": 0, "truncated_aspect_spans": 0},
        },
    }


class TestAteTrainingStep22(unittest.TestCase):
    def setUp(self):
        self.train_raw = [raw_row("food"), raw_row("service")]
        self.val_raw = [raw_row("price")]
        self.train_tok = [encoded_row(), encoded_row()]
        self.val_tok = [encoded_row()]

    def test_manifest_validates_step21_scope(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "manifest.json"
            p.write_text(json.dumps(valid_manifest()), encoding="utf-8")
            self.assertEqual(read_and_validate_manifest(p)["max_length"], 128)

    def test_manifest_missing_file(self):
        with self.assertRaisesRegex(ValueError, "Missing"):
            read_and_validate_manifest(Path("/this-folder-is-not-present/manifest.json"))

    def test_manifest_rejects_wrong_tokenizer(self):
        m = valid_manifest()
        m["tokenizer"] = "some/other-tokenizer"
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "manifest.json"
            p.write_text(json.dumps(m), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "tokenizer"):
                read_and_validate_manifest(p)

    def test_manifest_rejects_label_mapping(self):
        m = valid_manifest()
        m["id2label"]["1"] = "I-Term"
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "manifest.json"
            p.write_text(json.dumps(m), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "BIO"):
                read_and_validate_manifest(p)

    def test_manifest_rejects_truncated_aspects(self):
        m = valid_manifest()
        m["tokenization"]["validation"]["truncated_aspect_spans"] = 1
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "manifest.json"
            p.write_text(json.dumps(m), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "truncated"):
                read_and_validate_manifest(p)

    def test_manifest_rejects_previously_trained_data(self):
        m = valid_manifest()
        m["training_performed"] = True
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "manifest.json"
            p.write_text(json.dumps(m), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "preprocessing-only"):
                read_and_validate_manifest(p)

    def test_only_train_and_validation_splits_accessed(self):
        paths = []
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            for prefix in ("raw_splits", "tokenized_splits"):
                (directory / prefix).mkdir()
            def loader(path):
                paths.append(Path(path).parts[-2:])
                return ["fake"]
            self.assertEqual(len(load_two_splits(directory, loader)), 4)
            self.assertEqual(paths, [
                ("tokenized_splits", "train"), ("tokenized_splits", "validation"),
                ("raw_splits", "train"), ("raw_splits", "validation"),
            ])
            self.assertNotIn("test", [name for _, name in paths])

    def test_rejects_missing_saved_dataset_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "Missing"):
                load_two_splits(Path(tmp), lambda path: None)

    def test_split_counts_and_no_review_leakage(self):
        result = check_splits(self.train_tok, self.val_tok, self.train_raw, self.val_raw, valid_manifest())
        self.assertEqual(result["split_counts"], {"train": 2, "validation": 1})
        self.assertEqual(result["train_validation_review_overlap"], 0)
        self.assertEqual(result["aligned_label_distribution"]["train"]["B-Term"], 2)

    def test_review_overlap_rejected(self):
        with self.assertRaisesRegex(ValueError, "LEAKAGE"):
            check_splits(self.train_tok, self.val_tok, self.train_raw, [raw_row("FOOD")], valid_manifest())

    def test_mismatched_labels_rejected(self):
        bad = [{**self.train_tok[0], "labels": [-100, 1]}] + self.train_tok[1:]
        with self.assertRaisesRegex(ValueError, "mismatch"):
            check_splits(bad, self.val_tok, self.train_raw, self.val_raw, valid_manifest())

    def test_split_count_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError, "length"):
            check_splits(self.train_tok[:-1], self.val_tok, self.train_raw, self.val_raw, valid_manifest())

    def test_training_plan_has_validation_only_checkpoint_selection(self):
        plan = training_plan(output_dir="out/checkpoints", epochs=8, batch_size=8,
                             eval_batch_size=16, learning_rate=2e-5, seed=42,
                             patience=2, weight_decay=0.01, fp16=False)
        self.assertEqual(plan["eval_strategy"], plan["save_strategy"])
        self.assertEqual(plan["eval_strategy"], "epoch")
        self.assertTrue(plan["load_best_model_at_end"])
        self.assertEqual(plan["metric_for_best_model"], "eval_span_f1")
        self.assertTrue(plan["greater_is_better"])
        self.assertEqual(plan["seed"], 42)
        self.assertFalse(plan["fp16"])

    def test_training_plan_rejects_bad_hyperparams(self):
        kw = dict(output_dir="out", epochs=0, batch_size=8, eval_batch_size=16,
                  learning_rate=2e-5, seed=42, patience=2, weight_decay=0.01, fp16=False)
        with self.assertRaisesRegex(ValueError, "positive"):
            training_plan(**kw)

    def test_empty_output_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "new"
            output_can_be_created(target)
            target.mkdir()
            (target / "x").write_text("do not overwrite", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not empty"):
                output_can_be_created(target)

    def test_exact_span_metric_perfect(self):
        gold = [[0, 1, 2, 0, 1, 0]]
        result = exact_span_scores(gold, gold)
        self.assertEqual(result["span_f1"], 1.0)
        self.assertEqual((result["span_tp"], result["span_fp"], result["span_fn"]), (2, 0, 0))

    def test_exact_span_metric_one_correct_one_incorrect(self):
        gold = [[0, 1, 2, 0, 1, 0]]
        pred = [[0, 1, 2, 1, 0, 0]]
        result = exact_span_scores(gold, pred)
        self.assertEqual((result["span_tp"], result["span_fp"], result["span_fn"]), (1, 1, 1))
        self.assertAlmostEqual(result["span_precision"], 0.5)
        self.assertAlmostEqual(result["span_recall"], 0.5)
        self.assertAlmostEqual(result["span_f1"], 0.5)

    def test_exact_span_metric_no_spans(self):
        result = exact_span_scores([[0, 0]], [[0, 0]])
        self.assertEqual(result["span_f1"], 0.0)
        self.assertEqual((result["span_tp"], result["span_fp"], result["span_fn"]), (0, 0, 0))

    def test_illegal_pred_i_becomes_new_pred_span(self):
        self.assertEqual(bio_entity_spans([2, 2, 0], strict_gold=False), {(0, 2)})
        with self.assertRaisesRegex(ValueError, "Gold BIO"):
            bio_entity_spans([2, 0], strict_gold=True)

    def test_real_word_positions_on_wordpiece_stub(self):
        raw = [{"Tokens": ["unaffordable", "service"], "Tags": [1, 0]}]
        saved = [{"input_ids": [101, 201, 202, 203, 102], "labels": [-100, 1, 2, 0, -100]}]
        self.assertEqual(first_wordpiece_positions(FakeFastTokenizer(), raw, saved, 128), [[1, 3]])

    def test_word_positions_reject_corrupt_token_ids(self):
        raw = [{"Tokens": ["service"], "Tags": [1]}]
        saved = [{"input_ids": [101, 9, 102], "labels": [-100, 1, -100]}]
        with self.assertRaisesRegex(ValueError, "disagrees"):
            first_wordpiece_positions(FakeFastTokenizer(), raw, saved, 128)

    def test_metric_function_uses_only_first_wordpiece(self):
        import numpy as np
        # 5 positions: CLS, B(food)-first, I(food)-continuation, O(service), SEP
        positions = [[1, 3]]
        gold = np.array([[-100, 1, 2, 0, -100]])
        logits = np.zeros((1, 5, 3), dtype=np.float32)
        # Perfect original WORD tags despite wrong subword continuation prediction.
        logits[0, 1, 1] = 10.0
        logits[0, 2, 0] = 10.0
        logits[0, 3, 0] = 10.0
        metric = make_compute_metrics(positions)((logits, gold))
        self.assertEqual(metric["span_f1"], 1.0)

    def test_metric_function_rejects_misaligned_batch(self):
        import numpy as np
        metrics = make_compute_metrics([[1], [1]])
        with self.assertRaisesRegex(ValueError, "row count"):
            metrics((np.zeros((1, 3, 3)), np.array([[-100, 1, -100]])))

    def test_full_cli_dry_run_never_loads_model_or_test(self):
        """Mock library IO, but exercise the real command-line preflight path."""
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "ate_preprocessing"
            base.mkdir()
            (base / "tokenizer").mkdir()
            for prefix in ("raw_splits", "tokenized_splits"):
                (base / prefix).mkdir()
            (base / "manifest.json").write_text(json.dumps(valid_manifest()), encoding="utf-8")
            loaded = []
            def fake_loader(path):
                loaded.append(path)
                parts = Path(path).parts
                kind, name = parts[-2:]
                return {
                    ("tokenized_splits", "train"): self.train_tok,
                    ("tokenized_splits", "validation"): self.val_tok,
                    ("raw_splits", "train"): self.train_raw,
                    ("raw_splits", "validation"): self.val_raw,
                }[(kind, name)]
            mocked_datasets = types.ModuleType("datasets")
            mocked_datasets.load_from_disk = fake_loader
            mocked_transformers = types.ModuleType("transformers")
            class AutoTokenizerStub:
                @staticmethod
                def from_pretrained(*args, **kwargs):
                    return FakeFastTokenizer()
            mocked_transformers.AutoTokenizer = AutoTokenizerStub
            output = Path(tmp) / "training"
            result_stdout = StringIO()
            with patch.dict(sys.modules, {"datasets": mocked_datasets, "transformers": mocked_transformers}):
                with redirect_stdout(result_stdout):
                    return_code = main(["--data", str(base), "--output", str(output), "--dry-run"])
            self.assertEqual(return_code, 0)
            self.assertEqual(len(loaded), 4)
            self.assertNotIn("test", [Path(path).name for path in loaded])
            self.assertFalse(output.exists())
            self.assertIn("DRY RUN FINISHED", result_stdout.getvalue())
            self.assertIn('"training_performed": false', result_stdout.getvalue())

    def test_step_scope_constant_excludes_test(self):
        self.assertEqual(SPLITS_FOR_TRAINING, ("train", "validation"))


if __name__ == "__main__":
    unittest.main()
