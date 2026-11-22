"""Offline unit tests for ATE step 2.1. No model, network, or HF download."""

import unittest
import importlib.util
import tempfile
from pathlib import Path

from src.ate_preprocessing import (
    bio_spans, encode_split, group_ate_rows, prepare_splits,
    review_key, tags_from_spans, tokenize_and_align_labels, validate_row,
)


class FakeBatchEncoding(dict):
    def __init__(self, input_ids, attention_mask, words):
        super().__init__(input_ids=input_ids, attention_mask=attention_mask)
        self._words = words

    def word_ids(self, batch_index):
        return self._words[batch_index]


class FakeFastTokenizer:
    """Simple wordpiece simulation to test aligner without outside packages."""
    is_fast = True

    def __call__(self, token_batches, *, is_split_into_words, truncation, max_length,
                 padding, return_attention_mask, add_special_tokens):
        assert is_split_into_words and not padding and add_special_tokens
        input_ids, masks, words = [], [], []
        for tokens in token_batches:
            pieces, ids = [101], [None]
            for word_id, token in enumerate(tokens):
                chunks = [token] if token != "unaffordable" else ["un", "##affordable"]
                for chunk in chunks:
                    pieces.append(len(chunk) + 200)
                    ids.append(word_id)
            pieces.append(102)
            ids.append(None)
            if truncation and len(pieces) > max_length:
                pieces = pieces[:max_length - 1] + [102]
                ids = ids[:max_length - 1] + [None]
            input_ids.append(pieces)
            masks.append([1] * len(pieces))
            words.append(ids)
        return FakeBatchEncoding(input_ids, masks, words)


def row(tokens, tags, polarity=None):
    result = {"Tokens": tokens, "Tags": [str(t) for t in tags]}
    if polarity is not None:
        result["Polarities"] = polarity
    return result


class TestAtePreprocessing(unittest.TestCase):
    def test_original_label_mapping_and_valid_bio_spans(self):
        self.assertEqual(bio_spans([0, 1, 2, 0, 1, 0]), [(1, 3), (4, 5)])
        self.assertEqual(tags_from_spans(6, [(1, 3), (4, 5)]), [0, 1, 2, 0, 1, 0])

    def test_bio_rejects_stray_i(self):
        with self.assertRaisesRegex(ValueError, "I-Term"):
            validate_row(row(["food"], [2]), source="test", index=0)

    def test_bio_rejects_overlapping_aspect_spans(self):
        with self.assertRaisesRegex(ValueError, "Overlapping"):
            tags_from_spans(4, [(1, 3), (2, 4)])

    def test_split_group_merges_aspects_from_same_review(self):
        rows = [
            row(["The", "food", "and", "service", "."], [0, 1, 0, 0, 0], ["2"]),
            row(["The", "food", "and", "service", "."], [0, 0, 0, 1, 0], ["0"]),
        ]
        groups, stats = group_ate_rows(rows, source="train")
        self.assertEqual(stats["source_rows"], 2)
        self.assertEqual(stats["unique_reviews"], 1)
        self.assertEqual(stats["merged_additional_aspect_spans"], 1)
        self.assertEqual(next(iter(groups.values()))["Tags"], [0, 1, 0, 1, 0])

    def test_polarities_are_not_used_for_ate(self):
        a = row(["food"], [1], ["2"])
        b = row(["food"], [1], ["0"])
        group, _ = group_ate_rows([a, b], source="train")
        self.assertEqual(next(iter(group.values()))["Tags"], [1])
        self.assertNotIn("Polarities", next(iter(group.values())))

    def test_duplicate_review_normalization_is_case_insensitive(self):
        self.assertEqual(review_key(["Food", "Good"]), review_key(["food", "GOOD"]))
        self.assertNotEqual(review_key(["a", "bc"]), review_key(["ab", "c"]))

    def test_final_splits_have_no_review_leakage(self):
        train = [row(["item", str(i)], [1, 0]) for i in range(12)]
        train.append(row(["overlap"], [1]))
        train.append(row(["overlap"], [1]))  # two original rows to remove
        test = [row(["overlap"], [1]), row(["new-test"], [1])]
        splits, stats = prepare_splits(train, test, validation_ratio=0.25, seed=8)
        keys = {s: {review_key(r["Tokens"]) for r in records} for s, records in splits.items()}
        self.assertFalse(keys["train"] & keys["validation"])
        self.assertFalse(keys["train"] & keys["test"])
        self.assertFalse(keys["validation"] & keys["test"])
        self.assertEqual(stats["cross_source_duplicate_review_groups_removed_from_train"], 1)
        self.assertEqual(stats["cross_source_duplicate_training_rows_removed"], 2)
        self.assertEqual(stats["split_counts"], {"train": 9, "validation": 3, "test": 2})

    def test_overlap_policy_error_prevents_leakage(self):
        train = [row(["x"], [1]), row(["y"], [1]), row(["z"], [1])]
        test = [row(["x"], [1])]
        with self.assertRaisesRegex(ValueError, "overlap"):
            prepare_splits(train, test, overlap_policy="error")

    def test_split_is_reproducible_under_row_order_change(self):
        train = [row(["text", str(i)], [0, 1]) for i in range(30)]
        test = [row(["outside"], [1])]
        a, _ = prepare_splits(train, test, seed=42)
        b, _ = prepare_splits(list(reversed(train)), test, seed=42)
        self.assertEqual(
            {s: {review_key(r["Tokens"]) for r in a[s]} for s in a},
            {s: {review_key(r["Tokens"]) for r in b[s]} for s in b},
        )

    def test_wordpiece_b_to_i_and_special_token_ignore(self):
        batch = {"Tokens": [["unaffordable", "service"]], "Tags": [[1, 0]]}
        output = tokenize_and_align_labels(batch, FakeFastTokenizer())
        self.assertEqual(output["labels"], [[-100, 1, 2, 0, -100]])
        self.assertEqual(output["attention_mask"], [[1, 1, 1, 1, 1]])
        self.assertEqual(len(output["labels"][0]), len(output["input_ids"][0]))

    def test_multiword_aspect_and_subword_i(self):
        batch = {"Tokens": [["unaffordable", "unaffordable", "today"]], "Tags": [[1, 2, 0]]}
        output = tokenize_and_align_labels(batch, FakeFastTokenizer())
        self.assertEqual(output["labels"], [[-100, 1, 2, 2, 2, 0, -100]])

    def test_truncation_is_reported_including_lost_aspect(self):
        records = [{"Tokens": ["a", "b", "c", "d", "service"], "Tags": [0, 0, 0, 0, 1]}]
        _, stats = encode_split(records, FakeFastTokenizer(), max_length=5)
        self.assertEqual(stats, {"truncated_reviews": 1, "truncated_aspect_spans": 1})

    def test_pdf_example_gold_ate_labels_not_model_predictions(self):
        tokens = ["The", "food", "was", "delicious", ",", "but", "the", "service", "was", "terrible", "."]
        tags = [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0]
        grouped, _ = group_ate_rows([row(tokens, tags)], source="pdf-fixture")
        example = next(iter(grouped.values()))
        self.assertEqual(example["Tags"], tags)
        self.assertEqual(
            [example["Tokens"][start:end] for start, end in bio_spans(example["Tags"])],
            [["food"], ["service"]],
        )

    @unittest.skipUnless(importlib.util.find_spec("transformers") is not None, "transformers not installed")
    def test_real_fast_wordpiece_tokenizer_offline(self):
        # Real HF tokenization without model weights or network access.
        from transformers import BertTokenizerFast
        vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "un", "##affordable", "service"]
        with tempfile.TemporaryDirectory() as tmp:
            vocab_path = Path(tmp) / "vocab.txt"
            vocab_path.write_text("\n".join(vocab) + "\n", encoding="utf-8")
            fast = BertTokenizerFast(vocab_file=str(vocab_path), do_lower_case=True)
            encoded = tokenize_and_align_labels(
                {"Tokens": [["unaffordable", "service"]], "Tags": [[1, 0]]}, fast
            )
            self.assertEqual(encoded["labels"], [[-100, 1, 2, 0, -100]])

    def test_partial_wordpiece_truncation_is_detected(self):
        records = [{"Tokens": ["unaffordable", "service"], "Tags": [1, 0]}]
        _, stats = encode_split(records, FakeFastTokenizer(), max_length=3)
        self.assertEqual(stats, {"truncated_reviews": 1, "truncated_aspect_spans": 1})

    def test_invalid_mismatched_lengths_rejected(self):
        with self.assertRaisesRegex(ValueError, "equal-length"):
            group_ate_rows([row(["food", "service"], [1])], source="train")

    def test_invalid_slow_tokenizer_rejected(self):
        class Slow:
            is_fast = False
        with self.assertRaisesRegex(ValueError, "fast tokenizer"):
            tokenize_and_align_labels({"Tokens": [["food"]], "Tags": [[1]]}, Slow())


if __name__ == "__main__":
    unittest.main()
