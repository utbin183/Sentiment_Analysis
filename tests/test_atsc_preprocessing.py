"""Step 3.2 offline unit tests. Fake tokenizer/gold labels are SYNTHETIC, not real model results."""
from __future__ import annotations

from contextlib import redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

from atsc_preprocessing import (
    POLARITY_ID_TO_NAME, TEXT_ORIGIN, SPLITS, TOKENIZER_NAME, DATASET_NAME,
    read_audit_report, validate_pair, load_pairs, validate_ate_membership,
    check_tokenizer, tokenize_gold_pairs, class_imbalance_report, main,
    review_key_from_text, mark_target_span, register_target_markers,
    TARGET_START_MARKER, TARGET_END_MARKER, repeated_target_report,
)


def pair(tokens: list[str], span: tuple[int, int], label: int, split: str) -> dict:
    review = " ".join(tokens)
    a,b = span
    start = len(" ".join(tokens[:a])) + (1 if a else 0)
    aspect = " ".join(tokens[a:b])
    return {"review":review, "aspect":aspect, "start_word":a,"end_word":b,
            "start":start,"end":start+len(aspect),"sentiment_id":label,
            "sentiment":POLARITY_ID_TO_NAME[label],"split":split,"text_origin":TEXT_ORIGIN}


def fixture():
    food = ["The","food","was","great",",","service","slow","."]
    price = ["The","price","was","okay","."]
    staff = ["Staff","was","lovely","."]
    pasta = ["Pasta","was","cold","."]
    return {
        "train":[pair(food,(1,2),2,"train"),pair(food,(5,6),0,"train"),pair(price,(1,2),1,"train")],
        "validation":[pair(staff,(0,1),2,"validation")],
        "test":[pair(pasta,(0,1),0,"test")],
    }


def report_for(pairs):
    names = POLARITY_ID_TO_NAME.values()
    return {"status":"AUDIT PASSED", "dataset":DATASET_NAME,
            "polarity_mapping":{str(k):v for k,v in POLARITY_ID_TO_NAME.items()},
            "unlabeled_polarity":-1, "review_overlap_between_splits":0,
            "sentiment_train_validation_test_strategy":"reuse_exact_ate_step_2_1_saved_review_splits",
            "ate_gold_not_predictions":True,
            "character_offsets":"reconstructed_space_joined_tokens_not_original_raw_review_offsets",
            "pair_counts":{s:len(pairs[s]) for s in SPLITS},
            "unique_paired_reviews":{s:len({review_key_from_text(r['review']) for r in pairs[s]}) for s in SPLITS},
            "class_distribution":{s:{name:sum(r['sentiment']==name for r in pairs[s])
                                       for name in names} for s in SPLITS}}


class FakeTokenizer:
    def __init__(self):
        self.is_fast = True
        self.model_max_length = 512
        self.pad_token_id = 0
        self.unk_token_id = 100
        self.vocab = {TARGET_START_MARKER: 2, TARGET_END_MARKER: 3}
        self.vocab_size = 30522
        self.registered_markers = []
        self.calls = []
        self.length_overrides = {}
        self.saved = False
    def num_special_tokens_to_add(self, pair=False):
        return 3 if pair else 2
    def get_vocab(self):
        return self.vocab
    def __len__(self):
        return self.vocab_size
    def add_special_tokens(self, specification):
        self.registered_markers = specification["additional_special_tokens"]
        return 0
    def convert_tokens_to_ids(self, text):
        return self.vocab.get(text, self.unk_token_id)
    def __call__(self, review, aspect, *, add_special_tokens, padding, truncation, return_attention_mask):
        self.calls.append((review, aspect, add_special_tokens, padding, truncation, return_attention_mask))
        # mimic [CLS] marked review [SEP] aspect [SEP]; deterministic toy IDs.
        words_a = [self.vocab.get(w, 10 + sum(map(ord,w)) % 73) for w in review.split()]
        words_b = [10 + sum(map(ord,w))%73 for w in aspect.split()]
        ids = [101] + words_a + [102] + words_b + [102]
        override = self.length_overrides.get(aspect)
        if override is not None:
            # Keep target markers in fake overlong inputs to test length handling.
            ids = [101, 2] + [11]*(override-4) + [3, 102]
        return {"input_ids":ids,"attention_mask":[1]*len(ids)}
    def save_pretrained(self, dirname):
        self.saved = True
        Path(dirname).mkdir(parents=True,exist_ok=True)
        (Path(dirname)/"fake_tokenizer.json").write_text("{}",encoding="utf-8")


class TestsStep32(unittest.TestCase):
    def setUp(self):
        self.pairs = fixture()
        self.report = report_for(self.pairs)
        self.tok = FakeTokenizer()

    def write_fixture(self, root):
        root=Path(root)
        audit=root/"atsc_audit"
        (audit/"pairs").mkdir(parents=True, exist_ok=True)
        (audit/"audit_report.json").write_text(json.dumps(self.report),encoding="utf-8")
        for split in SPLITS:
            with (audit/"pairs"/f"{split}.jsonl").open("w",encoding="utf-8") as f:
                for row in self.pairs[split]:
                    f.write(json.dumps(row,ensure_ascii=False)+"\n")
        ate=root/"ate_preprocessing"
        (ate/"raw_splits").mkdir(parents=True, exist_ok=True)
        (ate/"tokenizer").mkdir(parents=True, exist_ok=True)
        ate_data={split:[{"Tokens":r["review"].split(" ")} for r in self.pairs[split]]
                  for split in SPLITS}
        ate_data = {split: list({tuple(r["Tokens"]): r for r in rows}.values())
                    for split, rows in ate_data.items()}
        (ate/"manifest.json").write_text(json.dumps({"dataset":DATASET_NAME,
            "tokenizer":TOKENIZER_NAME,"review_overlap_between_final_splits":0,
            "split_counts":{s:len(ate_data[s]) for s in SPLITS}}),encoding="utf-8")
        return audit, ate, ate_data

    def test_mapping_exact_and_ignore_not_class(self):
        self.assertEqual(POLARITY_ID_TO_NAME, {0:"negative",1:"neutral",2:"positive"})
        self.assertNotIn(-1,POLARITY_ID_TO_NAME)

    def test_valid_gold_span_and_label(self):
        r = validate_pair(self.pairs["train"][0],split="train",index=0)
        self.assertEqual(r["aspect"],"food")
        self.assertEqual(r["review"][r["start"]:r["end"]],"food")

    def test_multiword_aspect(self):
        r=pair(["Great", "customer", "service", "!"], (1,3), 2, "train")
        self.assertEqual(validate_pair(r,split="train",index=0)["aspect"],"customer service")

    def test_punctuation_and_space_reconstruction(self):
        r=pair(["The","food",",","service","."],(3,4),0,"test")
        self.assertEqual(r["review"],"The food , service .")
        self.assertEqual(validate_pair(r,split="test",index=0)["start"],11)

    def test_wrong_class_name_rejected(self):
        r={**self.pairs["train"][0],"sentiment":"negative"}
        with self.assertRaisesRegex(ValueError,"disagrees"):
            validate_pair(r,split="train",index=0)

    def test_minus_one_rejected(self):
        r={**self.pairs["train"][0],"sentiment_id":-1}
        with self.assertRaisesRegex(ValueError,"must be 0"):
            validate_pair(r,split="train",index=0)

    def test_bool_label_rejected(self):
        r={**self.pairs["train"][0],"sentiment_id":True}
        with self.assertRaisesRegex(ValueError,"integer"):
            validate_pair(r,split="train",index=0)

    def test_missing_field_rejected(self):
        r=dict(self.pairs["train"][0]); del r["start"]
        with self.assertRaisesRegex(ValueError,"missing fields"):
            validate_pair(r,split="train",index=0)

    def test_bad_character_span_rejected(self):
        r={**self.pairs["train"][0],"start":3}
        with self.assertRaisesRegex(ValueError,"character offsets"):
            validate_pair(r,split="train",index=0)

    def test_bad_word_span_rejected(self):
        r={**self.pairs["train"][0],"start_word":0}
        with self.assertRaisesRegex(ValueError,"offsets disagree"):
            validate_pair(r,split="train",index=0)

    def test_wrong_split_rejected(self):
        with self.assertRaisesRegex(ValueError,"belongs to split"):
            validate_pair(self.pairs["train"][0],split="test",index=0)

    def test_unknown_origin_rejected(self):
        r={**self.pairs["train"][0],"text_origin":"raw"}
        with self.assertRaisesRegex(ValueError,"text_origin"):
            validate_pair(r,split="train",index=0)

    def test_review_key_preserves_token_boundaries_and_ignores_case(self):
        self.assertEqual(review_key_from_text("The FOOD"),review_key_from_text("the food"))
        self.assertNotEqual(review_key_from_text("ab c"),review_key_from_text("a bc"))

    def test_read_report_happy_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=Path(tmp)/"audit_report.json"
            f.write_text(json.dumps(self.report),encoding="utf-8")
            self.assertEqual(read_audit_report(f)["pair_counts"]["train"],3)

    def test_read_report_missing_rejected(self):
        with self.assertRaises(FileNotFoundError):
            read_audit_report(Path("definitely_missing_ATSC_audit_020302.json"))

    def test_read_report_status_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=Path(tmp)/"audit_report.json"
            f.write_text(json.dumps({**self.report,"status":"FAILED"}),encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"AUDIT PASSED"):
                read_audit_report(f)

    def test_report_wrong_mapping_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=Path(tmp)/"audit_report.json"
            f.write_text(json.dumps({**self.report,"polarity_mapping":{"0":"positive"}}),encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"mapping mismatch"):
                read_audit_report(f)

    def test_report_count_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=Path(tmp)/"audit_report.json"
            report={**self.report,"pair_counts":{**self.report["pair_counts"],"train":4}}
            f.write_text(json.dumps(report),encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"sum to pair count"):
                read_audit_report(f)

    def test_load_pairs_valid_counts_and_sentiment(self):
        with tempfile.TemporaryDirectory() as tmp:
            audit,_,_=self.write_fixture(tmp)
            actual=load_pairs(audit/"pairs",self.report)
            self.assertEqual({s:len(actual[s]) for s in SPLITS},{"train":3,"validation":1,"test":1})

    def test_cross_split_leakage_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.pairs["validation"]=[{**self.pairs["train"][0],"split":"validation"}]
            audit,_,_=self.write_fixture(tmp)
            with self.assertRaisesRegex(ValueError,"Review leakage"):
                load_pairs(audit/"pairs",report_for(self.pairs))

    def test_case_insensitive_leakage_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.pairs["test"]=[{**pair(["THE","FOOD","WAS","GREAT",",","SERVICE","SLOW","."],(1,2),2,"test")}]
            audit,_,_=self.write_fixture(tmp)
            with self.assertRaisesRegex(ValueError,"Review leakage"):
                load_pairs(audit/"pairs",report_for(self.pairs))

    def test_duplicate_span_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.pairs["train"].append(dict(self.pairs["train"][0]))
            audit,_,_=self.write_fixture(tmp)
            with self.assertRaisesRegex(ValueError,"Duplicate gold span"):
                load_pairs(audit/"pairs",report_for(self.pairs))

    def test_repeated_aspect_conflicting_sentiments_kept_with_distinct_marked_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=pair(["service","great","but","service","bad"],(0,1),2,"train")
            r2=pair(["service","great","but","service","bad"],(3,4),0,"train")
            self.pairs["train"]=[r,r2]
            audit,_,_=self.write_fixture(tmp)
            rows = load_pairs(audit/"pairs",report_for(self.pairs))["train"]
            self.assertEqual(len(rows),2)
            register_target_markers(self.tok)
            encoded,_=tokenize_gold_pairs(rows,self.tok,max_length=128,split="train")
            self.assertEqual([r["labels"] for r in encoded],[2,0])
            self.assertNotEqual(encoded[0]["input_ids"],encoded[1]["input_ids"])
            self.assertEqual(self.tok.calls[0][0],"[unused1] service [unused2] great but service bad")
            self.assertEqual(self.tok.calls[1][0],"service great but [unused1] service [unused2] bad")
            stats = repeated_target_report(self.pairs)
            self.assertEqual(stats["train"]["conflicting_sentiment_repeated_text_groups"], 1)
            self.assertEqual(stats["train"]["conflicting_sentiment_repeated_text_pairs"], 2)

    def test_same_aspect_text_same_label_keeps_distinct_spans(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=pair(["service","and","service"],(0,1),2,"train")
            r2=pair(["service","and","service"],(2,3),2,"train")
            self.pairs["train"]=[r,r2]
            audit,_,_=self.write_fixture(tmp)
            self.assertEqual(len(load_pairs(audit/"pairs",report_for(self.pairs))["train"]),2)

    def test_member_check_valid(self):
        ate={s:[{"Tokens":r["review"].split(" ")} for r in rows] for s,rows in self.pairs.items()}
        ate["train"]=[ate["train"][0],ate["train"][2]]
        self.assertEqual(validate_ate_membership(self.pairs,ate),{"train":2,"validation":1,"test":1})

    def test_member_check_wrong_owner_rejected(self):
        ate={s:[{"Tokens":r["review"].split(" ")} for r in rows] for s,rows in self.pairs.items()}
        ate["train"]=[ate["train"][0],ate["train"][2]]
        ate["train"].remove(ate["train"][0])
        with self.assertRaisesRegex(ValueError,"does not belong"):
            validate_ate_membership(self.pairs,ate)

    def test_member_check_ate_leak_rejected(self):
        ate={s:[{"Tokens":r["review"].split(" ")} for r in rows] for s,rows in self.pairs.items()}
        ate["train"]=[ate["train"][0],ate["train"][2]]
        ate["validation"].append(dict(ate["train"][0]))
        with self.assertRaisesRegex(ValueError,"duplicate reviews"):
            validate_ate_membership(self.pairs,ate)

    def test_check_tokenizer_rejects_slow(self):
        self.tok.is_fast=False
        with self.assertRaisesRegex(ValueError,"fast tokenizer"):
            check_tokenizer(self.tok,max_length=128)

    def test_check_tokenizer_length_bounds(self):
        with self.assertRaisesRegex(ValueError,"max_length"):
            check_tokenizer(self.tok,max_length=600)
        with self.assertRaisesRegex(ValueError,"max_length"):
            check_tokenizer(self.tok,max_length=5)

    def test_pair_tokenizer_review_aspect_order_and_no_truncation(self):
        register_target_markers(self.tok)
        result,stat=tokenize_gold_pairs(self.pairs["train"],self.tok,max_length=128,split="train")
        self.assertEqual(len(result),3)
        self.assertEqual(result[0]["labels"],2)
        self.assertEqual(self.tok.calls[0][:2],("The [unused1] food [unused2] was great , service slow .","food"))
        self.assertEqual(self.tok.calls[0][2:],(True,False,False,True))
        self.assertEqual(result[0]["input_ids"].count(102),2)
        self.assertEqual(stat["over_max_length"],0)

    def test_tokenized_records_contain_only_model_inputs(self):
        register_target_markers(self.tok)
        encoded,_=tokenize_gold_pairs(self.pairs["train"],self.tok,max_length=128,split="train")
        self.assertEqual(set(encoded[0]),{"input_ids","attention_mask","labels"})
        self.assertNotIn("sentiment",encoded[0])

    def test_overlong_refused_not_truncated(self):
        register_target_markers(self.tok)
        self.tok.length_overrides["food"]=129
        with self.assertRaisesRegex(ValueError,"exceed max_length=128"):
            tokenize_gold_pairs(self.pairs["train"],self.tok,max_length=128,split="train")

    def test_all_three_classes_retained_and_ratios(self):
        dist=class_imbalance_report(self.pairs)
        self.assertEqual(dist["class_counts"]["train"],{"negative":1,"neutral":1,"positive":1})
        self.assertEqual(dist["class_proportions"]["train"],{"negative":0.333333,"neutral":0.333333,"positive":0.333333})
        self.assertEqual(dist["suggested_train_balanced_weights_NOT_APPLIED"],{"negative":1,"neutral":1,"positive":1})

    def test_missing_train_class_gets_none_weight(self):
        pairs=fixture(); pairs["train"]=[pairs["train"][0]]
        stats=class_imbalance_report(pairs)
        self.assertIsNone(stats["suggested_train_balanced_weights_NOT_APPLIED"]["negative"])

    def test_full_cli_offline_mock_no_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            audit,ate,ate_data=self.write_fixture(tmp)
            tokenizer=self.tok
            class DummyDataset:
                @staticmethod
                def from_list(rows): return rows
            class DummyDD(dict):
                def save_to_disk(self, path):
                    Path(path).mkdir(parents=True)
                    for s,rows in self.items():
                        (Path(path)/f"{s}.json").write_text(json.dumps(rows),encoding="utf-8")
            ds=ModuleType("datasets")
            ds.Dataset=DummyDataset
            ds.DatasetDict=DummyDD
            ds.load_from_disk=lambda path:ate_data
            tf=ModuleType("transformers")
            class DummyAutoTokenizer:
                @staticmethod
                def from_pretrained(path,**kwargs):
                    assert Path(path)==ate/"tokenizer"
                    assert kwargs=={"local_files_only":True,"use_fast":True}
                    return tokenizer
            tf.AutoTokenizer=DummyAutoTokenizer
            output=Path(tmp)/"atsc_preprocessing"
            stdout=io.StringIO()
            with patch.dict(sys.modules,{"datasets":ds,"transformers":tf}),redirect_stdout(stdout):
                rc=main(["--audit",str(audit),"--ate-data",str(ate),"--output",str(output)])
            self.assertEqual(rc,0)
            summary=json.loads((output/"manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["pair_counts"],{"train":3,"validation":1,"test":1})
            self.assertTrue(summary["held_out_test_kept_separate"])
            self.assertFalse(summary["training_performed"])
            self.assertFalse(summary["model_weights_loaded"])
            self.assertEqual(summary["target_disambiguation"],"span_markers_from_gold_start_end_offsets")
            self.assertEqual(summary["target_markers"],{"[unused1]":2,"[unused2]":3})
            self.assertFalse(summary["vocabulary_resized"])
            self.assertIsNone(summary["model_metrics"])
            self.assertTrue(tokenizer.saved)
            self.assertEqual(len(json.loads((output/"tokenized_splits"/"train.json").read_text())),3)
            self.assertIn("NO MODEL TRAINED",stdout.getvalue())

    def test_marker_span_exact_and_distinguishes_occurrences(self):
        review="service great but service bad"
        self.assertEqual(mark_target_span(review,0,7),
                         "[unused1] service [unused2] great but service bad")
        self.assertEqual(mark_target_span(review,18,25),
                         "service great but [unused1] service [unused2] bad")
        self.assertEqual(review[18:25],"service")

    def test_marker_span_preserves_multiword_aspect(self):
        review="Great customer service, terrible drinks"
        self.assertEqual(mark_target_span(review,6,22),
                         "Great [unused1] customer service [unused2] , terrible drinks")

    def test_marker_rejects_invalid_offsets_and_marker_collision(self):
        with self.assertRaisesRegex(ValueError,"Invalid target span"):
            mark_target_span("service",3,10)
        with self.assertRaisesRegex(ValueError,"reserved target marker"):
            mark_target_span("[unused1] service",10,17)

    def test_marker_registration_refuses_missing_vocab_tokens(self):
        self.tok.vocab.pop(TARGET_END_MARKER)
        with self.assertRaisesRegex(ValueError,"lacks target markers"):
            register_target_markers(self.tok)

    def test_marker_registration_preserves_vocabulary_size(self):
        original_size=len(self.tok)
        self.assertEqual(register_target_markers(self.tok),{"[unused1]":2,"[unused2]":3})
        self.assertEqual(len(self.tok),original_size)

    def test_local_distilbert_fast_tokenizer_recognizes_existing_unused_markers(self):
        """Optional integration test on machine with saved ATE tokenizer."""
        tokenizer_dir = Path("artifacts/ate_preprocessing/tokenizer")
        if not tokenizer_dir.is_dir():
            self.skipTest("Local ATE tokenizer not present in this testing environment")
        try:
            from transformers import AutoTokenizer
        except ImportError:
            self.skipTest("transformers is not installed")
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True, use_fast=True)
        marker_ids = register_target_markers(tokenizer)
        review = mark_target_span("service great but service bad", 18, 25)
        result = tokenizer(review, "service", add_special_tokens=True, padding=False,
                           truncation=False, return_attention_mask=True)
        self.assertEqual(result["input_ids"].count(marker_ids[TARGET_START_MARKER]), 1)
        self.assertEqual(result["input_ids"].count(marker_ids[TARGET_END_MARKER]), 1)

    def test_cli_prevents_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/"output"; output.mkdir()
            (output/"important.txt").write_text("safe",encoding="utf-8")
            with redirect_stderr(io.StringIO()),self.assertRaises(SystemExit) as cm:
                main(["--output",str(output)])
            self.assertEqual(cm.exception.code,2)
            self.assertEqual((output/"important.txt").read_text(),"safe")

    def test_no_model_or_training_imports(self):
        import atsc_preprocessing
        code=Path(atsc_preprocessing.__file__).read_text(encoding="utf-8")
        self.assertNotIn("AutoModelFor",code)
        self.assertNotIn("Trainer(",code)
        self.assertNotIn(".train(",code)
        self.assertNotIn("torch.load(",code)


if __name__ == "__main__":
    unittest.main()
