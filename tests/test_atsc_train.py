"""Offline tests for Step 3.3. NO Hugging Face download or real training.

All datasets/logits in this file are SYNTHETIC. Reported unit-test scores are
NOT model metrics. The held-out test split is never accessed by the CLI mocks.
"""
from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from atsc_train import (
    DATASET_NAME, ID2LABEL, LABEL2ID, MODEL_NAME, TARGET_MARKERS,
    SPLITS_FOR_TRAINING, build_parser, check_tokenizer,
    classification_metrics, load_two_splits, main, make_compute_metrics,
    output_can_be_created, read_and_validate_manifest, training_plan,
    validate_encoded_rows,
)


def sample_manifest():
    return {
        "status":"PREPROCESSING FINISHED",
        "source_dataset":DATASET_NAME, "tokenizer_name":MODEL_NAME,
        "gold_pairs_only":True, "ate_predictions_used":False,
        "held_out_test_kept_separate":True,
        "training_performed":False, "model_weights_loaded":False,
        "review_overlap_between_splits":0, "vocabulary_resized":False,
        "truncation":False,
        "target_disambiguation":"span_markers_from_gold_start_end_offsets",
        "target_markers":TARGET_MARKERS.copy(),
        "label_mapping":{str(k): v for k,v in ID2LABEL.items()},
        "input_format":"[CLS] review with [unused1] target [unused2] [SEP] aspect [SEP]",
        "max_length":128,
        "pair_counts":{"train":3,"validation":2,"test":3},
        "review_counts":{"train":3,"validation":2,"test":3},
        "imbalance":{"class_counts":{
            "train":{"negative":1,"neutral":1,"positive":1},
            "validation":{"negative":0,"neutral":1,"positive":1},
            "test":{"negative":1,"neutral":1,"positive":1},
        }},
        "encoded_token_lengths":{
            "train":{"over_max_length":0},
            "validation":{"over_max_length":0},
            "test":{"over_max_length":0},
        },
    }


def row(i, y):
    return {
        "input_ids":[101, 10+i, 2, 50+i, 3, 102, 50+i, 102],
        "attention_mask":[1]*8,
        "labels":y,
    }


def sample_data():
    return [row(i, i) for i in range(3)], [row(10, 1), row(11, 2)]


class FakeTokenizer:
    is_fast=True
    pad_token_id=0
    model_max_length=512
    additional_special_tokens=["[unused1]", "[unused2]"]
    def __len__(self):
        return 30522
    def convert_tokens_to_ids(self, name):
        return TARGET_MARKERS.get(name, 100)
    def save_pretrained(self, path):
        (Path(path)/"tokenizer_config.json").write_text("{}", encoding="utf-8")


def write_artifacts(root):
    data=root/"data"
    for split in ("train", "validation", "test"):
        (data/"tokenized_splits"/split).mkdir(parents=True)
    (data/"tokenizer").mkdir()
    (data/"manifest.json").write_text(json.dumps(sample_manifest()), encoding="utf-8")
    return data


def mock_source_modules(root, calls, *, allow_train=False):
    train, val=sample_data()
    def loader(path):
        split=Path(path).name
        calls.append(split)
        if split == "test":
            raise AssertionError("HELD-OUT TEST WAS OPENED")
        return {"train":train, "validation":val}[split]
    fake_datasets=types.ModuleType("datasets")
    fake_datasets.load_from_disk=loader
    fake_transformers=types.ModuleType("transformers")
    fake_transformers.AutoTokenizer=types.SimpleNamespace(
        from_pretrained=lambda path, **kw: FakeTokenizer())
    if allow_train:
        class FakeTorch(types.ModuleType):
            cuda=types.SimpleNamespace(is_available=lambda:False)
        fake_torch=FakeTorch("torch")
        class FakeModel:
            def get_input_embeddings(self):
                return types.SimpleNamespace(num_embeddings=30522)
        fake_transformers.AutoModelForSequenceClassification=types.SimpleNamespace(
            from_pretrained=lambda *a, **kw:FakeModel())
        fake_transformers.DataCollatorWithPadding=lambda **kwargs:object()
        fake_transformers.EarlyStoppingCallback=lambda **kwargs:object()
        fake_transformers.TrainingArguments=lambda **kwargs:types.SimpleNamespace(**kwargs)
        fake_transformers.set_seed=lambda seed:None
        class FakeTrainer:
            def __init__(self, **kwargs):
                self.params=kwargs
                assert kwargs["args"].metric_for_best_model == "eval_macro_f1"
                assert len(kwargs["train_dataset"]) == 3
                assert len(kwargs["eval_dataset"]) == 2
                self.state=types.SimpleNamespace(
                    best_model_checkpoint=str(root/"out"/"checkpoints"/"checkpoint-SYNTHETIC"),
                    best_metric=0.5,
                    log_history=[{"epoch":1,"eval_macro_f1":0.5,"synthetic":True}])
            def train(self):
                return types.SimpleNamespace(metrics={"train_loss":0.9})
            def evaluate(self, **kwargs):
                return {"eval_macro_f1":0.5, "eval_loss":0.7}
            def save_model(self, path):
                (Path(path)/"model.safetensors").write_bytes(b"FAKE")
        fake_transformers.Trainer=FakeTrainer
        return {"datasets":fake_datasets,"transformers":fake_transformers,"torch":fake_torch}
    return {"datasets":fake_datasets,"transformers":fake_transformers}


class Step33Tests(unittest.TestCase):
    def test_mapping_fixed(self):
        self.assertEqual(ID2LABEL, {0:"negative",1:"neutral",2:"positive"})
        self.assertEqual(LABEL2ID["neutral"],1)
    def test_training_only_two_splits(self):
        self.assertEqual(SPLITS_FOR_TRAINING,("train","validation"))
    def test_valid_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            file=Path(tmp)/"manifest.json"
            file.write_text(json.dumps(sample_manifest()),encoding="utf-8")
            self.assertEqual(read_and_validate_manifest(file)["max_length"],128)
    def test_missing_manifest_rejected(self):
        with self.assertRaises(FileNotFoundError):
            read_and_validate_manifest(Path("totally-not-there-13984.json"))
    def assert_manifest_bad(self,change,match):
        m=sample_manifest(); change(m)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"manifest.json"
            p.write_text(json.dumps(m),encoding="utf-8")
            with self.assertRaisesRegex(ValueError,match):
                read_and_validate_manifest(p)
    def test_invalid_mapping(self):
        self.assert_manifest_bad(lambda m:m["label_mapping"].update({"1":"positive"}),"polarity mapping")
    def test_negative_one_cannot_be_class(self):
        self.assert_manifest_bad(lambda m:m["label_mapping"].update({"-1":"negative"}),"polarity mapping")
    def test_wrong_marker_id(self):
        self.assert_manifest_bad(lambda m:m["target_markers"].update({"[unused2]":120}),"target marker")
    def test_review_leak_manifest(self):
        self.assert_manifest_bad(lambda m:m.update(review_overlap_between_splits=1),"leakage")
    def test_gold_must_be_used(self):
        self.assert_manifest_bad(lambda m:m.update(gold_pairs_only=False),"GOLD")
    def test_ate_predictions_not_allowed(self):
        self.assert_manifest_bad(lambda m:m.update(ate_predictions_used=True),"GOLD")
    def test_test_split_separate_required(self):
        self.assert_manifest_bad(lambda m:m.update(held_out_test_kept_separate=False),"test")
    def test_pretrained_tokenizer_vocab_not_resized(self):
        self.assert_manifest_bad(lambda m:m.update(vocabulary_resized=True),"vocabulary")
    def test_truncation_rejected(self):
        self.assert_manifest_bad(lambda m:m.update(truncation=True),"truncate")
    def test_max_length_bound(self):
        self.assert_manifest_bad(lambda m:m.update(max_length=513),"512")
    def test_sentiment_counts_versus_pairs(self):
        self.assert_manifest_bad(lambda m:m["imbalance"]["class_counts"]["train"].update(positive=4),"Class counts")
    def test_missing_training_class(self):
        self.assert_manifest_bad(lambda m:m["imbalance"]["class_counts"]["train"].update(negative=0, positive=2),"All 3")
    def test_no_test_data_opened(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=write_artifacts(Path(tmp)); calls=[]
            train,val=load_two_splits(d,lambda path: calls.append(Path(path).name) or sample_data()[0 if Path(path).name=="train" else 1])
            self.assertEqual(calls,["train","validation"])
            self.assertEqual((len(train),len(val)),(3,2))
    def test_missing_train_dir_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                load_two_splits(Path(tmp),lambda _:[])
    def test_valid_rows_and_counts(self):
        tr,va=sample_data()
        outcome=validate_encoded_rows(tr,va,sample_manifest(),vocab_size=30522)
        self.assertEqual(outcome["class_distribution"]["train"],{"negative":1,"neutral":1,"positive":1})
        self.assertEqual(outcome["encoded_train_validation_overlap"],0)
    def test_bad_label_rejected(self):
        tr,va=sample_data();tr[0]["labels"]=-1
        with self.assertRaisesRegex(ValueError,"sentiment label"):
            validate_encoded_rows(tr,va,sample_manifest(),vocab_size=30522)
    def test_bool_label_rejected(self):
        tr,va=sample_data();tr[0]["labels"]=True
        with self.assertRaisesRegex(ValueError,"sentiment label"):
            validate_encoded_rows(tr,va,sample_manifest(),vocab_size=30522)
    def test_marker_missing_rejected(self):
        tr,va=sample_data();tr[0]["input_ids"][2]=33
        with self.assertRaisesRegex(ValueError,"marker"):
            validate_encoded_rows(tr,va,sample_manifest(),vocab_size=30522)
    def test_marker_wrong_order_rejected(self):
        tr,va=sample_data();tr[0]["input_ids"][2],tr[0]["input_ids"][4] = 3,2
        with self.assertRaisesRegex(ValueError,"marker"):
            validate_encoded_rows(tr,va,sample_manifest(),vocab_size=30522)
    def test_overlong_rejected(self):
        tr,va=sample_data();tr[0]["input_ids"] = tr[0]["input_ids"] + [5]*200;tr[0]["attention_mask"]=[1]*208
        with self.assertRaisesRegex(ValueError,"sequence length"):
            validate_encoded_rows(tr,va,sample_manifest(),vocab_size=30522)
    def test_invalid_attention_mask(self):
        tr,va=sample_data();tr[0]["attention_mask"][0]=0
        with self.assertRaisesRegex(ValueError,"attention_mask"):
            validate_encoded_rows(tr,va,sample_manifest(),vocab_size=30522)
    def test_vocab_range_rejected(self):
        tr,va=sample_data();tr[0]["input_ids"][1]=30522
        with self.assertRaisesRegex(ValueError,"token ID"):
            validate_encoded_rows(tr,va,sample_manifest(),vocab_size=30522)
    def test_extra_dataset_column_rejected(self):
        tr,va=sample_data();tr[0]["text"]="hidden"
        with self.assertRaisesRegex(ValueError,"unexpected fields"):
            validate_encoded_rows(tr,va,sample_manifest(),vocab_size=30522)
    def test_exact_encoded_cross_split_leakage(self):
        tr,va=sample_data();va[0]=tr[0].copy()
        with self.assertRaisesRegex(ValueError,"LEAKAGE"):
            validate_encoded_rows(tr,va,sample_manifest(),vocab_size=30522)
    def test_duplicate_train_input_rejected(self):
        tr,va=sample_data();tr[1]=tr[0].copy()
        with self.assertRaisesRegex(ValueError,"duplicated"):
            validate_encoded_rows(tr,va,sample_manifest(),vocab_size=30522)
    def test_wrong_distribution_rejected(self):
        tr,va=sample_data();tr[1]["labels"]=0
        with self.assertRaisesRegex(ValueError,"distribution"):
            validate_encoded_rows(tr,va,sample_manifest(),vocab_size=30522)
    def test_valid_tokenizer(self):
        check_tokenizer(FakeTokenizer(),sample_manifest())
    def test_slow_tokenizer_rejected(self):
        t=FakeTokenizer();t.is_fast=False
        with self.assertRaisesRegex(ValueError,"FAST"):
            check_tokenizer(t,sample_manifest())
    def test_marker_not_registered_rejected(self):
        t=FakeTokenizer();t.additional_special_tokens=[]
        with self.assertRaisesRegex(ValueError,"registered"):
            check_tokenizer(t,sample_manifest())
    def test_tokenizer_vocab_resize_rejected(self):
        t=FakeTokenizer();t.__len__=lambda:40000 # instance override does not affect len special lookup
        class Bigger(FakeTokenizer):
            def __len__(self):return 40000
        with self.assertRaisesRegex(ValueError,"resized"):
            check_tokenizer(Bigger(),sample_manifest())
    def test_metrics_all_correct(self):
        r=classification_metrics([0,1,2],[0,1,2])
        self.assertEqual((r["macro_f1"],r["accuracy"]),(1,1))
    def test_metrics_macro_vs_accuracy(self):
        r=classification_metrics([0,1,2],[0,0,2])
        self.assertAlmostEqual(r["macro_f1"],5/9)
        self.assertAlmostEqual(r["accuracy"],2/3)
        self.assertEqual(r["neutral_support"],1)
    def test_metrics_all_positive(self):
        r=classification_metrics([0,1,2,2],[2,2,2,2])
        self.assertEqual(r["negative_f1"],0)
        self.assertEqual(r["neutral_f1"],0)
        self.assertAlmostEqual(r["positive_f1"],2/3)
        self.assertAlmostEqual(r["macro_f1"],2/9)
    def test_metrics_zero_samples_rejected(self):
        with self.assertRaises(ValueError):classification_metrics([],[])
    def test_metrics_invalid_class_rejected(self):
        with self.assertRaises(ValueError):classification_metrics([0],[-1])
    def test_metric_callback_logits(self):
        import numpy as np
        fn=make_compute_metrics()
        scores=fn((np.array([[1.,0,0],[0,10.,0],[0,0,3.]]),np.array([0,1,2])))
        self.assertEqual(scores["macro_f1"],1)
    def test_metric_callback_wrong_shape(self):
        import numpy as np
        with self.assertRaises(ValueError):
            make_compute_metrics()((np.zeros((2,2)),np.zeros(2)))
    def test_metric_callback_non_finite(self):
        import numpy as np
        with self.assertRaises(ValueError):
            make_compute_metrics()((np.array([[float("nan"),0,0]]),np.array([0])))
    def test_plan_uses_validation_macro_f1(self):
        p=training_plan(output="out/checkpoints",epochs=8,batch_size=8,eval_batch_size=16,
                        lr=2e-5, weight_decay=0.01,patience=2,seed=42,fp16=False,device="auto")
        self.assertEqual((p["eval_strategy"],p["save_strategy"]),("epoch","epoch"))
        self.assertEqual(p["metric_for_best_model"],"eval_macro_f1")
        self.assertTrue(p["load_best_model_at_end"])
        self.assertTrue(p["greater_is_better"])
        self.assertEqual(p["report_to"],"none")
    def test_cpu_plan_no_fp16(self):
        p=training_plan(output="z",epochs=1,batch_size=4,eval_batch_size=8,
                        lr=2e-5,weight_decay=0.01,patience=1,seed=42,fp16=False,device="cpu")
        self.assertTrue(p["use_cpu"])
        with self.assertRaises(ValueError):
            training_plan(output="z",epochs=1,batch_size=4,eval_batch_size=8,
                          lr=2e-5,weight_decay=0.01,patience=1,seed=42,fp16=True,device="cpu")
    def test_bad_hyperparameter_rejected(self):
        with self.assertRaises(ValueError):
            training_plan(output="z",epochs=0,batch_size=4,eval_batch_size=8,
                          lr=2e-5,weight_decay=0.01,patience=1,seed=42,fp16=False,device="auto")
    def test_output_directory_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            o=Path(tmp)/"out";o.mkdir();(o/"file").write_text("not overwrite")
            with self.assertRaisesRegex(ValueError,"not empty"):
                output_can_be_created(o)
    def test_parser_dry_run_and_defaults(self):
        args=build_parser().parse_args(["--dry-run"])
        self.assertTrue(args.dry_run)
        self.assertEqual(args.data,"artifacts/atsc_preprocessing")
    def test_cli_dry_run_no_model_or_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=write_artifacts(root);calls=[]
            with patch.dict("sys.modules",mock_source_modules(root,calls)), redirect_stdout(StringIO()) as out:
                rc=main(["--data",str(data),"--output",str(root/"out"),"--dry-run"])
            self.assertEqual(rc,0)
            self.assertEqual(calls,["train","validation"])
            self.assertIn("PRE-FLIGHT PASSED",out.getvalue())
            self.assertIn('"training_performed": false',out.getvalue())
            self.assertFalse((root/"out").exists())
    def test_cli_mock_training_writes_summary_and_tokenizer(self):
        """SYNTHETIC fake Trainer (never touches model weights / no real F1)."""
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=write_artifacts(root);calls=[]
            modules=mock_source_modules(root,calls,allow_train=True)
            with patch.dict("sys.modules",modules),redirect_stdout(StringIO()) as out:
                rc=main(["--data",str(data),"--output",str(root/"out"),"--device","cpu"])
            self.assertEqual(rc,0)
            self.assertEqual(calls,["train","validation"])
            summary=json.loads((root/"out"/"run_summary.json").read_text(encoding="utf-8"))
            self.assertFalse(summary["test_used"])
            self.assertEqual(summary["selection_metric"],"eval_macro_f1")
            self.assertEqual(summary["best_validation_macro_f1"],0.5)
            self.assertTrue((root/"out"/"best_model"/"model.safetensors").exists())
            self.assertTrue((root/"out"/"best_model"/"tokenizer_config.json").exists())
            self.assertTrue((root/"out"/"training_history.json").exists())
            self.assertIn("TRAINING FINISHED",out.getvalue())
    def test_tensorflow_backend_disabled_early(self):
        import atsc_train
        self.assertEqual(atsc_train.os.environ["USE_TF"],"0")
    def test_scope_does_not_include_test_loader(self):
        import inspect, atsc_train
        src=inspect.getsource(atsc_train.load_two_splits)
        self.assertNotIn("load_from_disk(str(data_dir / 'tokenized_splits'))",src)
        self.assertNotIn('"test"', src)


if __name__ == "__main__":
    unittest.main()
