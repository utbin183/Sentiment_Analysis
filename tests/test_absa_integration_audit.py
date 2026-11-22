"""Phase 4.1 compatibility audit tests: synthetic, offline, no trained weights/data."""
from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import absa_integration_audit as audit


REVIEW = "The food was delicious, but the service was terrible."


def valid_ate(text="food", start=4, end=8, confidence=0.99):
    return [{"text": text, "start": start, "end": end, "confidence": confidence}]


def valid_atsc():
    return {
        "review": REVIEW,
        "aspect": {"text": "food", "start": 4, "end": 8},
        "sentiment": "positive", "label_id": 2,
        "probabilities_softmax_uncalibrated": {"negative": 0.01, "neutral": 0.02, "positive": 0.97},
        "predicted_probability_uncalibrated": 0.97,
    }


def write_json(file, obj):
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(json.dumps(obj), encoding="utf-8")


def fake_artifacts(root: Path):
    ate_model = root / "ate_best_model"
    training = root / "atsc_training"
    atsc_model = training / "best_model"
    data = root / "atsc_preprocessing"
    for model in (ate_model, atsc_model):
        model.mkdir(parents=True)
        for name in ("tokenizer.json", "tokenizer_config.json"):
            (model / name).write_text("{}", encoding="utf-8")
        (model / "model.safetensors").write_bytes(b"same-synthetic-weight-bytes")
    write_json(ate_model / "config.json", {
        "model_type": "distilbert", "vocab_size": 30522,
        "id2label": {str(k): v for k, v in audit.BIO_LABELS.items()},
        "architectures": ["DistilBertForTokenClassification"],
    })
    write_json(atsc_model / "config.json", {
        "model_type": "distilbert", "vocab_size": 30522,
        "id2label": {str(k): v for k, v in audit.LABELS.items()},
        "label2id": {v: k for k, v in audit.LABELS.items()},
        "architectures": ["DistilBertForSequenceClassification"],
    })
    write_json(data / "manifest.json", {
        "status": "PREPROCESSING FINISHED", "tokenizer_name": audit.atsc.MODEL_NAME,
        "vocabulary_resized": False, "target_markers": audit.MARKERS,
        "target_disambiguation": "span_markers_from_gold_start_end_offsets",
        "input_format": audit.INPUT_FORMAT,
        "label_mapping": {str(k): v for k, v in audit.LABELS.items()},
        "truncation": False, "gold_pairs_only": True, "max_length": 128,
        "held_out_test_kept_separate": True, "ate_predictions_used": False,
    })
    write_json(training / "run_summary.json", {
        "status": "TRAINING FINISHED", "training_performed": True, "test_used": False,
        "selection_metric": "eval_macro_f1", "best_model_dir": str(atsc_model),
        "best_checkpoint": str(training / "checkpoints" / "checkpoint-1616"),
    })
    checkpoint = training / "checkpoints" / "checkpoint-1616"
    checkpoint.mkdir(parents=True)
    (checkpoint / "model.safetensors").write_bytes(b"same-synthetic-weight-bytes")
    return ate_model, atsc_model, data, training


class InterfaceTests(unittest.TestCase):
    def test_interface_contracts(self):
        r = audit.check_interface_contracts()
        self.assertEqual(r["status"], "PASS")
        self.assertTrue(r["marker_helper_identity"])

    def test_ate_and_atsc_have_separate_heads(self):
        r = audit.check_interface_contracts()
        self.assertEqual(r["heads"]["ate"], "token_classification")
        self.assertEqual(r["heads"]["atsc"], "sequence_classification")

    def test_ate_signature_break_rejected(self):
        with patch.object(audit.ate.AspectExtractor, "extract", lambda self, text: None):
            with self.assertRaisesRegex(ValueError, "signature"):
                audit.check_interface_contracts()

    def test_atsc_signature_break_rejected(self):
        with patch.object(audit.atsc.ATSCClassifier, "predict", lambda self, x: None):
            with self.assertRaisesRegex(ValueError, "signature"):
                audit.check_interface_contracts()

    def test_marker_helper_identity_break_rejected(self):
        with patch.object(audit.atsc, "mark_target_span", lambda *args: None):
            with self.assertRaisesRegex(ValueError, "reuse"):
                audit.check_interface_contracts()

    def test_ate_bio_mapping_break_rejected(self):
        with patch.object(audit.ate, "ID2LABEL", {0: "negative", 1: "neutral", 2: "positive"}):
            with self.assertRaisesRegex(ValueError, "BIO"):
                audit.check_interface_contracts()

    def test_atsc_sentiment_mapping_break_rejected(self):
        with patch.object(audit.atsc, "ID2LABEL", {0: "O"}):
            with self.assertRaisesRegex(ValueError, "ATSC"):
                audit.check_interface_contracts()

    def test_atsc_marker_mapping_break_rejected(self):
        with patch.object(audit.atsc, "MARKERS", {"[unused1]": 0}):
            with self.assertRaisesRegex(ValueError, "ATSC"):
                audit.check_interface_contracts()

    def test_check_interface_does_not_run_inference(self):
        with patch.object(audit.ate, "load_extractor", Mock(side_effect=AssertionError("loaded"))), \
             patch.object(audit.atsc, "load_classifier", Mock(side_effect=AssertionError("loaded"))):
            audit.check_interface_contracts()


class ATEOutputContractTests(unittest.TestCase):
    def test_valid_aspect(self):
        r = audit.validate_ate_aspects(REVIEW, valid_ate())
        self.assertEqual(r["verified_spans"], [{"text": "food", "start": 4, "end": 8}])

    def test_multiple_spans(self):
        r = audit.validate_ate_aspects(REVIEW, valid_ate() + valid_ate("service", 32, 39, 0.9))
        self.assertEqual(r["aspect_count"], 2)

    def test_empty_predictions_allowed(self):
        self.assertEqual(audit.validate_ate_aspects(REVIEW, [])["aspect_count"], 0)

    def test_empty_review_rejected(self):
        with self.assertRaises(ValueError): audit.validate_ate_aspects(" ", [])

    def test_list_required(self):
        with self.assertRaises(ValueError): audit.validate_ate_aspects(REVIEW, {})

    def test_missing_field_rejected(self):
        with self.assertRaises(ValueError): audit.validate_ate_aspects(REVIEW, [{"text": "food"}])

    def test_offset_not_int_rejected(self):
        for v in (4.0, True, "4"):
            with self.subTest(value=v), self.assertRaises(ValueError):
                audit.validate_ate_aspects(REVIEW, valid_ate(start=v))

    def test_span_out_of_range_rejected(self):
        with self.assertRaises(ValueError): audit.validate_ate_aspects(REVIEW, valid_ate(end=1000))

    def test_aspect_mismatch_rejected(self):
        with self.assertRaises(ValueError): audit.validate_ate_aspects(REVIEW, valid_ate(text="Food"))

    def test_aspects_reverse_order_rejected(self):
        with self.assertRaises(ValueError):
            audit.validate_ate_aspects(REVIEW, valid_ate("service", 32, 39) + valid_ate())

    def test_aspects_overlap_rejected(self):
        with self.assertRaises(ValueError):
            audit.validate_ate_aspects(REVIEW, valid_ate() + valid_ate("ood", 5, 8))

    def test_confidence_nan_rejected(self):
        with self.assertRaises(ValueError): audit.validate_ate_aspects(REVIEW, valid_ate(confidence=float("nan")))

    def test_confidence_bool_rejected(self):
        with self.assertRaises(ValueError): audit.validate_ate_aspects(REVIEW, valid_ate(confidence=True))

    def test_confidence_out_of_range_rejected(self):
        with self.assertRaises(ValueError): audit.validate_ate_aspects(REVIEW, valid_ate(confidence=1.01))

    def test_unicode_python_offsets(self):
        r = "Café food is great"
        s = r.index("food")
        self.assertEqual(audit.validate_ate_aspects(r, valid_ate("food", s, s+4))["aspect_count"], 1)


class ATSCOutputContractTests(unittest.TestCase):
    def test_valid_output(self):
        self.assertEqual(audit.validate_atsc_result(REVIEW, "food", 4, 8, valid_atsc())["predicted_class"], "positive")

    def test_missing_field(self):
        r = valid_atsc(); r.pop("label_id")
        with self.assertRaises(ValueError): audit.validate_atsc_result(REVIEW,"food",4,8,r)

    def test_wrong_review(self):
        r = valid_atsc(); r["review"] = "different"
        with self.assertRaises(ValueError): audit.validate_atsc_result(REVIEW,"food",4,8,r)

    def test_wrong_aspect(self):
        r = valid_atsc(); r["aspect"]["start"] = 5
        with self.assertRaises(ValueError): audit.validate_atsc_result(REVIEW,"food",4,8,r)

    def test_wrong_label_name(self):
        r = valid_atsc(); r["sentiment"] = "negative"
        with self.assertRaises(ValueError): audit.validate_atsc_result(REVIEW,"food",4,8,r)

    def test_bool_label_rejected(self):
        r = valid_atsc(); r["label_id"] = True
        with self.assertRaises(ValueError): audit.validate_atsc_result(REVIEW,"food",4,8,r)

    def test_missing_class_probabilities(self):
        r = valid_atsc(); r["probabilities_softmax_uncalibrated"].pop("neutral")
        with self.assertRaises(ValueError): audit.validate_atsc_result(REVIEW,"food",4,8,r)

    def test_probabilities_do_not_sum_one(self):
        r = valid_atsc(); r["probabilities_softmax_uncalibrated"]["positive"] = 0.6
        with self.assertRaises(ValueError): audit.validate_atsc_result(REVIEW,"food",4,8,r)

    def test_probabilities_argmax_conflict(self):
        r = valid_atsc(); r["probabilities_softmax_uncalibrated"] = {"negative":.95,"neutral":.03,"positive":.02}
        r["predicted_probability_uncalibrated"] = .02
        with self.assertRaises(ValueError): audit.validate_atsc_result(REVIEW,"food",4,8,r)

    def test_predicted_probability_conflict(self):
        r = valid_atsc(); r["predicted_probability_uncalibrated"] = .85
        with self.assertRaises(ValueError): audit.validate_atsc_result(REVIEW,"food",4,8,r)

    def test_softmax_uncalibrated_explicit(self):
        self.assertFalse(audit.validate_atsc_result(REVIEW,"food",4,8,valid_atsc())["softmax_is_calibrated"])


class TextAndOffsetTests(unittest.TestCase):
    def test_9_synthetic_fixtures(self):
        r = audit.audit_span_fixtures()
        self.assertEqual(r["synthetic_cases"],9)
        self.assertEqual(r["status"], "PASS")

    def test_punctuation_and_spacing_drift_exposed(self):
        r = audit.audit_span_fixtures()
        self.assertGreater(r["raw_text_differs_from_space_joined_ate_tokens_in_cases"], 0)

    def test_repeated_occurrences_distinct(self):
        r = audit.audit_span_fixtures()
        x = {c["case"]:c["marked_text"] for c in r["cases"]}
        self.assertNotEqual(x["repeated_first"], x["repeated_last"])

    def test_no_rewrite_to_normalized_text(self):
        r = audit.audit_span_fixtures()
        x = {c["case"]:c for c in r["cases"]}
        self.assertEqual(x["extra_whitespace"]["start"], 6)

    def test_multiword_aspect(self):
        r = audit.audit_span_fixtures()
        self.assertEqual(next(c for c in r["cases"] if c["case"] == "multi_word")["aspect"], "dining room")

    def test_span_wrong_occurrence_fails(self):
        r = "service improved, then service declined"
        with self.assertRaises(ValueError): audit.atsc.validate_span(r,"service",1,8)

    def test_finding_levels(self):
        with self.assertRaises(ValueError): audit._finding("BAD", "x", "x")


class SavedArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.paths = fake_artifacts(Path(self.temp.name))

    def test_metadata_valid(self):
        r = audit.check_saved_artifacts(*self.paths)
        self.assertEqual(r["best_checkpoint"], "checkpoint-1616")
        self.assertFalse(r["weights_loaded"])

    def test_missing_ate_model_rejected(self):
        (self.paths[0] / "config.json").unlink()
        with self.assertRaises(FileNotFoundError): audit.check_saved_artifacts(*self.paths)

    def test_wrong_ate_head_rejected(self):
        p = self.paths[0] / "config.json"; v = json.loads(p.read_text()); v["architectures"] = ["DistilBertForSequenceClassification"]; write_json(p,v)
        with self.assertRaisesRegex(ValueError, "architecture"):
            audit.check_saved_artifacts(*self.paths)

    def test_wrong_atsc_head_rejected(self):
        p = self.paths[1] / "config.json"; v = json.loads(p.read_text()); v["architectures"] = ["DistilBertForTokenClassification"]; write_json(p,v)
        with self.assertRaisesRegex(ValueError, "architecture"):
            audit.check_saved_artifacts(*self.paths)

    def test_wrong_ate_bio_mapping(self):
        p = self.paths[0] / "config.json"; v = json.loads(p.read_text()); v["id2label"] = {"0":"positive"}; write_json(p,v)
        with self.assertRaisesRegex(ValueError,"label"):
            audit.check_saved_artifacts(*self.paths)

    def test_wrong_atsc_label2id(self):
        p = self.paths[1] / "config.json"; v = json.loads(p.read_text()); v["label2id"] = {"positive":1}; write_json(p,v)
        with self.assertRaisesRegex(ValueError,"label2id"):
            audit.check_saved_artifacts(*self.paths)

    def test_unfinished_training_rejected(self):
        p = self.paths[3] / "run_summary.json"; v = json.loads(p.read_text()); v["status"] = "RUNNING"; write_json(p,v)
        with self.assertRaisesRegex(ValueError,"training"):
            audit.check_saved_artifacts(*self.paths)

    def test_test_used_rejected(self):
        p = self.paths[3] / "run_summary.json"; v = json.loads(p.read_text()); v["test_used"] = True; write_json(p,v)
        with self.assertRaisesRegex(ValueError,"test"):
            audit.check_saved_artifacts(*self.paths)

    def test_invalid_marker_rejected(self):
        p = self.paths[2] / "manifest.json"; v = json.loads(p.read_text()); v["target_markers"] = {}; write_json(p,v)
        with self.assertRaisesRegex(ValueError,"marker"):
            audit.check_saved_artifacts(*self.paths)

    def test_test_not_held_out_rejected(self):
        p = self.paths[2] / "manifest.json"; v = json.loads(p.read_text()); v["held_out_test_kept_separate"] = False; write_json(p,v)
        with self.assertRaisesRegex(ValueError,"provenance"):
            audit.check_saved_artifacts(*self.paths)

    def test_weights_sha_identical(self):
        result = audit.verify_selected_checkpoint(self.paths[1], self.paths[3])
        self.assertTrue(result["identical"])
        self.assertEqual(len(result["sha256"]),64)

    def test_weights_sha_different_rejected(self):
        (self.paths[1] / "model.safetensors").write_bytes(b"different")
        with self.assertRaisesRegex(ValueError,"differ"):
            audit.verify_selected_checkpoint(self.paths[1], self.paths[3])

    def test_checkpoint_missing_rejected(self):
        (self.paths[3] / "checkpoints" / "checkpoint-1616" / "model.safetensors").unlink()
        with self.assertRaises(FileNotFoundError): audit.verify_selected_checkpoint(self.paths[1], self.paths[3])

    def test_training_summary_windows_checkpoint_paths(self):
        p=self.paths[3] / "run_summary.json"
        v=json.loads(p.read_text())
        v["best_model_dir"] = r"artifacts\atsc_training\best_model"
        v["best_checkpoint"] = r"artifacts\atsc_training\checkpoints\checkpoint-1616"
        write_json(p,v)
        self.assertEqual(audit.check_saved_artifacts(*self.paths)["best_checkpoint"], "checkpoint-1616")
        self.assertTrue(audit.verify_selected_checkpoint(self.paths[1],self.paths[3])["identical"])

    def test_weights_untouched(self):
        src = self.paths[1] / "model.safetensors"
        before = src.read_bytes()
        audit.verify_selected_checkpoint(self.paths[1], self.paths[3])
        self.assertEqual(src.read_bytes(), before)

    def test_report_ready_with_risks(self):
        r = audit.build_report(ate_model=self.paths[0],atsc_model=self.paths[1],atsc_data=self.paths[2],training_dir=self.paths[3])
        self.assertEqual(r["status"],"AUDIT READY WITH RISKS")
        self.assertFalse(r["pipeline_created"])

    def test_report_with_sha_readonly(self):
        r = audit.build_report(ate_model=self.paths[0],atsc_model=self.paths[1],atsc_data=self.paths[2],training_dir=self.paths[3],verify_checkpoint=True)
        self.assertEqual(r["checks"]["checkpoint_integrity"]["status"],"PASS")

    def test_missing_artifact_blocks_report(self):
        (self.paths[1] / "config.json").unlink()
        r = audit.build_report(ate_model=self.paths[0],atsc_model=self.paths[1],atsc_data=self.paths[2],training_dir=self.paths[3])
        self.assertEqual(r["status"],"AUDIT BLOCKED")

    def test_missing_artifact_only_for_synthetic_mode(self):
        (self.paths[1] / "config.json").unlink()
        r = audit.build_report(ate_model=self.paths[0],atsc_model=self.paths[1],atsc_data=self.paths[2],training_dir=self.paths[3],strict_artifacts=False)
        self.assertEqual(r["status"],"AUDIT READY WITH RISKS")
        self.assertEqual(r["checks"]["saved_artifacts"]["status"],"NOT_CHECKED")

    def test_simulated_tokenizer_check(self):
        check = {"status":"PASS","budget_divergence_cases":["boundary_stress"]}
        with patch.object(audit,"check_tokenizer_pair",return_value=check):
            r=audit.build_report(ate_model=self.paths[0],atsc_model=self.paths[1],atsc_data=self.paths[2],training_dir=self.paths[3],check_tokenizers=True)
        self.assertTrue(any(f["code"]=="TOKEN_BUDGET_DIVERGENCE_REPRODUCED" for f in r["findings"]))

    def test_tokenizer_failure_blocks_report(self):
        with patch.object(audit,"check_tokenizer_pair",side_effect=ValueError("bad tokenizer")):
            r=audit.build_report(ate_model=self.paths[0],atsc_model=self.paths[1],atsc_data=self.paths[2],training_dir=self.paths[3],check_tokenizers=True)
        self.assertEqual(r["status"],"AUDIT BLOCKED")

    def test_model_loading_failure_blocks_report(self):
        with patch.object(audit,"smoke_test_models",side_effect=RuntimeError("bad checkpoint")):
            r=audit.build_report(ate_model=self.paths[0],atsc_model=self.paths[1],atsc_data=self.paths[2],training_dir=self.paths[3],check_models=True)
        self.assertEqual(r["status"],"AUDIT BLOCKED")


class IndependentSmokeTests(unittest.TestCase):
    def test_model_smoke_does_not_feed_ate_prediction_to_atsc(self):
        ate_fake = Mock()
        ate_fake.extract.return_value = valid_ate()
        atsc_fake = Mock()
        atsc_fake.predict.return_value = valid_atsc()
        other = valid_atsc()
        other["aspect"] = {"text":"service","start":32,"end":39}
        other["sentiment"]="negative"; other["label_id"]=0
        other["probabilities_softmax_uncalibrated"]={"negative":.97,"neutral":.02,"positive":.01}
        other["predicted_probability_uncalibrated"] = .97
        atsc_fake.predict.side_effect=[valid_atsc(),other]
        with patch.object(audit.ate,"load_extractor",return_value=ate_fake), \
             patch.object(audit.atsc,"load_classifier",return_value=atsc_fake):
            r=audit.smoke_test_models(Path("ate"),Path("atsc"),Path("manifest"),"cpu")
        self.assertFalse(r["ate_predictions_passed_to_atsc"])
        self.assertEqual(atsc_fake.predict.call_args_list[0].args,(REVIEW,"food",4,8))
        self.assertEqual(atsc_fake.predict.call_args_list[1].args,(REVIEW,"service",32,39))

    def test_bad_device_rejected_before_model_load(self):
        with self.assertRaises(ValueError):
            audit.smoke_test_models(Path("x"),Path("y"),Path("z"),"tpu")


class CLITests(unittest.TestCase):
    def test_cli_help(self):
        p=audit.build_parser()
        text=p.format_help()
        self.assertIn("--dry-run",text)
        self.assertIn("--check-model-loading",text)

    def test_cli_dry_run_writes_nothing(self):
        with TemporaryDirectory() as tmp:
            output=Path(tmp)/"report.json"
            with redirect_stdout(io.StringIO()) as buffer:
                code=audit.main(["--dry-run","--allow-missing-artifacts","--output",str(output)])
            self.assertEqual(code,0)
            self.assertFalse(output.exists())
            self.assertIn('"pipeline_created": false',buffer.getvalue())

    def test_cli_blocks_without_artifacts(self):
        with TemporaryDirectory() as tmp:
            output=Path(tmp)/"report.json"
            with redirect_stdout(io.StringIO()):
                rc=audit.main(["--dry-run","--ate-model",str(Path(tmp)/"missing"),
                               "--atsc-model",str(Path(tmp)/"missing2"),
                               "--atsc-data",str(Path(tmp)/"missing3"),
                               "--training",str(Path(tmp)/"missing4"), "--output",str(output)])
            self.assertEqual(rc,2)
            self.assertFalse(output.exists())

    def test_cli_saved_report(self):
        with TemporaryDirectory() as tmp:
            output=Path(tmp)/"report.json"
            with redirect_stdout(io.StringIO()):
                rc=audit.main(["--allow-missing-artifacts","--output",str(output)])
            self.assertEqual(rc,0)
            self.assertTrue(output.is_file())
            self.assertFalse(json.loads(output.read_text())["training_performed"])

    def test_cli_overwrite_refused(self):
        with TemporaryDirectory() as tmp:
            output=Path(tmp)/"report.json"; output.write_text("keep",encoding="utf-8")
            with self.assertRaises(SystemExit):
                audit.main(["--output",str(output)])
            self.assertEqual(output.read_text(),"keep")

    def test_cli_dry_run_and_models_refused(self):
        with self.assertRaises(SystemExit):
            audit.main(["--dry-run","--check-model-loading"])

    def test_cli_default_no_model_weights(self):
        p=audit.build_parser().parse_args([])
        self.assertFalse(p.check_model_loading)
        self.assertFalse(p.check_tokenizers)
        self.assertFalse(p.verify_checkpoint)

    def test_no_train_no_test_loader_in_audit(self):
        source=inspect_source()
        for forbidden in ("Trainer(", "trainer.train(", "DatasetDict(", "load_from_disk("):
            self.assertNotIn(forbidden,source)


def inspect_source():
    import inspect
    return inspect.getsource(audit)


class FakeATEFastTokenizer:
    is_fast = True

    def __len__(self):
        return 30522

    def __call__(self, batch, **kwargs):
        if not isinstance(batch,list) or len(batch) != 1 or not isinstance(batch[0],list):
            raise ValueError("Expected ATE word tokens")
        return {"input_ids": [[101] + [10] * len(batch[0]) + [102]]}


class FakeATSCFastTokenizer:
    is_fast = True
    model_max_length = 512
    padding_side = "right"
    pad_token_id = 0
    additional_special_tokens = ["[unused1]", "[unused2]"]

    def __len__(self):
        return 30522

    def convert_tokens_to_ids(self, name):
        return {"[unused1]":2,"[unused2]":3}.get(name,10)

    def num_special_tokens_to_add(self,pair=False):
        return 3 if pair else 2

    def __call__(self, marked_review, aspect, **kwargs):
        tokens = marked_review.split() + ["[SEP]"] + aspect.split()
        ids = [101] + [self.convert_tokens_to_ids(t) for t in tokens] + [102]
        return {"input_ids":ids,"attention_mask":[1]*len(ids)}


class TokenizerReadOnlyTests(unittest.TestCase):
    def test_real_tokenizer_logic_simulated_without_weights(self):
        fake_module = SimpleNamespace(AutoTokenizer=SimpleNamespace(
            from_pretrained=Mock(side_effect=[FakeATEFastTokenizer(),FakeATSCFastTokenizer()])))
        with patch.dict(sys.modules,{"transformers":fake_module}), \
             patch.object(audit.atsc,"read_preprocessing_manifest",return_value=128):
            r=audit.check_tokenizer_pair(Path("ate"),Path("atsc"),Path("manifest"))
        self.assertEqual(r["status"],"PASS")
        self.assertEqual(len(r["scenarios"]),10)
        self.assertIn("boundary_stress", r["budget_divergence_cases"])
        self.assertFalse(r["model_weights_loaded"])

    def test_tokenizer_markers_must_match(self):
        bad=FakeATSCFastTokenizer()
        bad.additional_special_tokens = []
        fake_module=SimpleNamespace(AutoTokenizer=SimpleNamespace(
            from_pretrained=Mock(side_effect=[FakeATEFastTokenizer(),bad])))
        with patch.dict(sys.modules,{"transformers":fake_module}), \
             patch.object(audit.atsc,"read_preprocessing_manifest",return_value=128):
            with self.assertRaisesRegex(ValueError,"registered"):
                audit.check_tokenizer_pair(Path("ate"),Path("atsc"),Path("manifest"))


if __name__ == "__main__":
    unittest.main()
