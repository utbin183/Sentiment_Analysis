"""Phase 4.2 adapter tests. All review/aspects are SYNTHETIC.

No model weights, ATE/ATSC inference, training datasets, or test split opened.
"""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch, Mock

import absa_adapter as adapter


REVIEW = "The food was delicious, but the service was terrible."


def sample(text="food", start=4, end=8, confidence=.987):
    return {"text": text, "start": start, "end": end, "confidence": confidence}


def payload(review=REVIEW, aspects=None):
    return {"review": review, "aspects": [sample()] if aspects is None else aspects}


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
        self.calls = []
        self.forced_ids = None
        self.raise_error = False

    def __len__(self):
        return 30522

    def convert_tokens_to_ids(self, token):
        return self.vocab.get(token, 100)

    def num_special_tokens_to_add(self, pair=False):
        return 3 if pair else 2

    def __call__(self, review, aspect, **kwargs):
        self.calls.append((review, aspect, kwargs))
        if self.raise_error:
            raise ValueError("synthetic tokenizer error")
        if self.forced_ids is not None:
            ids = list(self.forced_ids)
        else:
            def encode(txt):
                output = []
                for token in txt.split():
                    if token not in self.vocab:
                        self.vocab[token] = len(self.vocab) + 200
                    output.append(self.vocab[token])
                return output
            ids = [101] + encode(review) + [102] + encode(aspect) + [102]
        return {"input_ids": ids, "attention_mask": [1]*len(ids)}


class PayloadContractTests(unittest.TestCase):
    def test_valid(self):
        r, a = adapter.require_payload(payload())
        self.assertEqual(r, REVIEW)
        self.assertEqual(a[0]["text"], "food")

    def test_empty_aspects_valid(self):
        self.assertEqual(adapter.require_payload(payload(aspects=[]))[1], [])

    def test_not_dict(self):
        for value in ([], "", 4, None):
            with self.subTest(v=value), self.assertRaises(ValueError):
                adapter.require_payload(value)

    def test_missing_review(self):
        with self.assertRaisesRegex(ValueError, "review"):
            adapter.require_payload({"aspects": []})

    def test_missing_aspects(self):
        with self.assertRaisesRegex(ValueError, "aspects"):
            adapter.require_payload({"review": REVIEW})

    def test_bad_review(self):
        for value in (None, 12, "", " \n\t"):
            with self.subTest(v=value), self.assertRaises(ValueError):
                adapter.require_payload(payload(review=value))

    def test_bad_aspect_container(self):
        for value in (None, {}, (), "food"):
            with self.subTest(v=value), self.assertRaises(ValueError):
                adapter.require_payload({"review": REVIEW, "aspects": value})

    def test_does_not_strip_review(self):
        review = "  food!  \n"
        self.assertEqual(adapter.require_payload(payload(review=review, aspects=[]))[0], review)

    def test_ignores_unrelated_top_level_keys(self):
        obj = payload(); obj["source"] = "step2.4"
        self.assertEqual(adapter.require_payload(obj)[0], REVIEW)


class RecordTests(unittest.TestCase):
    def check_error(self, record, code):
        with self.assertRaises(adapter.AdapterError) as c:
            adapter.validate_ate_record(REVIEW, record)
        self.assertEqual(c.exception.code, code)

    def test_good_record(self):
        r=adapter.validate_ate_record(REVIEW,sample())
        self.assertEqual(r, {"text":"food","start":4,"end":8,"ate_confidence_uncalibrated":.987})

    def test_missing_field_each(self):
        for key in ("text", "start", "end", "confidence"):
            with self.subTest(key=key):
                item=sample(); del item[key]
                self.check_error(item,"INVALID_ASPECT_RECORD")

    def test_nondict(self):
        for value in (None,[],"food",4):
            with self.subTest(value=value): self.check_error(value,"INVALID_ASPECT_RECORD")

    def test_wrong_text(self):
        self.check_error(sample(text="Food"),"INVALID_SPAN")

    def test_second_occurrence_wrong_offset(self):
        self.check_error(sample("service",4,11),"INVALID_SPAN")

    def test_negative_start(self):
        self.check_error(sample(start=-1),"INVALID_SPAN")

    def test_start_end_equal(self):
        self.check_error(sample(start=4,end=4),"INVALID_SPAN")

    def test_end_beyond_review(self):
        self.check_error(sample(end=999),"INVALID_SPAN")

    def test_float_offset(self):
        self.check_error(sample(start=4.0),"INVALID_SPAN")

    def test_bool_offset(self):
        self.check_error(sample(start=True),"INVALID_SPAN")

    def test_empty_aspect(self):
        self.check_error(sample(text=""),"INVALID_SPAN")

    def test_confidence_invalid(self):
        for val in (-.1, 1.1, True, float("nan"), float("inf"), None, "0.5"):
            with self.subTest(v=str(val)):
                self.check_error(sample(confidence=val),"INVALID_ATE_CONFIDENCE")

    def test_confidence_boundaries(self):
        for val in (0, 1, 0.0, 1.0):
            self.assertEqual(adapter.validate_ate_record(REVIEW,sample(confidence=val))["ate_confidence_uncalibrated"],float(val))


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.tok=FakeTokenizer()

    def run_adapt(self, a=None, review=REVIEW):
        return adapter.adapt_ate_output(payload(review,a),self.tok)

    def test_single_good(self):
        r=self.run_adapt()
        self.assertEqual(r["status"],"READY")
        self.assertEqual(r["summary"],{"input_aspects":1,"ready":1,"errors":0})
        self.assertEqual(r["aspects"][0]["status"],"READY_FOR_ATSC")
        self.assertEqual(r["aspects"][0]["atsc_input"],{"aspect":"food","start":4,"end":8})

    def test_empty(self):
        r=self.run_adapt([])
        self.assertEqual(r["status"],"NO_ASPECTS")
        self.assertEqual(r["aspects"],[])

    def test_two_different(self):
        r=self.run_adapt([sample(),sample("service",32,39)])
        self.assertEqual(r["summary"],{"input_aspects":2,"ready":2,"errors":0})
        self.assertEqual([x["index"] for x in r["aspects"]],[0,1])

    def test_same_word_distinct_occurrences(self):
        review="The service was great, but later the service was terrible."
        first=review.index("service"); second=review.rindex("service")
        r=self.run_adapt([sample("service",first,first+7),sample("service",second,second+7)],review)
        self.assertEqual(r["summary"]["ready"],2)
        self.assertNotEqual(self.tok.calls[0][0],self.tok.calls[2][0])

    def test_duplicate_identical_span_rejected(self):
        r=self.run_adapt([sample(),sample()])
        self.assertEqual(r["status"],"PARTIAL")
        self.assertEqual(r["aspects"][1]["error"]["code"],"DUPLICATE_SPAN")
        self.assertEqual(r["summary"],{"input_aspects":2,"ready":1,"errors":1})

    def test_nonadjacent_duplicate_rejected(self):
        r=self.run_adapt([sample(),sample("service",32,39),sample()])
        self.assertEqual(r["aspects"][2]["error"]["code"],"DUPLICATE_SPAN")

    def test_overlapping_rejected(self):
        r=self.run_adapt([sample(),sample("ood",5,8)])
        self.assertEqual(r["aspects"][1]["error"]["code"],"OVERLAPPING_SPAN")

    def test_backwards_nonoverlapping_input_preserved(self):
        r=self.run_adapt([sample("service",32,39),sample()])
        self.assertEqual(r["summary"]["ready"],2)
        self.assertEqual([x["aspect"]["text"] for x in r["aspects"]],["service","food"])

    def test_incorrect_offsets_one_error_not_all(self):
        r=self.run_adapt([sample(text="Food"),sample("service",32,39)])
        self.assertEqual(r["status"],"PARTIAL")
        self.assertEqual(r["aspects"][0]["error"]["code"],"INVALID_SPAN")
        self.assertEqual(r["aspects"][1]["status"],"READY_FOR_ATSC")

    def test_every_input_has_one_result(self):
        a=[None,sample(),sample(),sample("service",32,39),{"text":"x"}]
        r=self.run_adapt(a)
        self.assertEqual(len(r["aspects"]),len(a))
        self.assertEqual([x["index"] for x in r["aspects"]],list(range(len(a))))

    def test_all_invalid(self):
        r=self.run_adapt([None,sample(text="x")])
        self.assertEqual(r["status"],"REJECTED")
        self.assertEqual(r["summary"]["errors"],2)

    def test_raw_review_identity(self):
        rev="  The  food,\nwas great.  "
        s=rev.index("food")
        r=self.run_adapt([sample("food",s,s+4)],rev)
        self.assertEqual(r["review"],rev)
        self.assertFalse(r["review_modified"])
        self.assertEqual(r["aspects"][0]["aspect"]["start"],s)

    def test_unicode_character_offsets(self):
        rev="Café food is good."
        start=rev.index("food")
        r=self.run_adapt([sample("food",start,start+4)],rev)
        self.assertEqual(r["status"],"READY")
        self.assertEqual(r["aspects"][0]["aspect"]["start"],5)

    def test_multiword_aspect(self):
        rev="The dining room is nice."
        s=rev.index("dining room")
        r=self.run_adapt([sample("dining room",s,s+11)],rev)
        self.assertEqual(r["status"],"READY")

    def test_punctuation(self):
        rev="Great food!"
        s=rev.index("food")
        self.assertEqual(self.run_adapt([sample("food",s,s+4)],rev)["status"],"READY")

    def test_marker_collision_per_aspect_error(self):
        rev="[unused1] food is nice."
        s=rev.index("food")
        r=self.run_adapt([sample("food",s,s+4)],rev)
        self.assertEqual(r["aspects"][0]["error"]["code"],"MARKER_COLLISION_OR_SPAN")

    def test_exact_128_accepted(self):
        rev="food " + "x "*121
        r=self.run_adapt([sample("food",0,4)],rev)
        self.assertEqual(r["aspects"][0]["token_budget"]["used"],128)
        self.assertFalse(r["aspects"][0]["token_budget"]["truncated"])

    def test_129_rejected_and_reported(self):
        rev="food " + "x "*122
        r=self.run_adapt([sample("food",0,4)],rev)
        self.assertEqual(r["status"],"REJECTED")
        err=r["aspects"][0]["error"]
        self.assertEqual(err["code"],"ATSC_TOKEN_OVERFLOW")
        self.assertEqual(err["details"]["required_tokens"],129)
        self.assertEqual(err["details"]["excess_tokens"],1)

    def test_overflow_no_following_aspect_dropped(self):
        rev="food " + "x "*122 + " service"
        s=rev.rindex("service")
        r=self.run_adapt([sample("food",0,4),sample("service",s,s+7)],rev)
        self.assertEqual(len(r["aspects"]),2)
        self.assertEqual(r["summary"]["errors"],2)

    def test_no_silent_truncation(self):
        r=self.run_adapt()
        for _,_,kwargs in self.tok.calls:
            self.assertEqual(kwargs["truncation"],False)
            self.assertEqual(kwargs["padding"],False)
            self.assertEqual(kwargs["add_special_tokens"],True)

    def test_uses_existing_encoder(self):
        with patch.object(adapter.atsc,"encode_pair", wraps=adapter.atsc.encode_pair) as spy:
            self.run_adapt()
            spy.assert_called_once()

    def test_no_model_or_data_calls(self):
        with patch.object(adapter.atsc,"load_classifier",side_effect=AssertionError("model loaded")):
            r=self.run_adapt()
        self.assertFalse(r["atsc_predictions_performed"])
        self.assertFalse(r["pipeline_created"])
        self.assertFalse(r["training_performed"])
        self.assertFalse(r["gold_test_evaluation_performed"])

    def test_wrong_limit_rejected(self):
        for limit in (64,256,True,128.0):
            with self.subTest(limit=limit),self.assertRaises(ValueError):
                adapter.adapt_ate_output(payload(),self.tok,max_length=limit)

    def test_bad_tokenizer_rejected_globally(self):
        self.tok.additional_special_tokens=[]
        with self.assertRaises(ValueError): self.run_adapt()

    def test_tokenizer_error_is_per_aspect(self):
        self.tok.raise_error=True
        r=self.run_adapt()
        self.assertEqual(r["aspects"][0]["error"]["code"],"TOKENIZATION_FAILED")

    def test_bad_token_ids_refused(self):
        self.tok.forced_ids=[101,2,999999,3,102,200,102,200]
        r=self.run_adapt()
        self.assertEqual(r["aspects"][0]["error"]["code"],"ATSC_ENCODING_FAILED")

    def test_duplicate_invalid_first_does_not_block_valid(self):
        r=self.run_adapt([sample("wrong"),sample()])
        self.assertEqual(r["summary"]["ready"],1)

    def test_cannot_write_sentiment(self):
        r=self.run_adapt()
        self.assertNotIn("sentiment",r["aspects"][0])
        self.assertNotIn("probabilities",r["aspects"][0])


class LoaderAndCLITests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.in_file=self.root/"ate.json"
        self.in_file.write_text(json.dumps(payload()),encoding="utf-8")
        self.tok=FakeTokenizer()

    def run_cli(self,*args):
        out=io.StringIO()
        with patch.object(adapter,"load_tokenizer",return_value=(self.tok,128)), contextlib.redirect_stdout(out):
            rc=adapter.main(["--ate-json",str(self.in_file),*args])
        return rc,json.loads(out.getvalue())

    def test_cli_dry_run(self):
        code,report=self.run_cli("--dry-run")
        self.assertEqual(code,0)
        self.assertEqual(report["status"],"READY")

    def test_cli_output_new(self):
        out=self.root/"some"/"adapter.json"
        code,report=self.run_cli("--output",str(out))
        self.assertEqual(code,0)
        self.assertEqual(json.loads(out.read_text(encoding="utf-8")),report)

    def test_cli_no_overwrite(self):
        out=self.root/"exists.json"; out.write_text("original")
        with self.assertRaises(SystemExit) as cm:
            with contextlib.redirect_stderr(io.StringIO()):
                adapter.main(["--ate-json",str(self.in_file),"--output",str(out)])
        self.assertEqual(cm.exception.code,2)
        self.assertEqual(out.read_text(),"original")

    def test_cli_dry_run_with_existing_output_doesnt_write(self):
        out=self.root/"exists.json"; out.write_text("original")
        code,_=self.run_cli("--dry-run","--output",str(out))
        self.assertEqual(code,0)
        self.assertEqual(out.read_text(),"original")

    def test_cli_strict_failed(self):
        self.in_file.write_text(json.dumps(payload(aspects=[sample(text="Food")])),encoding="utf-8")
        code,report=self.run_cli("--strict")
        self.assertEqual(code,2)
        self.assertEqual(report["status"],"REJECTED")

    def test_cli_invalid_payload_before_tokenizer_load(self):
        self.in_file.write_text(json.dumps({"review":REVIEW}),encoding="utf-8")
        with patch.object(adapter,"load_tokenizer",side_effect=AssertionError("should not load")):
            with self.assertRaises(SystemExit) as cm, contextlib.redirect_stderr(io.StringIO()):
                adapter.main(["--ate-json",str(self.in_file)])
        self.assertEqual(cm.exception.code,2)

    def test_cli_unicode_preserved(self):
        r="Café food is pleasant."
        self.in_file.write_text(json.dumps(payload(r,[sample("food",5,9)]),ensure_ascii=False),encoding="utf-8")
        _,report=self.run_cli()
        self.assertEqual(report["review"],r)

    def test_cli_json_read_with_bom(self):
        self.in_file.write_text(json.dumps(payload()),encoding="utf-8-sig")
        _,report=self.run_cli()
        self.assertEqual(report["status"],"READY")

    def test_cli_json_parse_error(self):
        self.in_file.write_text("this isn't JSON")
        with self.assertRaises(SystemExit) as cm, contextlib.redirect_stderr(io.StringIO()):
            adapter.main(["--ate-json",str(self.in_file)])
        self.assertEqual(cm.exception.code,2)

    def test_cli_requires_ate_json(self):
        with self.assertRaises(SystemExit) as cm, contextlib.redirect_stderr(io.StringIO()):
            adapter.build_parser().parse_args([])
        self.assertEqual(cm.exception.code,2)

    def test_loader_missing_manifest(self):
        with self.assertRaises(FileNotFoundError):
            adapter.load_tokenizer(self.root/"missing")

    def test_loader_rejects_other_length(self):
        with patch.object(adapter.atsc,"read_preprocessing_manifest",return_value=256):
            with self.assertRaises(ValueError):
                adapter.load_tokenizer(self.root)

    def test_loader_no_saved_tokenizer_dir(self):
        with patch.object(adapter.atsc,"read_preprocessing_manifest",return_value=128):
            with self.assertRaises(FileNotFoundError):
                adapter.load_tokenizer(self.root)

    
    def test_loader_local_only(self):
        from transformers import AutoTokenizer

        folder = self.root / "tokenizer"
        folder.mkdir()

        def fake_from_pretrained(path, **kwargs):
            self.assertEqual(Path(path), folder)
            self.assertEqual(
                kwargs,
                {
                    "local_files_only": True,
                    "use_fast": True,
                },
            )
            return self.tok

        with patch.object(
            adapter.atsc,
            "read_preprocessing_manifest",
            return_value=128,
        ), patch.object(
            AutoTokenizer,
            "from_pretrained",
            side_effect=fake_from_pretrained,
        ):
            tokenizer, maxlen = adapter.load_tokenizer(self.root)

        self.assertIs(tokenizer, self.tok)
        self.assertEqual(maxlen, 128)


    
    def test_loader_rejects_bad_real_tokenizer_contract(self):
        from transformers import AutoTokenizer

        folder = self.root / "tokenizer"
        folder.mkdir()

        self.tok.is_fast = False

        with patch.object(
            adapter.atsc,
            "read_preprocessing_manifest",
            return_value=128,
        ), patch.object(
            AutoTokenizer,
            "from_pretrained",
            return_value=self.tok,
        ):
            with self.assertRaises(ValueError):
                adapter.load_tokenizer(self.root)


    def test_cli_no_training_model_import(self):
        # This entry point only loads a tokenizer; no classifier/model weights.
        with patch.object(adapter.atsc,"load_classifier",side_effect=AssertionError("no model")), \
             patch.object(adapter.atsc,"predict_sentiment",side_effect=AssertionError("no predictions")):
            code,_=self.run_cli()
        self.assertEqual(code,0)


class RealLocalTokenizerTests(unittest.TestCase):
    """Windows integration tests: actual saved tokenizer, NEVER model weights."""

    def get_tokenizer(self):
        folder=Path("artifacts/atsc_preprocessing/tokenizer")
        if not folder.is_dir():
            self.skipTest("Saved Step 3.2 tokenizer is unavailable in this environment")
        tok,maxlen=adapter.load_tokenizer(folder.parent)
        self.assertEqual(maxlen,128)
        return tok

    def test_saved_tokenizer_two_offsets(self):
        tok=self.get_tokenizer()
        result=adapter.adapt_ate_output(payload(aspects=[sample(),sample("service",32,39)]),tok)
        self.assertEqual(result["status"],"READY")
        self.assertEqual(result["summary"]["ready"],2)

    def test_saved_tokenizer_boundary_overflow(self):
        tok=self.get_tokenizer()
        review="food " + "service "*160
        result=adapter.adapt_ate_output(payload(review,[sample("food",0,4)]),tok)
        self.assertEqual(result["aspects"][0]["error"]["code"],"ATSC_TOKEN_OVERFLOW")
        self.assertGreater(result["aspects"][0]["error"]["details"]["required_tokens"],128)


if __name__=="__main__":
    unittest.main()
