"""Step 4.3 tests. Synthetic review/model/tokenizer; no gold/test data or training.

Optional real-local smoke is activated ONLY with ABSA_REAL_SMOKE=1 and uses a
fixed example review, NOT any held-out test dataset.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import absa_pipeline as pipe

REVIEW = "The food was delicious, but the service was terrible."


def item(text, start, end, confidence=.9):
    return {"text":text,"start":start,"end":end,"confidence":confidence}


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

    def __len__(self):
        return 30522

    def num_special_tokens_to_add(self, pair=False):
        return 3 if pair else 2

    def convert_tokens_to_ids(self, token):
        return self.vocab.get(token, 100)

    def __call__(self, a, b, **kwargs):
        self.calls.append((a, b, kwargs))
        if self.forced_ids is not None:
            ids = list(self.forced_ids)
        else:
            def encode(s):
                vals = []
                for tok in s.split():
                    if tok not in self.vocab:
                        self.vocab[tok] = len(self.vocab) + 200
                    vals.append(self.vocab[tok])
                return vals
            ids = [101] + encode(a) + [102] + encode(b) + [102]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


class FakeExtractor:
    def __init__(self, aspects=None, failure=None):
        self.aspects = aspects if aspects is not None else [item("food",4,8), item("service",32,39)]
        self.failure = failure
        self.calls = []

    def extract(self, review):
        self.calls.append(review)
        if self.failure:
            raise self.failure
        return self.aspects


class FakeClassifier:
    def __init__(self, tokenizer=None, max_length=128, output_overrides=None, failures=None):
        self.tokenizer = tokenizer if tokenizer is not None else FakeTokenizer()
        self.max_length = max_length
        self.output_overrides = output_overrides or {}
        self.failures = failures or {}
        self.calls = []

    def predict(self, review, aspect, start, end):
        self.calls.append((review, aspect, start, end))
        if start in self.failures:
            raise self.failures[start]
        label = 2 if aspect == "food" else 0
        probs = {"negative": .01, "neutral": .02, "positive": .97} if label == 2 else \
                {"negative": .97, "neutral": .02, "positive": .01}
        result = {"review":review, "aspect":{"text":aspect,"start":start,"end":end},
                  "sentiment":pipe.atsc.ID2LABEL[label], "label_id": label,
                  "probabilities_softmax_uncalibrated":probs,
                  "predicted_probability_uncalibrated":probs[pipe.atsc.ID2LABEL[label]]}
        result.update(self.output_overrides.get(start, {}))
        return result


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.ate = FakeExtractor()
        self.atsc = FakeClassifier()
        self.pipeline = pipe.ABSAInferencePipeline(self.ate, self.atsc)

    def test_two_aspects_complete(self):
        r=self.pipeline.predict(REVIEW)
        self.assertEqual(r["status"],"COMPLETE")
        self.assertEqual(r["summary"],{"extracted":2,"classified":2,"errors":0,"global_errors":0})
        self.assertEqual([x["sentiment"] for x in r["aspects"]],["positive","negative"])
        self.assertEqual([x["status"] for x in r["aspects"]],["CLASSIFIED","CLASSIFIED"])

    def test_preserve_exact_review(self):
        review="  The food was great.\n"
        self.ate.aspects=[item("food",review.index("food"),review.index("food")+4)]
        out=self.pipeline.predict(review)
        self.assertEqual(out["review"],review)
        self.assertEqual(self.atsc.calls[0][0],review)
        self.assertFalse(out["review_modified"])

    def test_character_offsets_passed_unchanged(self):
        self.pipeline.predict(REVIEW)
        self.assertEqual(self.atsc.calls,[(REVIEW,"food",4,8),(REVIEW,"service",32,39)])

    def test_no_pred_on_empty(self):
        self.ate.aspects=[]
        out=self.pipeline.predict(REVIEW)
        self.assertEqual(out["status"],"NO_ASPECTS")
        self.assertEqual(out["summary"]["classified"],0)
        self.assertEqual(self.atsc.calls,[])

    def test_bad_span_only_one_error(self):
        self.ate.aspects=[item("food",4,8),item("service",4,11)]
        out=self.pipeline.predict(REVIEW)
        self.assertEqual(out["status"],"PARTIAL")
        self.assertEqual(out["summary"]["classified"],1)
        self.assertEqual(out["aspects"][1]["error"]["code"],"INVALID_SPAN")
        self.assertIsNone(out["aspects"][1]["sentiment"])
        self.assertEqual(len(self.atsc.calls),1)

    def test_duplicate_span_only_one_forward(self):
        self.ate.aspects=[item("food",4,8),item("food",4,8)]
        out=self.pipeline.predict(REVIEW)
        self.assertEqual(out["summary"]["errors"],1)
        self.assertEqual(out["aspects"][1]["error"]["code"],"DUPLICATE_SPAN")
        self.assertEqual(len(self.atsc.calls),1)

    def test_overlap_rejected(self):
        self.ate.aspects=[item("food",4,8),item("ood",5,8)]
        out=self.pipeline.predict(REVIEW)
        self.assertEqual(out["aspects"][1]["error"]["code"],"OVERLAPPING_SPAN")

    def test_repeated_text_two_occurrences(self):
        review="The service was great, but later the service was terrible."
        positions=[review.index("service"),review.rindex("service")]
        self.ate.aspects=[item("service",p,p+7) for p in positions]
        out=self.pipeline.predict(review)
        self.assertEqual(out["status"],"COMPLETE")
        self.assertEqual([c[2] for c in self.atsc.calls],positions)
        self.assertEqual(out["summary"]["classified"],2)
        self.assertNotEqual(self.atsc.tokenizer.calls[0][0],self.atsc.tokenizer.calls[2][0])

    def test_unicode_code_point_offsets(self):
        review="Café food was excellent."
        p=review.index("food");self.ate.aspects=[item("food",p,p+4)]
        out=self.pipeline.predict(review)
        self.assertEqual(out["aspects"][0]["aspect"]["start"],p)

    def test_overflow_per_aspect(self):
        self.atsc.tokenizer.forced_ids=[101,2]+list(range(200,324))+[3,102,102]
        self.ate.aspects=[item("food",4,8)]
        out=self.pipeline.predict(REVIEW)
        self.assertEqual(out["status"],"FAILED")
        self.assertEqual(out["aspects"][0]["error"]["code"],"ATSC_TOKEN_OVERFLOW")
        self.assertEqual(len(self.atsc.calls),0)
        self.assertGreater(out["aspects"][0]["error"]["details"]["required_tokens"],128)

    def test_ate_error_global(self):
        self.ate.failure=RuntimeError("fake model down")
        out=self.pipeline.predict(REVIEW)
        self.assertEqual(out["status"],"ATE_ERROR")
        self.assertEqual(out["error"]["code"],"ATE_INFERENCE_FAILED")
        self.assertEqual(out["summary"]["global_errors"],1)
        self.assertFalse(out["ate_extraction_performed"])

    def test_ate_overflow_global(self):
        self.ate.failure=ValueError("Review requires 130 tokens; No input was truncated")
        out=self.pipeline.predict(REVIEW)
        self.assertEqual(out["error"]["code"],"ATE_TOKEN_OVERFLOW")
        self.assertEqual(len(self.atsc.calls),0)

    def test_ate_wrong_return(self):
        self.ate.aspects={"food":"wrong"}
        out=self.pipeline.predict(REVIEW)
        self.assertEqual(out["error"]["code"],"ATE_OUTPUT_INVALID")

    def test_atsc_error_per_aspect_then_continue(self):
        self.atsc.failures={4:RuntimeError("forward error")}
        out=self.pipeline.predict(REVIEW)
        self.assertEqual(out["status"],"PARTIAL")
        self.assertEqual(out["aspects"][0]["error"]["code"],"ATSC_INFERENCE_FAILED")
        self.assertEqual(out["aspects"][1]["sentiment"],"negative")
        self.assertEqual(len(self.atsc.calls),2)

    def test_all_atsc_fail(self):
        self.atsc.failures={4:RuntimeError("bad"),32:RuntimeError("bad")}
        out=self.pipeline.predict(REVIEW)
        self.assertEqual(out["status"],"FAILED")
        self.assertEqual(out["summary"]["errors"],2)

    def test_atcs_malformed_result_per_aspect(self):
        self.atsc.output_overrides={4:{"review":"changed"}}
        out=self.pipeline.predict(REVIEW)
        self.assertEqual(out["aspects"][0]["error"]["code"],"ATSC_OUTPUT_INVALID")
        self.assertEqual(out["summary"]["classified"],1)

    def test_ate_and_atsc_load_only_once_across_reviews(self):
        self.pipeline.predict(REVIEW)
        self.pipeline.predict(REVIEW)
        self.assertEqual(len(self.ate.calls),2)
        self.assertEqual(len(self.atsc.calls),4)
        self.assertIs(self.pipeline.extractor,self.ate)
        self.assertIs(self.pipeline.classifier,self.atsc)

    def test_probabilities_are_uncalibrated(self):
        out=self.pipeline.predict(REVIEW)
        self.assertEqual(out["aspects"][0]["probabilities_softmax_uncalibrated"]["positive"],.97)
        self.assertNotIn("confidence_calibrated",str(out))

    def test_no_gold_or_training_flags(self):
        out=self.pipeline.predict(REVIEW)
        for key in ("training_performed","gold_test_evaluation_performed","weights_modified"):
            self.assertFalse(out[key])

    def test_empty_review_rejected_before_extract(self):
        for bad in (None,"","  \n  ",4):
            with self.subTest(v=bad), self.assertRaises(ValueError):
                self.pipeline.predict(bad)
        self.assertEqual(self.ate.calls,[])

    def test_output_json_serializable(self):
        out=self.pipeline.predict(REVIEW)
        self.assertEqual(json.loads(json.dumps(out,allow_nan=False))["status"],"COMPLETE")

    def test_indexes_and_order_preserved(self):
        self.ate.aspects=[item("service",32,39),item("food",4,8)]
        out=self.pipeline.predict(REVIEW)
        self.assertEqual([r["index"] for r in out["aspects"]],[0,1])
        self.assertEqual([r["aspect"]["text"] for r in out["aspects"]],["service","food"])

    def test_token_budget(self):
        out=self.pipeline.predict(REVIEW)
        for x in out["aspects"]:
            self.assertLessEqual(x["token_budget"]["used"],128)
            self.assertFalse(x["token_budget"]["truncated"])

    def test_source_confidence_preserved(self):
        self.ate.aspects=[item("food",4,8,.45678)]
        out=self.pipeline.predict(REVIEW)
        self.assertEqual(out["aspects"][0]["aspect"]["ate_confidence_uncalibrated"],.45678)

    def test_will_not_write_any_file(self):
        with patch.object(Path,"write_text",side_effect=AssertionError("No write allowed")):
            self.pipeline.predict(REVIEW)

    def test_adapter_called_with_atsc_tokenizer(self):
        with patch.object(pipe.adapter,"adapt_ate_output",wraps=pipe.adapter.adapt_ate_output) as adapted:
            self.pipeline.predict(REVIEW)
        self.assertIs(adapted.call_args.args[1],self.atsc.tokenizer)

    def test_atsc_error_does_not_pretend_prediction(self):
        self.atsc.failures={4:ValueError("invalid logits")}
        out=self.pipeline.predict(REVIEW)
        x=out["aspects"][0]
        self.assertEqual(x["status"],"ERROR")
        self.assertIsNone(x["probabilities_softmax_uncalibrated"])
        self.assertIsNone(x["sentiment"])

    def test_adapter_global_failure(self):
        with patch.object(pipe.adapter,"adapt_ate_output",side_effect=ValueError("bad adapter")):
            out=self.pipeline.predict(REVIEW)
        self.assertEqual(out["status"],"PIPELINE_ERROR")
        self.assertEqual(out["summary"]["extracted"],2)
        self.assertFalse(out["atsc_predictions_performed"])


class SentimentValidationTests(unittest.TestCase):
    def setUp(self):
        self.expect={"text":"food","start":4,"end":8}
        self.raw=FakeClassifier().predict(REVIEW,"food",4,8)

    def validate(self):return pipe.validate_sentiment_output(self.raw,REVIEW,self.expect)

    def test_valid(self):
        self.assertEqual(self.validate()["sentiment"],"positive")

    def test_wrong_review(self):
        self.raw["review"]="different"
        with self.assertRaisesRegex(ValueError,"review"):self.validate()

    def test_wrong_aspect(self):
        self.raw["aspect"]["text"]="service"
        with self.assertRaises(ValueError):self.validate()

    def test_wrong_start(self):
        self.raw["aspect"]["start"]=5
        with self.assertRaises(ValueError):self.validate()

    def test_wrong_end(self):
        self.raw["aspect"]["end"]=9
        with self.assertRaises(ValueError):self.validate()

    def test_wrong_sentiment(self):
        self.raw["sentiment"]="negative"
        with self.assertRaises(ValueError):self.validate()

    def test_invalid_label(self):
        for label in (-1,3,True,None,2.0):
            with self.subTest(v=label):
                self.raw["label_id"]=label
                with self.assertRaises(ValueError):self.validate()

    def test_bad_probabilities(self):
        for probs in ({}, {"positive":1}, None):
            with self.subTest(v=probs):
                self.raw["probabilities_softmax_uncalibrated"]=probs
                with self.assertRaises(ValueError):self.validate()

    def test_probability_nan(self):
        self.raw["probabilities_softmax_uncalibrated"]["positive"]=float("nan")
        with self.assertRaises(ValueError):self.validate()

    def test_probability_inf(self):
        self.raw["probabilities_softmax_uncalibrated"]["positive"]=float("inf")
        with self.assertRaises(ValueError):self.validate()

    def test_probability_bool(self):
        self.raw["probabilities_softmax_uncalibrated"]["positive"]=True
        with self.assertRaises(ValueError):self.validate()

    def test_probability_over_one(self):
        self.raw["probabilities_softmax_uncalibrated"]["positive"]=1.2
        with self.assertRaises(ValueError):self.validate()

    def test_probability_sum(self):
        self.raw["probabilities_softmax_uncalibrated"]["positive"]=.7
        with self.assertRaises(ValueError):self.validate()

    def test_probability_argmax(self):
        self.raw["probabilities_softmax_uncalibrated"]={"negative":.6,"neutral":.2,"positive":.2}
        self.raw["predicted_probability_uncalibrated"]=.6
        with self.assertRaises(ValueError):self.validate()

    def test_probability_winner_mismatch(self):
        self.raw["predicted_probability_uncalibrated"]=.3
        with self.assertRaises(ValueError):self.validate()

    def test_missing_probabilities(self):
        del self.raw["probabilities_softmax_uncalibrated"]
        with self.assertRaises(ValueError):self.validate()


class LoaderAndCLITests(unittest.TestCase):
    def setUp(self):
        self.ate=FakeExtractor()
        self.atsc=FakeClassifier()

    def test_load_exact_once(self):
        with patch.object(pipe.ate,"load_extractor",return_value=self.ate) as a, \
             patch.object(pipe.atsc,"load_classifier",return_value=self.atsc) as b:
            p=pipe.load_pipeline(device="cpu")
        self.assertIsInstance(p,pipe.ABSAInferencePipeline)
        a.assert_called_once()
        b.assert_called_once()
        self.assertEqual(a.call_args.kwargs["max_length"],128)

    def test_load_paths(self):
        with patch.object(pipe.ate,"load_extractor",return_value=self.ate) as a, \
             patch.object(pipe.atsc,"load_classifier",return_value=self.atsc) as b:
            pipe.load_pipeline(ate_model_dir="ate",atsc_model_dir="atsc",preprocessing_dir="data",device="cpu")
        self.assertEqual(a.call_args.kwargs["model_dir"],"ate")
        self.assertEqual(b.call_args.kwargs["model_dir"],"atsc")
        self.assertEqual(b.call_args.kwargs["preprocessing_dir"],"data")

    def test_bad_device(self):
        with self.assertRaises(ValueError):pipe.load_pipeline(device="xx")

    def test_constructor_rejects_wrong_length(self):
        with self.assertRaises(ValueError):pipe.ABSAInferencePipeline(self.ate,FakeClassifier(max_length=256))

    def test_constructor_rejects_no_extract(self):
        with self.assertRaises(TypeError):pipe.ABSAInferencePipeline(object(),self.atsc)

    def test_constructor_rejects_no_predict(self):
        with self.assertRaises(TypeError):pipe.ABSAInferencePipeline(self.ate,object())

    def test_cli_requires_review(self):
        with self.assertRaises(SystemExit):pipe.build_parser().parse_args([])

    def test_cli_mutually_exclusive(self):
        with self.assertRaises(SystemExit):pipe.build_parser().parse_args(["--review","x","--input-file","x"])

    def test_cli_smoke_synthetic_no_file(self):
        p=pipe.ABSAInferencePipeline(self.ate,self.atsc)
        with patch.object(pipe,"load_pipeline",return_value=p),contextlib.redirect_stdout(io.StringIO()) as o:
            code=pipe.main(["--review",REVIEW,"--device","cpu"])
        self.assertEqual(code,0)
        self.assertEqual(json.loads(o.getvalue())["status"],"COMPLETE")

    def test_cli_preserves_utf8_file(self):
        r="Café food was excellent.\n"
        p=r.index("food");self.ate.aspects=[item("food",p,p+4)]
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"review.txt";path.write_text(r,encoding="utf-8")
            model=pipe.ABSAInferencePipeline(self.ate,self.atsc)
            with patch.object(pipe,"load_pipeline",return_value=model),contextlib.redirect_stdout(io.StringIO()) as o:
                status=pipe.main(["--input-file",str(path)])
        self.assertEqual(status,0)
        self.assertEqual(json.loads(o.getvalue())["review"],r)

    def test_cli_no_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"report.json";path.write_text("old",encoding="utf-8")
            with patch.object(pipe,"load_pipeline",side_effect=AssertionError("must not load")):
                with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                    pipe.main(["--review",REVIEW,"--output",str(path)])
            self.assertEqual(path.read_text(encoding="utf-8"),"old")

    def test_cli_new_file(self):
        model=pipe.ABSAInferencePipeline(self.ate,self.atsc)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"out.json"
            with patch.object(pipe,"load_pipeline",return_value=model),contextlib.redirect_stdout(io.StringIO()):
                code=pipe.main(["--review",REVIEW,"--output",str(path)])
            self.assertEqual(code,0)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["status"],"COMPLETE")

    def test_cli_strict_partial(self):
        self.ate.aspects=[item("food",4,8),item("service",4,11)]
        model=pipe.ABSAInferencePipeline(self.ate,self.atsc)
        with patch.object(pipe,"load_pipeline",return_value=model),contextlib.redirect_stdout(io.StringIO()) as o:
            code=pipe.main(["--review",REVIEW,"--strict"])
        self.assertEqual(code,2)
        self.assertEqual(json.loads(o.getvalue())["status"],"PARTIAL")

    def test_cli_invalid_before_model_load(self):
        with patch.object(pipe,"load_pipeline",side_effect=AssertionError("should not load")):
            with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                pipe.main(["--review","  "])

    def test_no_training_dataset_imports(self):
        src=Path(pipe.__file__).read_text(encoding="utf-8")
        for forbidden in ("Trainer(","train_dataset=","load_from_disk(","optimizer.step(","streamlit", "evaluate_test"):
            self.assertNotIn(forbidden,src)


@unittest.skipUnless(os.environ.get("ABSA_REAL_SMOKE") == "1", "Real checkpoint smoke opt-in: set ABSA_REAL_SMOKE=1")
class RealLocalModelsSmoke(unittest.TestCase):
    def test_models_on_real_review_only(self):
        # Opt-in test: requires user's actual saved checkpoints; no gold/test split is opened.
        self.assertTrue(Path(pipe.DEFAULT_ATE_MODEL).is_dir(),"Missing local ATE checkpoint")
        self.assertTrue(Path(pipe.DEFAULT_ATSC_MODEL).is_dir(),"Missing local ATSC checkpoint")
        pipeline=pipe.load_pipeline(device="cpu")
        out=pipeline.predict(REVIEW)
        self.assertEqual(out["status"],"COMPLETE")
        self.assertEqual(out["summary"]["extracted"],2)
        self.assertEqual(out["summary"]["classified"],2)
        self.assertEqual([(a["aspect"]["text"],a["sentiment"]) for a in out["aspects"]],
                         [("food","positive"),("service","negative")])
        self.assertEqual(out["review"],REVIEW)


if __name__=="__main__":
    unittest.main()
