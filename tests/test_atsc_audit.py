"""Step 3.1 regression tests: all labels and predictions here are *synthetic gold*, NOT model metrics."""
from __future__ import annotations

import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout, redirect_stderr

# In user's layout run with PYTHONPATH="$PWD/src;$PWD" (PowerShell).
from atsc_audit import (
    DATASET_NAME,
    POLARITY_ID_TO_NAME,
    build_gold_pairs,
    integer_labels,
    main,
    make_text_and_offsets,
    parse_atsc_row,
    write_jsonl,
)


def row(tokens, tags, polarities):
    return {"Tokens": tokens, "Tags": tags, "Polarities": polarities}


def fixture():
    # Synthetic text/labels used only to test the data pipeline.
    a = ["The", "food", "was", "great", ",", "service", "slow", "."]
    tags_a = [0,1,0,0,0,1,0,0]
    b = ["Price", "was", "ok", "."]
    c = ["The", "pasta", "was", "hot", "."]
    d = ["The", "staff", "was", "rude", "."]
    src_train = [
        row(a,tags_a,[-1,2,-1,-1,-1,-1,-1,-1]),
        row(a,tags_a,[-1,-1,-1,-1,-1,0,-1,-1]),
        row(b,[1,0,0,0],[1,-1,-1,-1]),
        row(c,[0,1,0,0,0],[-1,2,-1,-1,-1]),
        # One source train/test duplicate must be removed from ATSC train/val.
        row(d,[0,1,0,0,0],[-1,0,-1,-1,-1]),
    ]
    src_test = [row(d,[0,1,0,0,0],[-1,0,-1,-1,-1])]
    ate = {
        "train":[{"Tokens": a, "Tags":tags_a}],
        "validation":[{"Tokens": b, "Tags":[1,0,0,0]}, {"Tokens": c, "Tags":[0,1,0,0,0]}],
        "test":[{"Tokens":d, "Tags":[0,1,0,0,0]}],
    }
    return src_train,src_test,ate


class TestATSCAudit(unittest.TestCase):
    def test_mapping_is_explicit_not_inferred(self):
        self.assertEqual(POLARITY_ID_TO_NAME,{0:"negative",1:"neutral",2:"positive"})

    def test_int_and_string_labels(self):
        self.assertEqual(integer_labels(["-1",0,"1",2],source="fixture"),[-1,0,1,2])

    def test_invalid_new_polarity_cannot_be_mapped(self):
        with self.assertRaisesRegex(ValueError,"unknown"):
            integer_labels([3],source="fixture")

    def test_bool_rejected(self):
        with self.assertRaisesRegex(ValueError,"non-integer"):
            integer_labels([True],source="fixture")

    def test_float_rejected(self):
        with self.assertRaisesRegex(ValueError,"non-integer"):
            integer_labels([0.0],source="fixture")

    def test_missing_polarities_rejected(self):
        with self.assertRaisesRegex(ValueError,"missing Polarities"):
            parse_atsc_row({"Tokens":["food"],"Tags":[1]},source="fixture",index=0)

    def test_polarity_length_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError,"length mismatch"):
            parse_atsc_row(row(["good","food"],[0,1],[-1]),source="fixture",index=0)

    def test_polarity_outside_bio_rejected(self):
        with self.assertRaisesRegex(ValueError,"outside BIO"):
            parse_atsc_row(row(["good","food"],[0,1],[2,-1]),source="fixture",index=0)

    def test_all_minus_one_aspect_is_not_negative(self):
        _,pairs,missing=parse_atsc_row(row(["nice","food"],[0,1],[-1,-1]),source="fixture",index=0)
        self.assertEqual(pairs,[])
        self.assertEqual(missing,1)

    def test_multiple_gold_polarities_one_row(self):
        tokens = ["food","but","service"]
        _,pairs,missing = parse_atsc_row(row(tokens,[1,0,1],[2,-1,0]),source="fixture",index=0)
        self.assertEqual(pairs,[(0,1,2),(2,3,0)])
        self.assertEqual(missing,0)

    def test_multiword_aspect_one_label(self):
        _,pairs,missing=parse_atsc_row(row(["great","customer","service"],[0,1,2],[-1,2,2]),source="fixture",index=0)
        self.assertEqual(pairs,[(1,3,2)])
        self.assertEqual(missing,0)

    def test_multiword_conflicting_polarities_rejected(self):
        with self.assertRaisesRegex(ValueError,"conflicting token"):
            parse_atsc_row(row(["customer","service"],[1,2],[2,0]),source="fixture",index=0)

    def test_partially_labeled_multiword_rejected(self):
        with self.assertRaisesRegex(ValueError,"partially annotated"):
            parse_atsc_row(row(["customer","service"],[1,2],[2,-1]),source="fixture",index=0)

    def test_invalid_bio_i_rejected(self):
        with self.assertRaisesRegex(ValueError,"I-Term"):
            parse_atsc_row(row(["service"],[2],[2]),source="fixture",index=0)

    def test_char_offsets_from_reconstructed_tokens(self):
        text,aspect,start,end=make_text_and_offsets(["The", "price", "was", "high", "."],1,2)
        self.assertEqual((text,aspect,start,end),("The price was high .","price",4,9))
        self.assertEqual(text[start:end],aspect)

    def test_offsets_for_starting_multiword(self):
        text,aspect,start,end=make_text_and_offsets(["customer","service","bad"],0,2)
        self.assertEqual((aspect,start,end),("customer service",0,16))
        self.assertEqual(text[start:end],aspect)

    def test_exact_saved_ate_membership_and_class_dist(self):
        tr,te,ate=fixture()
        pairs,stats=build_gold_pairs(tr,te,ate)
        self.assertEqual(stats["ate_review_splits"],{"train":1,"validation":2,"test":1})
        self.assertEqual(stats["pair_counts"],{"train":2,"validation":2,"test":1})
        self.assertEqual(stats["class_distribution"]["train"],{"negative":1,"neutral":0,"positive":1})
        self.assertEqual(stats["class_distribution"]["validation"],{"negative":0,"neutral":1,"positive":1})
        self.assertEqual(stats["class_distribution"]["test"],{"negative":1,"neutral":0,"positive":0})
        self.assertFalse(stats["model_trained"])

    def test_source_cross_split_review_dropped_from_train(self):
        tr,te,ate=fixture()
        pairs,stats=build_gold_pairs(tr,te,ate)
        self.assertEqual(stats["source_cross_split_duplicate_rows_omitted"],1)
        self.assertEqual([r["aspect"] for r in pairs["test"]],["staff"])
        self.assertFalse(any(r["aspect"]=="staff" for r in pairs["train"]+pairs["validation"]))

    def test_same_review_two_target_rows_become_two_pairs(self):
        tr,te,ate=fixture()
        pairs,_=build_gold_pairs(tr,te,ate)
        self.assertEqual([(r["aspect"],r["sentiment"]) for r in pairs["train"]],[("food","positive"),("service","negative")])

    def test_exact_duplicate_aspect_annotation_deduplicated(self):
        tr,te,ate=fixture()
        tr.insert(0,tr[0])
        pairs,stats=build_gold_pairs(tr,te,ate)
        self.assertEqual(stats["pair_counts"]["train"],2)
        self.assertEqual(stats["duplicate_polarity_occurrences_deduplicated"]["train"],1)

    def test_conflicting_repeated_gold_label_rejected(self):
        tr,te,ate=fixture()
        conflict=dict(tr[0]); conflict["Polarities"]=[-1,0,-1,-1,-1,-1,-1,-1]
        tr.append(conflict)
        with self.assertRaisesRegex(ValueError,"Conflicting polarity labels"):
            build_gold_pairs(tr,te,ate)

    def test_mismatch_saved_ate_aspects_rejected(self):
        tr,te,ate=fixture()
        ate["train"][0]["Tags"]=[0,1,0,0,0,0,0,0]
        with self.assertRaisesRegex(ValueError,"ATE gold spans differ"):
            build_gold_pairs(tr,te,ate)

    def test_missing_saved_review_rejected(self):
        tr,te,ate=fixture()
        ate["validation"].pop()
        with self.assertRaisesRegex(ValueError,"ATE saved splits do not correspond"):
            build_gold_pairs(tr,te,ate)

    def test_review_leakage_in_saved_ate_splits_rejected(self):
        tr,te,ate=fixture()
        ate["test"].append(dict(ate["train"][0]))
        with self.assertRaisesRegex(ValueError,"Duplicate review in ATE"):
            build_gold_pairs(tr,te,ate)

    def test_serialized_pairs_have_valid_offsets(self):
        tr,te,ate=fixture()
        pairs,_=build_gold_pairs(tr,te,ate)
        for rows in pairs.values():
            for p in rows:
                self.assertEqual(p["review"][p["start"]:p["end"]],p["aspect"])
                self.assertIn(p["sentiment"],POLARITY_ID_TO_NAME.values())
                self.assertEqual(p["text_origin"],"space_joined_source_tokens")

    def test_write_jsonl_unicode_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"data.jsonl"
            write_jsonl(path,[{"aspect":"café","sentiment":"positive"}])
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")),{"aspect":"café","sentiment":"positive"})

    def test_no_training_or_model_imports_in_module(self):
        import atsc_audit
        source=Path(atsc_audit.__file__).read_text(encoding="utf-8")
        self.assertNotIn("trainer.train(",source.lower())
        self.assertNotIn("AutoModelFor",source)
        self.assertNotIn("torch.load(",source)

    def test_cli_rejects_missing_ate_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit) as cm:
                main(["--ate-data",str(Path(tmp)/"missing"),"--output",str(Path(tmp)/"out")])
            self.assertEqual(cm.exception.code,2)

    def test_cli_synthetic_end_to_end_without_training(self):
        tr,te,ate=fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            atedir=root/"ate"
            (atedir/"raw_splits").mkdir(parents=True)
            (atedir/"manifest.json").write_text(json.dumps({
                "dataset":DATASET_NAME,"review_overlap_between_final_splits":0,
                "split_counts":{k:len(v) for k,v in ate.items()}
            }),encoding="utf-8")
            # Inject local in-memory gold objects; test never contacts HF Hub.
            module=ModuleType("datasets")
            module.load_dataset=lambda name:{"train":tr,"test":te}
            module.load_from_disk=lambda path:ate
            stdout=io.StringIO()
            with patch.dict(sys.modules,{"datasets":module}),redirect_stdout(stdout):
                rc=main(["--ate-data",str(atedir),"--output",str(root/"output")])
            self.assertEqual(rc,0)
            report=json.loads((root/"output"/"audit_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["pair_counts"],{"train":2,"validation":2,"test":1})
            self.assertIsNone(report["model_metrics"])
            with (root / "output" / "pairs" / "train.jsonl").open(encoding="utf-8") as f:
                self.assertEqual(sum(1 for _ in f), 2)
            self.assertIn("ATSC AUDIT FINISHED",stdout.getvalue())

    def test_cli_refuses_to_overwrite_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            out=root/"out"
            out.mkdir()
            (out/"audit_report.json").write_text("do not overwrite",encoding="utf-8")
            with self.assertRaises(SystemExit) as cm:
                main(["--ate-data",str(root/"doesnotmatter"),"--output",str(out)])
            self.assertEqual(cm.exception.code,2)
            self.assertEqual((out/"audit_report.json").read_text(encoding="utf-8"),"do not overwrite")


if __name__=="__main__":
    unittest.main()
