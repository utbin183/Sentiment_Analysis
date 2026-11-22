"""Offline, synthetic tests for Step 3.4. NO real test metrics/model inference."""
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import sys
from types import ModuleType, SimpleNamespace

SRC = Path(__file__).resolve().parents[1] / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
import atsc_evaluate as e


class FakeTokenizer:
    is_fast = True
    pad_token_id = 0
    additional_special_tokens = ['[unused1]', '[unused2]']

    def __len__(self):
        return 30522

    def convert_tokens_to_ids(self, token):
        return {'[unused1]': 2, '[unused2]': 3}.get(token, 100)

    def __call__(self, marked, aspect, **kw):
        # Deterministic synthetic BERT pair preserving target markers.
        assert '[unused1]' in marked and '[unused2]' in marked
        return {"input_ids": [101, 2, 20 if aspect == 'Food' else 21,
                              3, 22, 102, 40, 102],
                "attention_mask": [1] * 8}


def pre_manifest():
    return {
        'status':'PREPROCESSING FINISHED',
        'source_dataset':'thainq107/abte-restaurants',
        'tokenizer_name':e.EXPECTED_MODEL,
        'gold_pairs_only': True, 'ate_predictions_used': False,
        'held_out_test_kept_separate': True,
        'review_overlap_between_splits': 0,
        'truncation':False, 'vocabulary_resized':False,
        'target_markers': e.MARKERS,
        'target_disambiguation':'span_markers_from_gold_start_end_offsets',
        'label_mapping':{'0':'negative','1':'neutral','2':'positive'},
        'training_performed':False, 'model_weights_loaded':False,
        'pair_counts':{'train':5,'validation':3,'test':2},
        'encoded_token_lengths':{'test':{'over_max_length':0}},
        'max_length':128,
        'imbalance':{'class_counts':{'test':{'negative':1,'neutral':0,'positive':1}}},
    }


def summary():
    return {
        'step':'3.3', 'status':'TRAINING FINISHED', 'training_performed':True,
        'test_used':False, 'model':e.EXPECTED_MODEL,
        'selection_metric':'eval_macro_f1',
        'class_mapping':{'0':'negative','1':'neutral','2':'positive'},
        'target_markers':e.MARKERS,
        'best_model_dir':r'artifacts\atsc_training\best_model',
        'best_checkpoint':r'artifacts\atsc_training\checkpoints\checkpoint-12',
        'best_validation_macro_f1':0.72,
        'train_examples':5,'validation_examples':3,
    }


def test_rows():
    return [
        {'input_ids':[101, 2, 20, 3, 22, 102, 40, 102],
         'attention_mask':[1]*8, 'labels':2},
        {'input_ids':[101, 2, 21, 3, 22, 102, 40, 102],
         'attention_mask':[1]*8, 'labels':0},
    ]


def gold_rows():
    return [
        {'review':'Food was good', 'aspect':'Food', 'start':0,'end':4,
         'start_word':0,'end_word':1,'sentiment_id':2,'sentiment':'positive',
         'split':'test','text_origin':'space_joined_source_tokens'},
        {'review':'Service was slow','aspect':'Service','start':0,'end':7,
         'start_word':0,'end_word':1,'sentiment_id':0,'sentiment':'negative',
         'split':'test','text_origin':'space_joined_source_tokens'},
    ]


class TestATSC34(unittest.TestCase):
    def test_manifest_valid(self):
        self.assertEqual(e.validate_preprocessing_manifest(pre_manifest()), 2)

    def test_manifest_requires_done(self):
        p = pre_manifest(); p['status']='FAILED'
        with self.assertRaises(ValueError): e.validate_preprocessing_manifest(p)

    def test_manifest_refuses_ate_predictions(self):
        p=pre_manifest(); p['ate_predictions_used']=True
        with self.assertRaises(ValueError):e.validate_preprocessing_manifest(p)

    def test_manifest_refuses_test_leakage(self):
        p=pre_manifest(); p['review_overlap_between_splits']=1
        with self.assertRaises(ValueError):e.validate_preprocessing_manifest(p)

    def test_manifest_refuses_truncation(self):
        p=pre_manifest(); p['truncation']=True
        with self.assertRaises(ValueError):e.validate_preprocessing_manifest(p)

    def test_manifest_refuses_vocab_resize(self):
        p=pre_manifest(); p['vocabulary_resized']=True
        with self.assertRaises(ValueError):e.validate_preprocessing_manifest(p)

    def test_manifest_refuses_bad_label_names(self):
        p=pre_manifest(); p['label_mapping']['0']='positive'
        with self.assertRaises(ValueError):e.validate_preprocessing_manifest(p)

    def test_manifest_refuses_bad_marker(self):
        p=pre_manifest(); p['target_markers']={'[unused1]':7,'[unused2]':3}
        with self.assertRaises(ValueError):e.validate_preprocessing_manifest(p)

    def test_manifest_refuses_bad_count(self):
        p=pre_manifest(); p['pair_counts']['test']=3
        with self.assertRaises(ValueError):e.validate_preprocessing_manifest(p)

    def test_training_summary_valid(self):
        self.assertEqual(e.validate_training_summary(summary(),pre_manifest()), 'checkpoint-12')

    def test_training_summary_refuses_test_used(self):
        s=summary();s['test_used']=True
        with self.assertRaises(ValueError):e.validate_training_summary(s,pre_manifest())

    def test_training_summary_refuses_wrong_selection(self):
        s=summary();s['selection_metric']='eval_accuracy'
        with self.assertRaises(ValueError):e.validate_training_summary(s,pre_manifest())

    def test_training_summary_refuses_wrong_checkpoint_name(self):
        s=summary();s['best_checkpoint']='artifacts/atsc_training/checkpoints/checkpoint-best'
        with self.assertRaises(ValueError):e.validate_training_summary(s,pre_manifest())

    def test_training_summary_refuses_nonfinite(self):
        s=summary();s['best_validation_macro_f1']=float('nan')
        with self.assertRaises(ValueError):e.validate_training_summary(s,pre_manifest())

    def test_check_tokenizer_valid(self):
        self.assertIsNone(e.validate_tokenizer(FakeTokenizer()))

    def test_check_tokenizer_requires_special_markers(self):
        t=FakeTokenizer();t.additional_special_tokens=[]
        with self.assertRaises(ValueError):e.validate_tokenizer(t)

    def test_test_rows_valid(self):
        self.assertEqual(e.validate_test_rows(test_rows(),pre_manifest(),30522)['pairs'],2)

    def test_test_rows_count_mismatch(self):
        with self.assertRaises(ValueError):e.validate_test_rows(test_rows()[:1],pre_manifest(),30522)

    def test_test_rows_no_malformed_labels(self):
        rows=test_rows();rows[0]['labels']=-1
        with self.assertRaises(ValueError):e.validate_test_rows(rows,pre_manifest(),30522)

    def test_test_rows_no_extra_columns(self):
        rows=test_rows();rows[0]['review']='data'
        with self.assertRaises(ValueError):e.validate_test_rows(rows,pre_manifest(),30522)

    def test_test_rows_no_duplicate_input(self):
        rows=test_rows();rows[1]['input_ids']=rows[0]['input_ids'][:]
        with self.assertRaises(ValueError):e.validate_test_rows(rows,pre_manifest(),30522)

    def test_test_rows_no_missing_marker(self):
        rows=test_rows();rows[1]['input_ids'][1]=4
        with self.assertRaises(ValueError):e.validate_test_rows(rows,pre_manifest(),30522)

    def test_test_rows_no_bad_mask(self):
        rows=test_rows();rows[1]['attention_mask'][0]=0
        with self.assertRaises(ValueError):e.validate_test_rows(rows,pre_manifest(),30522)

    def test_test_rows_no_vocab_overflow(self):
        rows=test_rows();rows[1]['input_ids'][2]=31000
        with self.assertRaises(ValueError):e.validate_test_rows(rows,pre_manifest(),30522)

    def test_test_rows_no_mismatch_class_count(self):
        rows=test_rows();rows[1]['labels']=2
        with self.assertRaises(ValueError):e.validate_test_rows(rows,pre_manifest(),30522)

    def test_gold_alignment_valid(self):
        e.verify_gold_alignment(gold_rows(),test_rows(),FakeTokenizer())

    def test_gold_alignment_rejects_reversed_pairs(self):
        with self.assertRaisesRegex(ValueError,'mismatch|MISALIGNED'):
            e.verify_gold_alignment(list(reversed(gold_rows())),test_rows(),FakeTokenizer())

    def test_gold_alignment_rejects_modified_tokens(self):
        rows=test_rows();rows[1]['input_ids'][2]=1000
        with self.assertRaisesRegex(ValueError,'MISALIGNED'):
            e.verify_gold_alignment(gold_rows(),rows,FakeTokenizer())

    def test_metrics_three_classes(self):
        m=e.confusion_and_metrics([0,1,2,2],[0,2,2,1])
        self.assertEqual(m['confusion_matrix'],[[1,0,0],[0,0,1],[0,1,1]])
        self.assertAlmostEqual(m['accuracy'],0.5)
        self.assertAlmostEqual(m['macro_f1'], (1+0+0.5)/3)
        self.assertEqual(m['per_class']['negative']['support'],1)

    def test_metrics_perfect(self):
        m=e.confusion_and_metrics([0,1,2],[0,1,2])
        self.assertEqual(m['accuracy'],1)
        self.assertEqual(m['macro_f1'],1)

    def test_metrics_all_majority_class(self):
        m=e.confusion_and_metrics([0,1,2,2],[2,2,2,2])
        self.assertAlmostEqual(m['accuracy'],.5)
        self.assertLess(m['macro_f1'],.3)

    def test_metrics_rejects_bool_label(self):
        with self.assertRaises(ValueError):e.confusion_and_metrics([True],[1])

    def test_metrics_rejects_empty(self):
        with self.assertRaises(ValueError):e.confusion_and_metrics([],[])

    def test_metrics_rejects_length_mismatch(self):
        with self.assertRaises(ValueError):e.confusion_and_metrics([0,1],[0])

    def test_probability_argmax(self):
        self.assertEqual(e.validate_probability_rows([[0.1,0.3,0.6]],1),[2])

    def test_probability_rejects_nan(self):
        with self.assertRaises(ValueError):e.validate_probability_rows([[.5,float('nan'),.5]],1)

    def test_probability_rejects_not_sum_one(self):
        with self.assertRaises(ValueError):e.validate_probability_rows([[.5,.5,.5]],1)

    def test_error_analysis(self):
        mistakes,stats=e.create_error_analysis(gold_rows(),[[.1,.1,.8],[.1,.2,.7]])
        self.assertEqual(len(mistakes),1)
        self.assertEqual(mistakes[0]['aspect'],'Service')
        self.assertEqual(mistakes[0]['gold_label'],'negative')
        self.assertEqual(mistakes[0]['predicted_label'],'positive')
        self.assertEqual(stats['by_gold_to_predicted']['negative -> positive'],1)

    def test_error_analysis_none(self):
        mistakes,stats=e.create_error_analysis(gold_rows(),[[.1,.1,.8],[.8,.1,.1]])
        self.assertEqual(mistakes,[])
        self.assertEqual(stats['n_errors'],0)

    def test_read_gold_pairs(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'test.jsonl'
            p.write_text('\n'.join(json.dumps(x) for x in gold_rows()),encoding='utf-8')
            self.assertEqual(len(e.read_gold_test_pairs(p,2)),2)

    def test_read_gold_pairs_rejects_count(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'test.jsonl'
            p.write_text(json.dumps(gold_rows()[0]),encoding='utf-8')
            with self.assertRaises(ValueError):e.read_gold_test_pairs(p,2)

    def test_output_guard(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'result';p.mkdir();(p/'test_metrics.json').write_text('{}')
            with self.assertRaises(ValueError):e.output_guard(p)

    def test_write_reports_synthetic(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'result'
            mat=e.confusion_and_metrics([2,0],[2,2])
            examples,errors=e.create_error_analysis(gold_rows(),[[.1,.1,.8],[.1,.2,.7]])
            e.write_reports(p,mat,examples,errors)
            self.assertTrue((p/'test_metrics.json').is_file())
            self.assertEqual(len((p/'error_examples.jsonl').read_text().splitlines()),1)
            self.assertEqual((p/'confusion_matrix.csv').read_text().splitlines()[0],
                             'gold\\predicted,negative,neutral,positive')
            with self.assertRaises(ValueError):e.write_reports(p,mat,examples,errors)

    def test_read_json(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'a.json';p.write_text('{"x":1}')
            self.assertEqual(e.read_json(p)['x'],1)

    def test_sha256_file(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'a';p.write_bytes(b'hello')
            self.assertEqual(e.sha256_file(p),
                             '2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824')

    def test_weights_good_checkpoint_readonly(self):
        import numpy as np
        from safetensors.numpy import save_file
        with tempfile.TemporaryDirectory() as temp:
            best=Path(temp)/'best';ckpt=Path(temp)/'check';best.mkdir();ckpt.mkdir()
            tensors={'classifier.weight':np.ones((3,4),dtype=np.float32),
                     'classifier.bias':np.zeros((3,),dtype=np.float32)}
            save_file(tensors,str(ckpt/'model.safetensors'))
            shutil.copy2(ckpt/'model.safetensors',best/'model.safetensors')
            result=e.verify_unchanged_best_weights(best,ckpt)
            self.assertTrue(result['checkpoint_identical'])
            self.assertEqual(result['model_bytes'],(best/'model.safetensors').stat().st_size)

    def test_weights_corrupted_rejected(self):
        import numpy as np
        from safetensors.numpy import save_file
        with tempfile.TemporaryDirectory() as temp:
            best=Path(temp)/'best';ckpt=Path(temp)/'check';best.mkdir();ckpt.mkdir()
            save_file({'classifier.weight':np.ones((3,4),dtype=np.float32),
                       'classifier.bias':np.zeros((3,),dtype=np.float32)},str(ckpt/'model.safetensors'))
            (best/'model.safetensors').write_bytes(b'broken')
            with self.assertRaisesRegex(ValueError,'Cannot parse'):
                e.verify_unchanged_best_weights(best,ckpt)

    def test_weights_different_contents_rejected(self):
        import numpy as np
        from safetensors.numpy import save_file
        with tempfile.TemporaryDirectory() as temp:
            best=Path(temp)/'best';ckpt=Path(temp)/'check';best.mkdir();ckpt.mkdir()
            for dest,num in [(best,1),(ckpt,2)]:
                save_file({'classifier.weight':np.ones((3,4),dtype=np.float32)*num,
                           'classifier.bias':np.zeros((3,),dtype=np.float32)},str(dest/'model.safetensors'))
            with self.assertRaisesRegex(ValueError,'SHA-256'):
                e.verify_unchanged_best_weights(best,ckpt)

    def test_synthetic_dry_run_does_not_train_or_write(self):
        from safetensors.numpy import save_file
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=root/'pre';train=root/'training';audit=root/'audit';out=root/'results'
            (data/'tokenized_splits'/'test').mkdir(parents=True)
            (train/'best_model').mkdir(parents=True)
            (train/'checkpoints'/'checkpoint-12').mkdir(parents=True)
            (audit/'pairs').mkdir(parents=True)
            (data/'manifest.json').write_text(json.dumps(pre_manifest()),encoding='utf-8')
            (train/'run_summary.json').write_text(json.dumps(summary()),encoding='utf-8')
            (audit/'audit_report.json').write_text(json.dumps({'status':'AUDIT PASSED','pair_counts':{'test':2}}),encoding='utf-8')
            (audit/'pairs'/'test.jsonl').write_text('\n'.join(json.dumps(x) for x in gold_rows()), encoding='utf-8')
            weights={'classifier.weight':np.ones((3,4),dtype=np.float32),
                     'classifier.bias':np.zeros((3,),dtype=np.float32)}
            save_file(weights,str(train/'best_model'/'model.safetensors'))
            shutil.copy2(train/'best_model'/'model.safetensors',train/'checkpoints'/'checkpoint-12'/'model.safetensors')
            fake_config=SimpleNamespace(model_type='distilbert',num_labels=3,id2label=e.ID2LABEL,vocab_size=30522)
            ds=ModuleType('datasets');ds.load_from_disk=lambda path:test_rows()
            tr=ModuleType('transformers')
            tr.AutoTokenizer=SimpleNamespace(from_pretrained=lambda *args,**kwargs:FakeTokenizer())
            tr.AutoConfig=SimpleNamespace(from_pretrained=lambda *args,**kwargs:fake_config)
            # The synthetic test deliberately doesn't import Torch or model weights.
            with patch.dict(sys.modules,{'datasets':ds,'transformers':tr}):
                self.assertEqual(e.run(['--data',str(data),'--audit',str(audit),'--training',str(train),
                                        '--output',str(out),'--dry-run']),0)
            self.assertFalse(out.exists())

    def test_cli_defaults_no_training_flags(self):
        p=e.build_parser(); args=p.parse_args([])
        self.assertTrue(str(args.training).endswith('atsc_training'))
        self.assertEqual(args.device,'cpu')
        self.assertFalse(hasattr(args,'epochs'))
        self.assertFalse(hasattr(args,'learning_rate'))


if __name__ == '__main__':
    unittest.main()
