"""Offline unit tests for phase-1 audit; no dataset downloads or model training."""
import ast
import unittest
from pathlib import Path

from audit_phase1 import (audit_pdf, audit_source, clean_notebook_shell_lines,
                          run_alignment_unit_test)




ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "data" / "raw"

SOURCE = RAW_DIR / "aspect_based_term_extraction.py"
PDF = RAW_DIR / "aspect-based-sentiment-analysis.pdf"


class Phase1AuditTests(unittest.TestCase):
    def test_pdf_scope(self):
        result = audit_pdf(PDF)
        self.assertEqual(result['page_count'], 10)
        self.assertTrue(result['page10_mentions_ate'])
        self.assertTrue(result['page10_mentions_atsc'])
        self.assertTrue(result['page4_contains_example'])

    def test_original_script_contains_notebook_syntax(self):
        result = audit_source(SOURCE)
        self.assertFalse(result['original_python_syntax']['valid'])
        self.assertEqual(result['notebook_shell_command_lines'], [3, 54])
        self.assertEqual(result['original_python_syntax']['line'], 3)

    def test_python_functions_can_be_parsed_for_audit_without_training(self):
        source = SOURCE.read_text(encoding='utf-8')
        ast.parse(clean_notebook_shell_lines(source))

    def test_reproduce_subword_bio_alignment_bug(self):
        source = SOURCE.read_text(encoding='utf-8')
        tree = ast.parse(clean_notebook_shell_lines(source))
        result = run_alignment_unit_test(tree)
        self.assertTrue(result['ran'])
        self.assertEqual(result['actual_labels'], [1, 1])
        self.assertEqual(result['expected_bio_labels'], [1, 2])
        self.assertFalse(result['bio_alignment_correct'])

    def test_train_uses_test_for_model_selection(self):
        result = audit_source(SOURCE)
        self.assertEqual(result['training_dataset_split'], 'train')
        self.assertEqual(result['evaluation_dataset_split'], 'test')
        self.assertTrue(result['test_data_used_for_checkpoint_selection'])
        self.assertTrue(result['training_runs_at_module_top_level'])

    def test_source_does_not_consume_polarities(self):
        result = audit_source(SOURCE)
        self.assertFalse(result['source_uses_polarities_field'])
        self.assertEqual(result['source_dataset_identifiers'], ['thainq107/abte-restaurants'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
