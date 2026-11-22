"""Step 5.1: no-training tests for Streamlit ABSA presentation and pipeline wiring.

Most tests work without Streamlit/Transformers via a fake UI and fake pipeline.
An opt-in real local checkpoint smoke test is included for the Windows project.
"""
from __future__ import annotations

import contextlib
import copy
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import app

REVIEW = "The food was delicious, but the service was terrible."


def ready_report(review: str = REVIEW) -> dict:
    return {
        "phase": "4.3", "review": review, "review_modified": False,
        "offset_contract": "python_unicode_codepoints_half_open", "max_atsc_tokens": 128,
        "overflow_policy": "per_aspect_structured_error_no_truncation",
        "status": "COMPLETE", "summary": {"extracted": 2, "classified": 2, "errors": 0, "global_errors": 0},
        "aspects": [
            {"index": 0, "status": "CLASSIFIED", "aspect": {"text": "food", "start": 4, "end": 8,
             "ate_confidence_uncalibrated": .998326},
             "token_budget": {"used": 17, "limit": 128, "truncated": False},
             "sentiment": "positive", "label_id": 2,
             "probabilities_softmax_uncalibrated": {"negative": .001, "neutral": .002, "positive": .997},
             "predicted_probability_uncalibrated": .997, "error": None},
            {"index": 1, "status": "CLASSIFIED", "aspect": {"text": "service", "start": 32, "end": 39,
             "ate_confidence_uncalibrated": .998984},
             "token_budget": {"used": 17, "limit": 128, "truncated": False},
             "sentiment": "negative", "label_id": 0,
             "probabilities_softmax_uncalibrated": {"negative": .99, "neutral": .007, "positive": .003},
             "predicted_probability_uncalibrated": .99, "error": None},
        ], "ate_extraction_performed": True, "atsc_predictions_performed": True,
        "training_performed": False, "gold_test_evaluation_performed": False, "weights_modified": False,
    }


def fake_streamlit(*, text: str = REVIEW, submitted: bool = False, state: dict | None = None):
    """Streamlit facade that records every UI call without a browser/server."""
    fake = mock.MagicMock()
    fake.session_state = state if state is not None else {}
    fake.text_area.return_value = text
    fake.form_submit_button.return_value = submitted
    fake.selectbox.return_value = "cpu"
    fake.form.return_value = contextlib.nullcontext()
    fake.spinner.return_value = contextlib.nullcontext()
    fake.container.return_value = contextlib.nullcontext()
    fake.expander.return_value = contextlib.nullcontext()
    fake.columns.return_value = (mock.MagicMock(), mock.MagicMock(), mock.MagicMock())
    return fake


class HighlightTests(unittest.TestCase):
    def test_food_service_precise_offsets(self):
        html, marked, skipped = app.highlight_review(REVIEW, ready_report()["aspects"])
        self.assertIn('<mark class="absa-positive">food</mark>', html)
        self.assertIn('<mark class="absa-negative">service</mark>', html)
        self.assertEqual((marked, skipped), (2, 0))
        self.assertIn("but the ", html)

    def test_preserves_literal_newlines_and_spaces(self):
        review = "  The\nfood  was great.\r\n"
        row = {"aspect": {"start": 6, "end": 10, "text": "food"}, "status": "CLASSIFIED",
               "sentiment": "positive"}
        html, marked, skipped = app.highlight_review(review, [row])
        self.assertIn("  The\n", html)
        self.assertIn("  was great.\r\n", html)
        self.assertEqual((marked, skipped), (1, 0))

    def test_unicode_codepoint_offsets(self):
        review = "😀 food 😀"
        row = {"aspect": {"text": "food", "start": 2, "end": 6},
               "status": "CLASSIFIED", "sentiment": "neutral"}
        html, marked, skipped = app.highlight_review(review, [row])
        self.assertIn('😀 <mark class="absa-neutral">food</mark> 😀', html)
        self.assertEqual((marked, skipped), (1, 0))

    def test_non_bmp_inside_target(self):
        review = "a😀b"
        row = {"aspect": {"text": "😀", "start": 1, "end": 2},
               "status": "CLASSIFIED", "sentiment": "positive"}
        html, marked, _ = app.highlight_review(review, [row])
        self.assertIn('a<mark class="absa-positive">😀</mark>b', html)
        self.assertEqual(marked, 1)

    def test_repeated_aspect_distinct_occurrences(self):
        review = "service great, service awful"
        a = review.find("service")
        b = review.rfind("service")
        rows = [{"aspect": {"text": "service", "start": a, "end": a + 7},
                 "status": "CLASSIFIED", "sentiment": "positive"},
                {"aspect": {"text": "service", "start": b, "end": b + 7},
                 "status": "CLASSIFIED", "sentiment": "negative"}]
        html, marked, skipped = app.highlight_review(review, rows)
        self.assertEqual((marked, skipped), (2, 0))
        self.assertIn('class="absa-positive">service', html)
        self.assertIn('class="absa-negative">service', html)

    def test_out_of_order_spans_sorted_only_for_render(self):
        rows = ready_report()["aspects"][::-1]
        before = copy.deepcopy(rows)
        html, marked, skipped = app.highlight_review(REVIEW, rows)
        self.assertTrue(html.find('>food</mark>') < html.find('>service</mark>'))
        self.assertEqual(rows, before)
        self.assertEqual((marked, skipped), (2, 0))

    def test_duplicate_span_ignored_without_rewriting(self):
        first = ready_report()["aspects"][0]
        html, marked, skipped = app.highlight_review(REVIEW, [first, first])
        self.assertEqual((marked, skipped), (1, 1))
        self.assertEqual(html.count('<mark'), 1)

    def test_overlap_skip(self):
        review = "foodie"
        rows = [{"aspect": {"text": "food", "start": 0, "end": 4}},
                {"aspect": {"text": "oodie", "start": 1, "end": 6}}]
        html, marked, skipped = app.highlight_review(review, rows)
        self.assertEqual((marked, skipped), (1, 1))
        self.assertIn('>food</mark>ie', html)

    def test_unvalidated_span_skipped(self):
        for bad in [
            {"text": "food", "start": -1, "end": 8},
            {"text": "wrong", "start": 4, "end": 8},
            {"text": "food", "start": True, "end": 8},
            {"text": "food", "start": 4, "end": 88},
            {"text": "food", "start": 8, "end": 4},
            {"text": "food", "start": 4, "end": 4},
            {"text": "food", "start": 4.0, "end": 8},
            {"text": "food", "start": 4, "end": False},
        ]:
            with self.subTest(bad=bad):
                html, marked, skipped = app.highlight_review(REVIEW, [{"aspect": bad}])
                self.assertEqual((marked, skipped), (0, 1))
                self.assertNotIn('<mark', html)

    def test_invalid_rows_do_not_crash(self):
        html, marked, skipped = app.highlight_review("ok", [42, None, {"aspect": [1, 2]}])
        self.assertEqual((marked, skipped), (0, 3))
        self.assertIn('ok', html)

    def test_empty_review_no_aspects(self):
        html, marked, skipped = app.highlight_review("", [])
        self.assertIn('class="absa-review"', html)
        self.assertEqual((marked, skipped), (0, 0))

    def test_bad_types_rejected(self):
        with self.assertRaises(TypeError):
            app.highlight_review(None, [])
        with self.assertRaises(TypeError):
            app.highlight_review("x", {})

    def test_untrusted_html_review_escaped(self):
        review = '<script>alert("xss")</script> food & <img src=x onerror=alert(1)>'
        start = review.index("food")
        row = {"aspect": {"text": "food", "start": start, "end": start + 4},
               "status": "CLASSIFIED", "sentiment": "positive"}
        html, marked, skipped = app.highlight_review(review, [row])
        self.assertNotIn("<script>", html)
        self.assertNotIn("<img", html)
        self.assertIn('&lt;script&gt;', html)
        self.assertIn('&amp;', html)
        self.assertIn('class="absa-positive">food</mark>', html)
        self.assertEqual((marked, skipped), (1, 0))

    def test_untrusted_aspect_escaped(self):
        review = 'The <b>food</b> was great'
        start = review.index('<b>')
        end = review.index('</b>') + len('</b>')
        row = {"aspect": {"text": review[start:end], "start": start, "end": end},
               "status": "CLASSIFIED", "sentiment": "positive"}
        html, _, _ = app.highlight_review(review, [row])
        self.assertIn('&lt;b&gt;food&lt;/b&gt;', html)
        self.assertNotIn('<b>food</b>', html)

    def test_unknown_sentiment_has_safe_class(self):
        row = {"aspect": {"text": "food", "start": 4, "end": 8},
               "status": "CLASSIFIED", "sentiment": 'positive" onmouseover="evil()'}
        html, _, _ = app.highlight_review(REVIEW, [row])
        self.assertIn('class="absa-unknown"', html)
        self.assertNotIn('onmouseover', html)

    def test_error_aspect_highlighted_unknown(self):
        row = {"aspect": {"text": "food", "start": 4, "end": 8}, "status": "ERROR",
               "sentiment": "positive"}
        html, _, _ = app.highlight_review(REVIEW, [row])
        self.assertIn('class="absa-unknown"', html)


class PresentationContractTests(unittest.TestCase):
    def test_softmax_valid(self):
        self.assertEqual(app.valid_softmax({"negative": .1, "neutral": .2, "positive": .7}),
                         {"negative": .1, "neutral": .2, "positive": .7})

    def test_softmax_bad_payloads(self):
        cases = [None, [], {}, {"negative": .1, "neutral": .2},
                 {"negative": .1, "neutral": .2, "positive": .7, "other": 0},
                 {"negative": .1, "neutral": .2, "positive": .9},
                 {"negative": -0.1, "neutral": .1, "positive": 1.},
                 {"negative": float('nan'), "neutral": 0., "positive": 1.},
                 {"negative": float('inf'), "neutral": 0., "positive": 1.},
                 {"negative": True, "neutral": 0., "positive": 1.},
                 {"negative": "0.1", "neutral": 0.2, "positive": .7},
                 {"negative": 0., "neutral": 0., "positive": 1.1}]
        for obj in cases:
            with self.subTest(obj=str(obj)):
                self.assertIsNone(app.valid_softmax(obj))

    def test_budget_ready(self):
        self.assertEqual(app.format_budget({"used": 17, "limit": 128, "truncated": False}),
                         "ATSC tokens: 17/128 · truncated: false")

    def test_budget_refuses_bool(self):
        self.assertIn("không có", app.format_budget({"used": True, "limit": 128, "truncated": False}))

    def test_budget_missing(self):
        self.assertIn("không có", app.format_budget(None))

    def test_overflow_details_from_adapter(self):
        err = {"code": "ATSC_TOKEN_OVERFLOW", "details": {"required_tokens": 130,
               "max_length": 128, "excess_tokens": 2, "truncation": False}}
        self.assertEqual(app.format_overflow(err), "ATSC cần 130/128 tokens (+2); không truncation.")

    def test_overflow_invalid_data_no_fake_metrics(self):
        err = {"code": "ATSC_TOKEN_OVERFLOW", "details": {"required_tokens": 129,
               "max_length": 128, "excess_tokens": 200, "truncation": False}}
        self.assertIsNone(app.format_overflow(err))
        self.assertIsNone(app.format_overflow({"code": "ATE_INFERENCE_FAILED"}))

    def test_json_unicode_roundtrip(self):
        report = {"review": "Đồ ăn rất ngon 😀", "aspects": [], "status": "NO_ASPECTS"}
        s = app.serialize_report(report)
        self.assertIn("Đồ ăn rất ngon 😀", s)
        self.assertEqual(json.loads(s), report)

    def test_json_reject_nan(self):
        with self.assertRaises(ValueError):
            app.serialize_report({"confidence": float("nan")})

    def test_json_no_result_mutation(self):
        r = ready_report()
        original = copy.deepcopy(r)
        app.serialize_report(r)
        self.assertEqual(original, r)

    def test_default_is_restaurant_review(self):
        self.assertIn('food', app.DEFAULT_REVIEW)
        self.assertIn('service', app.DEFAULT_REVIEW)

    def test_models_paths_target_existing_artifacts(self):
        self.assertEqual(app.ATE_DIR.relative_to(app.ROOT).as_posix(),
                         'artifacts/ate_training/best_model')
        self.assertEqual(app.ATSC_DIR.relative_to(app.ROOT).as_posix(),
                         'artifacts/atsc_training/best_model')
        self.assertEqual(app.PREPROCESSING_DIR.relative_to(app.ROOT).as_posix(),
                         'artifacts/atsc_preprocessing')

    def test_model_import_lazy(self):
        self.assertNotIn('transformers', app.__dict__)
        self.assertNotIn('torch', app.__dict__)
        self.assertTrue(callable(app._load_existing_pipeline))

    def test_torch_backend_env_set(self):
        self.assertEqual(os.environ['USE_TF'], '0')
        self.assertEqual(os.environ['USE_TORCH'], '1')


class FakeStreamlitUITests(unittest.TestCase):
    def test_no_submission_does_not_load_models(self):
        st = fake_streamlit(submitted=False)
        with mock.patch.object(app, "st", st), \
             mock.patch.object(app, "get_cached_pipeline") as load:
            app.main()
        load.assert_not_called()
        st.download_button.assert_not_called()
        st.set_page_config.assert_called_once()

    def test_blank_review_before_loading(self):
        st = fake_streamlit(text="  \n ", submitted=True)
        with mock.patch.object(app, "st", st), \
             mock.patch.object(app, "get_cached_pipeline") as load:
            app.main()
        load.assert_not_called()
        st.warning.assert_called()

    def test_mocked_smoke_complete_results(self):
        st = fake_streamlit(submitted=True)
        pipeline = mock.Mock()
        pipeline.predict.return_value = ready_report()
        with mock.patch.object(app, "st", st), \
             mock.patch.object(app, "get_cached_pipeline", return_value=pipeline) as load:
            app.main()
        load.assert_called_once_with(str(app.ATE_DIR), str(app.ATSC_DIR),
                                     str(app.PREPROCESSING_DIR), "cpu")
        pipeline.predict.assert_called_once_with(REVIEW)
        self.assertEqual(st.session_state["absa_last_report"]["status"], "COMPLETE")
        st.success.assert_called_once()
        self.assertEqual(st.progress.call_count, 6)
        self.assertEqual(st.download_button.call_count, 1)
        data = st.download_button.call_args.kwargs["data"]
        self.assertEqual(json.loads(data)["review"], REVIEW)
        self.assertTrue(any(k.kwargs.get('unsafe_allow_html') is True and 'absa-positive' in k.args[0]
                            for k in st.markdown.call_args_list))

    def test_previous_result_rerun_does_not_reload_models(self):
        state = {"absa_last_report": ready_report()}
        st = fake_streamlit(submitted=False, state=state)
        with mock.patch.object(app, "st", st), \
             mock.patch.object(app, "get_cached_pipeline") as load:
            app.main()
        load.assert_not_called()
        st.download_button.assert_called_once()

    def test_new_review_replaces_old_report(self):
        text = "The food is good!"
        state = {"absa_last_report": ready_report()}
        st = fake_streamlit(text=text, submitted=True, state=state)
        pipe = mock.Mock()
        report = {"review": text, "status": "NO_ASPECTS", "summary": {
            "extracted": 0, "classified": 0, "errors": 0}, "aspects": []}
        pipe.predict.return_value = report
        with mock.patch.object(app, "st", st), mock.patch.object(app, "get_cached_pipeline", return_value=pipe):
            app.main()
        self.assertEqual(st.session_state['absa_last_report']['review'], text)
        st.info.assert_called_once()

    def test_blank_review_clears_stale(self):
        state = {"absa_last_report": ready_report()}
        st = fake_streamlit(text=' ', submitted=True, state=state)
        with mock.patch.object(app, 'st', st):
            app.main()
        self.assertNotIn('absa_last_report', st.session_state)

    def test_load_exception_clears_stale(self):
        state = {"absa_last_report": ready_report()}
        st = fake_streamlit(submitted=True, state=state)
        with mock.patch.object(app, 'st', st), \
             mock.patch.object(app, 'get_cached_pipeline', side_effect=FileNotFoundError('model')):
            app.main()
        self.assertNotIn('absa_last_report', st.session_state)
        st.error.assert_called()

    def test_malformed_report_is_not_shown(self):
        state = {}
        st = fake_streamlit(submitted=True, state=state)
        pipe = mock.Mock()
        pipe.predict.return_value = {"review": "changed", "aspects": []}
        with mock.patch.object(app, 'st', st), mock.patch.object(app, 'get_cached_pipeline', return_value=pipe):
            app.main()
        self.assertNotIn('absa_last_report', st.session_state)
        st.error.assert_called()

    def test_partial_with_structured_overflow(self):
        st = fake_streamlit()
        report = ready_report()
        report["status"] = 'PARTIAL'
        report['aspects'][1] = {
            "index": 1, "status": "ERROR",
            "aspect": {"text": "service", "start": 32, "end": 39},
            "token_budget": None, "sentiment": None,
            "probabilities_softmax_uncalibrated": None,
            "error": {"code": "ATSC_TOKEN_OVERFLOW", "message": "Too long", "details": {
                "required_tokens": 130, "max_length": 128, "excess_tokens": 2, "truncation": False}}}
        with mock.patch.object(app, 'st', st):
            app.render_report(report)
        st.warning.assert_called()
        self.assertEqual(st.progress.call_count, 3)
        self.assertTrue(any('130/128 tokens' in c.args[0] for c in st.caption.call_args_list))
        self.assertTrue(any('ATSC_TOKEN_OVERFLOW' in c.args[0] for c in st.error.call_args_list))

    def test_global_ate_error(self):
        st = fake_streamlit()
        report = {"review": REVIEW, "status": "ATE_ERROR", "summary": {
            "extracted": 0, "classified": 0, "errors": 0, "global_errors": 1},
            "error": {"code": "ATE_TOKEN_OVERFLOW", "message": "No input was truncated", "details": {}},
            "aspects": []}
        with mock.patch.object(app, 'st', st):
            app.render_report(report)
        self.assertTrue(any('ATE_TOKEN_OVERFLOW' in c.args[0] for c in st.error.call_args_list))
        st.progress.assert_not_called()

    def test_failed_no_fake_probabilities(self):
        st = fake_streamlit()
        row = {"index": 0, "status": "ERROR", "aspect": {"text": "food", "start": 4, "end": 8},
               "sentiment": None, "error": {"code": "ATSC_INFERENCE_FAILED", "message": "model error"}}
        with mock.patch.object(app, 'st', st):
            app.show_aspect(row, 0)
        st.progress.assert_not_called()
        st.error.assert_called_once()

    def test_invalid_probs_show_error_without_bars(self):
        st = fake_streamlit()
        row = ready_report()['aspects'][0]
        row['probabilities_softmax_uncalibrated']['negative'] = math.nan
        with mock.patch.object(app, 'st', st):
            app.show_aspect(row, 0)
        st.error.assert_called_once()
        st.progress.assert_not_called()

    def test_render_no_aspects(self):
        st = fake_streamlit()
        report = {"review": REVIEW, "status": "NO_ASPECTS", "summary": {
            "extracted": 0, "classified": 0, "errors": 0}, "aspects": []}
        with mock.patch.object(app, 'st', st):
            app.render_report(report)
        st.info.assert_called_once()
        st.progress.assert_not_called()
        st.download_button.assert_called_once()

    def test_project_relative_model_paths_not_cwd(self):
        self.assertTrue(app.ATE_DIR.is_absolute())
        self.assertEqual(app.ATE_DIR.parent.name, 'ate_training')


class CacheWiringTests(unittest.TestCase):
    def test_streamlit_cache_resource_decorator_applied(self):
        """Reload with a fake import to verify st.cache_resource wiring."""
        fake = types.ModuleType('streamlit')
        seen = []
        def decorate(**kwargs):
            seen.append(kwargs)
            return lambda f: f
        fake.cache_resource = decorate
        path = str(ROOT / 'app.py')
        spec = importlib.util.spec_from_file_location('_step5_1_cache_probe', path)
        module = importlib.util.module_from_spec(spec)
        with mock.patch.dict(sys.modules, {'streamlit': fake}):
            spec.loader.exec_module(module)
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0], {'show_spinner': False, 'max_entries': 2})
        self.assertTrue(callable(module.get_cached_pipeline))

    def test_loader_forwards_only_local_paths(self):
        fake_module = types.ModuleType('absa_pipeline')
        fake_module.load_pipeline = mock.Mock(return_value=object())
        with mock.patch.dict(sys.modules, {'absa_pipeline': fake_module}):
            result = app._load_existing_pipeline('ate', 'atsc', 'manifest', 'cpu')
        fake_module.load_pipeline.assert_called_once_with(
            ate_model_dir='ate', atsc_model_dir='atsc', preprocessing_dir='manifest', device='cpu')
        self.assertIsNotNone(result)


@unittest.skipUnless(os.environ.get('ABSA_STREAMLIT_REAL_SMOKE') == '1',
                     'Real model smoke opt-in: set ABSA_STREAMLIT_REAL_SMOKE=1')
class RealLocalCheckpointSmoke(unittest.TestCase):
    """Runs only with local ATE/ATSC checkpoints, never with any test split."""
    def test_existing_pipeline_on_known_review(self):
        for path in (app.ATE_DIR, app.ATSC_DIR, app.PREPROCESSING_DIR):
            self.assertTrue(path.is_dir(), f'Missing local artifact: {path}')
        pipe = app.get_cached_pipeline(str(app.ATE_DIR), str(app.ATSC_DIR),
                                       str(app.PREPROCESSING_DIR), 'cpu')
        result = pipe.predict(REVIEW)
        self.assertEqual(result['status'], 'COMPLETE')
        self.assertEqual(result['review'], REVIEW)
        self.assertIs(result['review_modified'], False)
        self.assertEqual(result['summary']['classified'], 2)
        self.assertEqual(result['summary']['errors'], 0)
        expectations = {'food': ('positive', 4, 8), 'service': ('negative', 32, 39)}
        for row in result['aspects']:
            aspect = row['aspect']
            name = aspect['text']
            self.assertIn(name, expectations)
            sentiment, start, end = expectations[name]
            self.assertEqual(row['status'], 'CLASSIFIED')
            self.assertEqual((row['sentiment'], aspect['start'], aspect['end']),
                             (sentiment, start, end))
            self.assertEqual(result['review'][start:end], name)
            self.assertLessEqual(row['token_budget']['used'], 128)
            self.assertIs(row['token_budget']['truncated'], False)
            self.assertIsNone(row['error'])
            self.assertIsNotNone(app.valid_softmax(row['probabilities_softmax_uncalibrated']))
        # Rendering helper uses these exact model-produced offsets.
        highlighted, n, skipped = app.highlight_review(result['review'], result['aspects'])
        self.assertEqual((n, skipped), (2, 0))
        self.assertIn('>food</mark>', highlighted)
        self.assertIn('>service</mark>', highlighted)


if __name__ == '__main__':
    unittest.main(verbosity=2)
