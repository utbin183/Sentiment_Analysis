"""Step 5.2 — Streamlit UI/UX and robustness tests, inference-only.

All tests are deterministic. They do not import real Transformers, read gold test
splits, train, write checkpoints, or run a browser. Browser tests are manual.
"""
from __future__ import annotations

import contextlib
import copy
import json
from pathlib import Path
import re
import unittest
from unittest import mock

import app
from tests.test_streamlit_app import fake_streamlit, ready_report, REVIEW


def make_review_report(review, aspects, status='COMPLETE', *, errors=0):
    return {
        'phase': '4.3', 'review': review, 'review_modified': False,
        'offset_contract': 'python_unicode_codepoints_half_open',
        'max_atsc_tokens': 128,
        'status': status,
        'summary': {'extracted': len(aspects), 'classified': len(aspects) - errors,
                    'errors': errors, 'global_errors': 0},
        'aspects': aspects,
    }


def classified(review, text, begin, label, idx, p=.98):
    probs = {v: (1.0-p)/2 for v in app.SENTIMENTS}
    probs[label] = p
    return {
        'index': idx, 'status': 'CLASSIFIED',
        'aspect': {'text': text, 'start': begin, 'end': begin+len(text),
                   'ate_confidence_uncalibrated': .92},
        'token_budget': {'used': 17, 'limit': 128, 'truncated': False},
        'sentiment': label, 'label_id': app.SENTIMENTS.index(label),
        'probabilities_softmax_uncalibrated': probs,
        'predicted_probability_uncalibrated': p, 'error': None,
    }


def overflow(review, text, begin, idx, *, required=130):
    return {
        'index': idx, 'status': 'ERROR',
        'aspect': {'text': text, 'start': begin, 'end': begin + len(text)},
        'token_budget': None, 'sentiment': None, 'label_id': None,
        'probabilities_softmax_uncalibrated': None,
        'predicted_probability_uncalibrated': None,
        'error': {'code': 'ATSC_TOKEN_OVERFLOW', 'message': 'Review too long',
                  'details': {'required_tokens': required, 'max_length': 128,
                              'excess_tokens': required-128, 'truncation': False}},
    }


class ResponsiveDesignTests(unittest.TestCase):
    def test_opaque_background_defined(self):
        self.assertIn('.stApp, [data-testid="stAppViewContainer"]', app.STYLE)
        self.assertIn('background-color: #0e1117 !important', app.STYLE)
        self.assertIn('background: #181e29', app.STYLE)

    def test_metric_grid_is_responsive(self):
        self.assertIn('grid-template-columns: repeat(3,minmax(0,1fr))', app.STYLE)
        self.assertIn('grid-template-columns: repeat(2,minmax(0,1fr))', app.STYLE)
        self.assertIn('grid-template-columns: 1fr', app.STYLE)

    def test_summary_html_escapes_untrusted_values_by_rejection(self):
        markup = app.summary_markup({'extracted': '<img src=x>', 'classified': 2, 'errors': False})
        self.assertNotIn('<img', markup)
        self.assertIn('>2</div>', markup)
        self.assertEqual(markup.count('absa-stat-number'), 3)

    def test_summary_nonnegative_only(self):
        markup = app.summary_markup({'extracted': -1, 'classified': 0, 'errors': 99})
        self.assertIn('>0</div>', markup)
        self.assertIn('>99</div>', markup)
        self.assertNotIn('>-1</div>', markup)

    def test_mobile_breakpoints(self):
        self.assertIn('@media (max-width: 700px)', app.STYLE)
        self.assertIn('@media (max-width: 380px)', app.STYLE)
        self.assertIn('max-width: 100%', app.STYLE)
        self.assertIn('overflow-wrap: anywhere', app.STYLE)

    def test_reduced_motion(self):
        self.assertIn('prefers-reduced-motion: reduce', app.STYLE)

    def test_legend_static_and_not_user_controlled(self):
        self.assertIn('aria-label="Chú giải sentiment"', app.LEGEND_HTML)
        self.assertIn('Positive', app.LEGEND_HTML)
        self.assertIn('Negative', app.LEGEND_HTML)
        self.assertIn('Neutral', app.LEGEND_HTML)
        self.assertIn('Chưa phân loại', app.LEGEND_HTML)
        self.assertNotIn('%s', app.LEGEND_HTML)

    def test_css_does_not_embed_review(self):
        self.assertNotIn(REVIEW, app.STYLE)
        self.assertNotIn(REVIEW, app.LEGEND_HTML)

    def test_short_and_long_text_no_preprocessing(self):
        review = '  😀\n' + 'verylongtoken' * 20 + '  \r\n'
        markup, n, skipped = app.highlight_review(review, [])
        self.assertIn('  😀\n', markup)
        self.assertIn('  \r\n', markup)
        self.assertEqual((n, skipped), (0, 0))

    def test_html_attack_never_in_markup_unescaped(self):
        review = '<svg onload=alert(1)>  service'
        begin = review.index('service')
        row = classified(review, 'service', begin, 'negative', 0)
        markup, n, skipped = app.highlight_review(review, [row])
        self.assertNotIn('<svg', markup)
        self.assertIn('&lt;svg', markup)
        self.assertEqual((n, skipped), (1, 0))


class DisplaySafetyTests(unittest.TestCase):
    def test_eligible_real_shaped_predictions(self):
        row = ready_report()['aspects'][0]
        out = app.presentation_prediction(row)
        self.assertIsNotNone(out)
        self.assertEqual(out[0], 'positive')

    def test_reject_sentiment_different_from_argmax(self):
        row = ready_report()['aspects'][0]
        row['sentiment'] = 'negative'
        self.assertIsNone(app.presentation_prediction(row))

    def test_reject_label_id_mismatch(self):
        row = ready_report()['aspects'][0]
        row['label_id'] = 0
        self.assertIsNone(app.presentation_prediction(row))

    def test_reject_nonint_label_id(self):
        row = ready_report()['aspects'][0]
        row['label_id'] = True
        self.assertIsNone(app.presentation_prediction(row))

    def test_reject_incorrect_winning_probability(self):
        row = ready_report()['aspects'][0]
        row['predicted_probability_uncalibrated'] = .2
        self.assertIsNone(app.presentation_prediction(row))

    def test_reject_failed_status_even_with_probs(self):
        row = ready_report()['aspects'][0]
        row['status'] = 'ERROR'
        self.assertIsNone(app.presentation_prediction(row))

    def test_reject_missing_probs(self):
        row = ready_report()['aspects'][0]
        row['probabilities_softmax_uncalibrated'] = None
        self.assertIsNone(app.presentation_prediction(row))

    def test_reject_nan_softmax(self):
        row = ready_report()['aspects'][0]
        row['probabilities_softmax_uncalibrated']['positive'] = float('nan')
        self.assertIsNone(app.presentation_prediction(row))

    def test_display_refuses_fake_prediction(self):
        st = fake_streamlit()
        row = ready_report()['aspects'][0]
        row['sentiment'] = 'negative'   # logits still clearly positive
        with mock.patch.object(app, 'st', st):
            app.show_aspect(row, 0)
        st.progress.assert_not_called()
        self.assertFalse(any('**Sentiment:' in c.args[0] for c in st.markdown.call_args_list))
        st.error.assert_called_once()

    def test_valid_probabilities_draw_only_three_bars(self):
        st = fake_streamlit()
        with mock.patch.object(app, 'st', st):
            app.show_aspect(ready_report()['aspects'][0], 0)
        self.assertEqual(st.progress.call_count, 3)
        st.error.assert_not_called()

    def test_global_report_rejects_bad_status_types(self):
        r = ready_report()
        for invalid in ({'x': 1}, [], 8, None, 'UNKNOWN'):
            with self.subTest(status=invalid):
                altered = copy.deepcopy(r)
                altered['status'] = invalid
                self.assertFalse(app.basic_report_ok(altered, REVIEW))

    def test_global_report_rejects_wrong_review(self):
        r = ready_report()
        r['review'] = 'normalized text'
        self.assertFalse(app.basic_report_ok(r, REVIEW))

    def test_global_report_rejects_bad_schema(self):
        r = ready_report()
        for key, value in (('aspects', None), ('summary', []), ('review_modified', True)):
            with self.subTest(key=key):
                other = copy.deepcopy(r)
                other[key] = value
                self.assertFalse(app.basic_report_ok(other, REVIEW))

    def test_report_not_mutated_by_display(self):
        st = fake_streamlit()
        r = ready_report()
        clone = copy.deepcopy(r)
        with mock.patch.object(app, 'st', st):
            app.render_report(r)
        self.assertEqual(r, clone)


class RobustnessUITests(unittest.TestCase):
    def test_repeated_service_occurrences_marked_separately(self):
        review = 'service great, service terrible'
        r = make_review_report(review, [
            classified(review, 'service', 0, 'positive', 0),
            classified(review, 'service', review.rindex('service'), 'negative', 1),
        ])
        st = fake_streamlit()
        with mock.patch.object(app, 'st', st):
            app.render_report(r)
        self.assertEqual(st.progress.call_count, 6)
        markup = [k.args[0] for k in st.markdown.call_args_list if k.kwargs.get('unsafe_allow_html')]
        self.assertTrue(any(k.count('<mark') == 2 and 'absa-positive' in k and 'absa-negative' in k
                            for k in markup))

    def test_no_aspects_clear_message_no_predictions(self):
        st = fake_streamlit()
        r = make_review_report('The restaurant is open.', [], status='NO_ASPECTS')
        with mock.patch.object(app, 'st', st):
            app.render_report(r)
        st.info.assert_called_once()
        st.progress.assert_not_called()
        self.assertTrue(any('Không có aspect' in c.args[0] for c in st.caption.call_args_list))
        st.download_button.assert_called_once()

    def test_partial_overflow_preserves_success_and_error(self):
        review = REVIEW
        rows = [classified(review, 'food', 4, 'positive', 0),
                overflow(review, 'service', 32, 1, required=130)]
        r = make_review_report(review, rows, 'PARTIAL', errors=1)
        st = fake_streamlit()
        with mock.patch.object(app, 'st', st):
            app.render_report(r)
        self.assertEqual(st.progress.call_count, 3)
        self.assertTrue(any('130/128 tokens' in c.args[0] for c in st.caption.call_args_list))
        self.assertTrue(any('ATSC_TOKEN_OVERFLOW' in c.args[0] for c in st.error.call_args_list))
        blob = st.download_button.call_args.kwargs['data']
        parsed = json.loads(blob)
        self.assertEqual(parsed['aspects'][1]['sentiment'], None)
        self.assertEqual(parsed['aspects'][1]['error']['details']['required_tokens'], 130)

    def test_all_failed_no_fake_probabilities(self):
        review = REVIEW
        r = make_review_report(review, [overflow(review, 'food', 4, 0)], 'FAILED', errors=1)
        st = fake_streamlit()
        with mock.patch.object(app, 'st', st):
            app.render_report(r)
        st.progress.assert_not_called()
        st.error.assert_called()
        blob = st.download_button.call_args.kwargs['data']
        self.assertIsNone(json.loads(blob)['aspects'][0]['sentiment'])

    def test_global_ate_error_no_fake_aspects(self):
        review = REVIEW
        r = make_review_report(review, [], 'ATE_ERROR')
        r['summary']['global_errors'] = 1
        r['error'] = {'code': 'ATE_TOKEN_OVERFLOW', 'message': 'No input was truncated'}
        st = fake_streamlit()
        with mock.patch.object(app, 'st', st):
            app.render_report(r)
        st.progress.assert_not_called()
        self.assertTrue(any('ATE_TOKEN_OVERFLOW' in x.args[0] for x in st.error.call_args_list))

    def test_model_missing_clears_previous_output(self):
        state = {'absa_last_report': ready_report(), 'absa_last_device': 'cpu'}
        st = fake_streamlit(submitted=True, state=state)
        with mock.patch.object(app, 'st', st), \
             mock.patch.object(app, 'get_cached_pipeline', side_effect=FileNotFoundError('best_model missing')):
            app.main()
        self.assertNotIn('absa_last_report', state)
        self.assertNotIn('absa_last_device', state)
        st.error.assert_called_once()
        st.download_button.assert_not_called()

    def test_change_device_without_submit_warns_stale(self):
        state = {'absa_last_report': ready_report(), 'absa_last_device': 'cpu'}
        st = fake_streamlit(submitted=False, state=state)
        st.selectbox.return_value = 'cuda'
        with mock.patch.object(app, 'st', st), \
             mock.patch.object(app, 'get_cached_pipeline') as loader:
            app.main()
        loader.assert_not_called()
        self.assertTrue(any('đã thay đổi' in c.args[0] for c in st.warning.call_args_list))

    def test_change_input_without_submit_warns_stale(self):
        state = {'absa_last_report': ready_report(), 'absa_last_device': 'cpu'}
        st = fake_streamlit(submitted=False, state=state, text='food is good')
        with mock.patch.object(app, 'st', st), mock.patch.object(app, 'get_cached_pipeline') as loader:
            app.main()
        loader.assert_not_called()
        self.assertTrue(any('đã thay đổi' in c.args[0] for c in st.warning.call_args_list))

    def test_blank_review_does_not_load_model(self):
        st = fake_streamlit(submitted=True, text='\n \t ')
        with mock.patch.object(app, 'st', st), mock.patch.object(app, 'get_cached_pipeline') as loader:
            app.main()
        loader.assert_not_called()
        st.warning.assert_called_once()

    def test_input_error_clears_last_report(self):
        state = {'absa_last_report': ready_report(), 'absa_last_device': 'cpu'}
        st = fake_streamlit(submitted=True)
        st.session_state = state
        fake_pipe = mock.Mock()
        fake_pipe.predict.return_value = {'review': REVIEW, 'aspects': [], 'status': 'UNKNOWN'}
        with mock.patch.object(app, 'st', st), \
             mock.patch.object(app, 'get_cached_pipeline', return_value=fake_pipe):
            app.main()
        self.assertNotIn('absa_last_report', state)
        st.error.assert_called_once()
        st.download_button.assert_not_called()

    def test_cached_pipeline_not_loaded_until_submit(self):
        st = fake_streamlit(submitted=False)
        with mock.patch.object(app, 'st', st), mock.patch.object(app, 'get_cached_pipeline') as loader:
            app.main()
        loader.assert_not_called()

    def test_model_load_once_for_one_valid_submit(self):
        st = fake_streamlit(submitted=True)
        pipe = mock.Mock()
        pipe.predict.return_value = ready_report()
        with mock.patch.object(app, 'st', st), \
             mock.patch.object(app, 'get_cached_pipeline', return_value=pipe) as loader:
            app.main()
        loader.assert_called_once()
        pipe.predict.assert_called_once_with(REVIEW)
        self.assertEqual(st.session_state.get('absa_last_device'), 'cpu')

    def test_more_results_not_shown_without_submit(self):
        st = fake_streamlit(submitted=False, state={})
        with mock.patch.object(app, 'st', st):
            app.main()
        st.progress.assert_not_called()
        st.download_button.assert_not_called()

    def test_security_review_not_in_style_or_legend(self):
        st = fake_streamlit(submitted=True, text=REVIEW)
        pipe = mock.Mock()
        pipe.predict.return_value = ready_report()
        with mock.patch.object(app, 'st', st), mock.patch.object(app, 'get_cached_pipeline', return_value=pipe):
            app.main()
        htmls = [c.args[0] for c in st.markdown.call_args_list if c.kwargs.get('unsafe_allow_html')]
        self.assertTrue(any('<mark' in x for x in htmls))
        self.assertTrue(any('<style>' in x for x in htmls))
        self.assertTrue(any('absa-legend' in x for x in htmls))


if __name__ == '__main__':
    unittest.main(verbosity=2)
