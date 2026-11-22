"""Step 5.2 — responsive, robust, inference-only Streamlit UI for restaurant ABSA.

Uses the **existing** src/absa_pipeline.py without modifying checkpoints.
The pipeline contains one ATE model and one ATSC model and is cached as one
resource across Streamlit reruns. No training, test evaluation, or disk writes.

Run from the project root: streamlit run app.py
"""
from __future__ import annotations

import html
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

# Set this before importing project modules / Transformers (Windows Keras 3).
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    import streamlit as st
except ModuleNotFoundError as exc:
    if exc.name != "streamlit":
        raise
    # Pure presentation helpers and mock UI tests remain testable without
    # Streamlit; running the app requires pip install -r requirements/...
    st = None

DEFAULT_REVIEW = "The food was delicious, but the service was terrible."
ATE_DIR = ROOT / "artifacts" / "ate_training" / "best_model"
ATSC_DIR = ROOT / "artifacts" / "atsc_training" / "best_model"
PREPROCESSING_DIR = ROOT / "artifacts" / "atsc_preprocessing"
SENTIMENTS = ("negative", "neutral", "positive")
SENTIMENT_LABELS = {"negative": "Negative", "neutral": "Neutral", "positive": "Positive"}
STATUS_LABELS = {
    "COMPLETE": "Hoàn tất: tất cả aspects đã được phân loại.",
    "PARTIAL": "Một số aspects gặp lỗi; những kết quả hợp lệ vẫn được giữ lại.",
    "FAILED": "Không phân loại thành công aspect nào.",
    "NO_ASPECTS": "Không phát hiện aspect nào trong review.",
    "ATE_ERROR": "ATE gặp lỗi trước khi hoàn thành trích xuất.",
    "PIPELINE_ERROR": "Pipeline gặp lỗi trước khi phân loại aspects.",
}

# All styling originates in this source, NEVER from an unescaped review.
STYLE = """<style>
/* Step 5.2: local, opaque dark theme; avoids background bleed-through. */
html, body, .stApp, [data-testid="stAppViewContainer"],
[data-testid="stHeader"] { background-color: #0e1117 !important; }
[data-testid="stAppViewContainer"] > .main { background-color: #0e1117 !important; }
.block-container { max-width: 940px !important; padding-top: 2rem; padding-bottom: 3.5rem; }
.absa-intro { color: #b4c0d1; font-size: .96rem; line-height: 1.55; }
.absa-review { white-space: pre-wrap; overflow-wrap: anywhere; font-size: 1.02rem;
  line-height: 2.05; padding: 1rem 1.1rem; background: #181e29;
  border: 1px solid #333d4e; border-radius: .75rem; color: #f4f7fc; }
.absa-review mark { color: inherit; border-radius: .22rem;
  padding: .10rem .13rem; font-weight: 700; }
.absa-positive { background-color: #154732; border-bottom: 2px solid #4dd298; }
.absa-negative { background-color: #542633; border-bottom: 2px solid #fc8799; }
.absa-neutral { background-color: #4b3b1b; border-bottom: 2px solid #f0c571; }
.absa-unknown { background-color: #374151; border-bottom: 2px solid #b3bccb; }
.absa-legend { display: flex; flex-wrap: wrap; gap: .45rem .95rem; margin: .65rem 0;
  color: #c6d0df; font-size: .82rem; }
.absa-legend span { display: inline-flex; align-items: center; gap: .38rem; }
.absa-dot { display: inline-block; width: .66rem; height: .66rem; border-radius: .22rem; }
.absa-dot-positive { background: #4dd298; }
.absa-dot-negative { background: #fc8799; }
.absa-dot-neutral { background: #f0c571; }
.absa-dot-unknown { background: #b3bccb; }
/* Summary grid is independent of Streamlit column internals and wraps on mobile. */
.absa-metrics { display: grid; grid-template-columns: repeat(3,minmax(0,1fr));
  gap: .72rem; margin: .85rem 0 1.15rem; }
.absa-stat { min-width: 0; background: #181e29; border: 1px solid #333d4e;
  border-radius: .75rem; padding: .85rem .95rem; }
.absa-stat-label { color: #bdc8da; font-size: .78rem; margin-bottom: .34rem; }
.absa-stat-number { color: #f6f8fe; font-size: 1.75rem; font-weight: 740; line-height: 1.1; }
@media (max-width: 540px) {
  .absa-metrics { grid-template-columns: repeat(2,minmax(0,1fr)); }
  .absa-stat:last-child { grid-column: 1 / -1; }
}
@media (max-width: 380px) {
  .absa-metrics { grid-template-columns: 1fr; }
  .absa-stat:last-child { grid-column: auto; }
}
@media (max-width: 700px) {
  .block-container { max-width: 100% !important; padding: 1.25rem .85rem 2.5rem !important; }
  .absa-review { padding: .8rem; font-size: .96rem; line-height: 1.9; }
  .absa-legend { font-size: .78rem; gap: .4rem .7rem; }
  h1 { font-size: 1.68rem !important; }
  [data-testid="stMetricValue"] { font-size: 1.55rem; }
}
@media (max-width: 380px) {
  .block-container { padding-left: .55rem !important; padding-right: .55rem !important; }
  .absa-review { font-size: .91rem; }
}
@media (prefers-reduced-motion: reduce) {
  .absa-review, .absa-review mark { scroll-behavior: auto; animation: none; transition: none; }
}
</style>"""

# Static HTML, no string interpolation of user-provided review data.
LEGEND_HTML = (
    '<div class="absa-legend" aria-label="Chú giải sentiment">'
    '<span><i class="absa-dot absa-dot-positive"></i>Positive</span>'
    '<span><i class="absa-dot absa-dot-negative"></i>Negative</span>'
    '<span><i class="absa-dot absa-dot-neutral"></i>Neutral</span>'
    '<span><i class="absa-dot absa-dot-unknown"></i>Chưa phân loại</span>'
    '</div>'
)


# UI-only validation; never changes pipeline outputs or model predictions.
def presentation_prediction(row: Mapping[str, Any]) -> tuple[str, dict[str, float]] | None:
    """Reject contradictory/incomplete displayed predictions, never fabricate bars.

    Keep this separate from the inference pipeline: the model already validates
    predictions; this is only a display-safety guard against malformed reports.
    """
    if not isinstance(row, Mapping) or row.get("status") != "CLASSIFIED":
        return None
    sentiment = row.get("sentiment")
    if sentiment not in SENTIMENTS:
        return None
    probs = valid_softmax(row.get("probabilities_softmax_uncalibrated"))
    if probs is None or max(SENTIMENTS, key=probs.__getitem__) != sentiment:
        return None
    ids = {"negative": 0, "neutral": 1, "positive": 2}
    label_id = row.get("label_id")
    if label_id is not None and (type(label_id) is not int or label_id != ids[sentiment]):
        return None
    winner = row.get("predicted_probability_uncalibrated")
    if winner is not None and (type(winner) not in (int, float) or
                               not math.isfinite(winner) or abs(winner - probs[sentiment]) > 1e-4):
        return None
    return sentiment, probs


def summary_markup(summary: Mapping[str, Any]) -> str:
    """Responsive metric cards with strictly numeric, HTML-safe values."""
    if not isinstance(summary, Mapping):
        summary = {}
    entries = (("Extracted", "extracted"), ("Classified", "classified"), ("Errors", "errors"))
    cards: list[str] = []
    for label, key in entries:
        count = summary.get(key)
        value = str(count) if type(count) is int and count >= 0 else "–"
        cards.append('<div class="absa-stat"><div class="absa-stat-label">'
                     + label + '</div><div class="absa-stat-number">'
                     + value + '</div></div>')
    return '<div class="absa-metrics" aria-label="Tổng quan kết quả">' + ''.join(cards) + '</div>'


def basic_report_ok(report: Any, original_review: str) -> bool:
    """Reject malformed global output before showing stale or inconsistent UI."""
    if not isinstance(report, Mapping) or report.get("review") != original_review:
        return False
    status = report.get("status")
    if not isinstance(status, str) or status not in STATUS_LABELS:
        return False
    if not isinstance(report.get("aspects"), list) or not isinstance(report.get("summary"), Mapping):
        return False
    if report.get("review_modified") is True:
        return False
    return True



def _load_existing_pipeline(ate_dir: str, atsc_dir: str, preprocessing_dir: str, device: str) -> Any:
    """Only point where the existing ABSA pipeline is imported/loaded."""
    from absa_pipeline import load_pipeline

    return load_pipeline(ate_model_dir=ate_dir,
                         atsc_model_dir=atsc_dir,
                         preprocessing_dir=preprocessing_dir,
                         device=device)


if st is not None:
    # This caches the single ABSAInferencePipeline object (both loaded models).
    # Identical path/device arguments reuse the instance across reruns/reviews.
    get_cached_pipeline = st.cache_resource(show_spinner=False, max_entries=2)(_load_existing_pipeline)
else:
    get_cached_pipeline = _load_existing_pipeline


def _valid_span(review: str, aspect: Any) -> tuple[int, int] | None:
    """Trust only literal, exact, half-open Python codepoint offsets."""
    if not isinstance(aspect, Mapping):
        return None
    start, end, text = aspect.get("start"), aspect.get("end"), aspect.get("text")
    if type(start) is not int or type(end) is not int or not isinstance(text, str):
        return None
    if not (0 <= start < end <= len(review)) or review[start:end] != text:
        return None
    return start, end


def highlight_review(review: str, rows: Sequence[Any]) -> tuple[str, int, int]:
    """Return safe HTML, number marked, number skipped.

    Never change review, re-tokenize, normalize or derive new offsets.  A
    malformed/overlapping span is skipped (not silently moved or rewritten).
    HTML is only rendered after escaping *every* user-provided character.
    """
    if not isinstance(review, str):
        raise TypeError("review must be str")
    if not isinstance(rows, (list, tuple)):
        raise TypeError("rows must be list or tuple")
    spans: list[tuple[int, int, int, str]] = []
    skipped = 0
    for row_index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            skipped += 1
            continue
        span = _valid_span(review, row.get("aspect"))
        if span is None:
            skipped += 1
            continue
        sentiment = row.get("sentiment") if row.get("status") == "CLASSIFIED" else None
        kind = sentiment if sentiment in SENTIMENTS else "unknown"
        spans.append((span[0], span[1], row_index, kind))

    # Offsets are Python string indexes, NOT JS UTF-16 or UTF-8 byte offsets.
    spans.sort(key=lambda x: (x[0], x[1], x[2]))
    cursor = 0
    parts: list[str] = []
    highlighted = 0
    for start, end, _index, kind in spans:
        if start < cursor:   # duplicate or overlapping aspect
            skipped += 1
            continue
        parts.append(html.escape(review[cursor:start], quote=True))
        safe_aspect = html.escape(review[start:end], quote=True)
        parts.append(f'<mark class="absa-{kind}">{safe_aspect}</mark>')
        cursor = end
        highlighted += 1
    parts.append(html.escape(review[cursor:], quote=True))
    return '<div class="absa-review">' + "".join(parts) + '</div>', highlighted, skipped


def valid_softmax(values: Any) -> dict[str, float] | None:
    """Display only a valid 3-class probability distribution, never calibrate."""
    if not isinstance(values, Mapping) or set(values.keys()) != set(SENTIMENTS):
        return None
    try:
        probs = {name: values[name] for name in SENTIMENTS}
    except (KeyError, TypeError):
        return None
    if any(type(value) not in (float, int) or not math.isfinite(value)
           or not (0.0 <= value <= 1.0) for value in probs.values()):
        return None
    if abs(sum(probs.values()) - 1.0) > 1e-4:
        return None
    return {key: float(value) for key, value in probs.items()}


def format_budget(budget: Any) -> str:
    if not isinstance(budget, Mapping):
        return "ATSC token budget: không có (lỗi trước bước tokenization)."
    used, limit, truncated = budget.get("used"), budget.get("limit"), budget.get("truncated")
    if type(used) is int and type(limit) is int and used >= 0 and limit > 0 and truncated is False:
        return f"ATSC tokens: {used}/{limit} · truncated: false"
    if type(used) is int and type(limit) is int and used >= 0 and limit > 0:
        return f"ATSC tokens: {used}/{limit} · truncated: {str(truncated).lower()}"
    return "ATSC token budget: không có dữ liệu hợp lệ."



def format_overflow(error: Any) -> str | None:
    """Show the full untruncated ATSC budget from structured adapter errors."""
    if not isinstance(error, Mapping) or error.get("code") != "ATSC_TOKEN_OVERFLOW":
        return None
    details = error.get("details")
    if not isinstance(details, Mapping):
        return None
    required, limit, excess = (details.get("required_tokens"),
                               details.get("max_length"), details.get("excess_tokens"))
    if (type(required) is int and type(limit) is int and type(excess) is int
            and required > limit > 0 and excess == required - limit
            and details.get("truncation") is False):
        return f"ATSC cần {required}/{limit} tokens (+{excess}); không truncation."
    return None


def serialize_report(report: Mapping[str, Any]) -> str:
    """Preserve the original report for download; reject NaN and Infinity."""
    return json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def show_status(report: Mapping[str, Any]) -> None:
    status = report.get("status", "UNKNOWN")
    message = STATUS_LABELS.get(status, f"Pipeline trả trạng thái không nhận diện: {status!s}")
    if status == "COMPLETE":
        st.success(message)
    elif status in ("PARTIAL", "NO_ASPECTS"):
        st.warning(message) if status == "PARTIAL" else st.info(message)
    else:
        st.error(message)


def show_error(error: Any) -> None:
    """Render structured errors as plain UI text, never unsafe HTML."""
    if not isinstance(error, Mapping):
        st.error("Lỗi không có cấu trúc hợp lệ.")
        return
    code = str(error.get("code", "UNKNOWN_ERROR"))
    message = str(error.get("message", "Không có mô tả lỗi."))
    st.error(f"{code}: {message}")
    details = error.get("details")
    if isinstance(details, Mapping) and details:
        st.caption("Chi tiết: " + json.dumps(dict(details), ensure_ascii=False, default=str))


def show_aspect(row: Mapping[str, Any], row_number: int) -> None:
    """Render an aspect, including failed ones, without inventing predictions."""
    with st.container(border=True):
        source = row.get("aspect")
        if not isinstance(source, Mapping):
            source = row.get("source_aspect") if isinstance(row.get("source_aspect"), Mapping) else {}
        name = str(source.get("text", "<aspect không hợp lệ>"))
        start, end = source.get("start"), source.get("end")
        st.text(f"Aspect {row_number + 1}: {name}")
        st.caption(f"Python character offsets: [{start}, {end})")
        confidence = source.get("ate_confidence_uncalibrated")
        if type(confidence) in (int, float) and math.isfinite(confidence) and 0 <= confidence <= 1:
            st.caption(f"ATE softmax (chưa hiệu chuẩn): {confidence * 100:.2f}%")
        st.caption(format_budget(row.get("token_budget")))
        if row.get("status") != "CLASSIFIED":
            overflow = format_overflow(row.get("error"))
            if overflow is not None:
                st.caption(overflow)
            show_error(row.get("error"))
            return
        display = presentation_prediction(row)
        if display is None:
            st.error("Kết quả sentiment / softmax không hợp lệ hoặc không nhất quán; không hiển thị dự đoán.")
            return
        sentiment, probs = display
        st.markdown(f"**Sentiment: {SENTIMENT_LABELS[sentiment]}**")
        for label in SENTIMENTS:
            st.progress(probs[label], text=f"{SENTIMENT_LABELS[label]}: {probs[label] * 100:.2f}%")


def render_report(report: Mapping[str, Any]) -> None:
    """Show complete, partial, no-aspects, or global-error results faithfully."""
    st.divider()
    st.subheader("Kết quả phân tích")
    show_status(report)
    summary = report.get("summary")
    if not isinstance(summary, Mapping):
        summary = {}
    st.markdown(summary_markup(summary), unsafe_allow_html=True)
    if report.get("error") is not None:
        show_error(report.get("error"))

    review = report.get("review")
    rows = report.get("aspects")
    if isinstance(review, str) and isinstance(rows, list):
        st.markdown("**Các aspect trong review (offsets giữ nguyên)**")
        markup, highlighted, skipped = highlight_review(review, rows)
        st.markdown(markup, unsafe_allow_html=True)
        if highlighted:
            st.markdown(LEGEND_HTML, unsafe_allow_html=True)
        if skipped:
            st.warning(f"{skipped} span không hợp lệ hoặc chồng lấn, không thể tô nổi bật; vẫn giữ trong JSON.")
        st.caption(f"Đã tô nổi bật {highlighted} aspect; chỉ số tính theo Python Unicode codepoints.")
        if not rows:
            st.caption("Không có aspect để phân loại. Bạn có thể thử một review cụ thể hơn.")
        if rows:
            st.subheader("Sentiment theo từng aspect")
            for index, row in enumerate(rows):
                if isinstance(row, Mapping):
                    show_aspect(row, index)
                else:
                    st.error(f"Aspect ở vị trí {index} không có cấu trúc JSON hợp lệ.")

    st.caption("Softmax và ATE confidence đều chưa được hiệu chuẩn; không phải xác suất dự đoán đúng.")
    with st.expander("JSON gốc (không điều chỉnh)"):
        st.json(report, expanded=False)
    try:
        serialized = serialize_report(report)
    except (TypeError, ValueError) as exc:
        st.error(f"Không thể xuất JSON hợp lệ: {exc}")
    else:
        st.download_button("Tải JSON kết quả", data=serialized, file_name="absa_result.json",
                           mime="application/json", use_container_width=True)


def main() -> None:
    if st is None:
        raise RuntimeError("Streamlit is not installed. Run: pip install -r requirements/requirements_step_5_1.txt")
    st.set_page_config(page_title="Restaurant Review · ABSA", page_icon="🍽️", layout="centered",
                       initial_sidebar_state="collapsed")
    st.markdown(STYLE, unsafe_allow_html=True)
    st.title("Restaurant Review · ABSA")
    st.caption("Aspect Term Extraction (ATE) → Offset-Preserving Adapter → Sentiment Classification (ATSC)")
    st.write("Nhập một review tiếng Anh. Model phát hiện từng khía cạnh và phân loại sentiment của từng khía cạnh.")
    st.caption("Chỉ chạy inference · Review và Python character offsets được giữ nguyên · Không train lại")
    device = st.selectbox("Thiết bị", options=("cpu", "auto", "cuda"), index=0,
                          help="Chọn cpu nếu chưa cài CUDA. Model chỉ được tải sau khi nhấn Analyze.")

    with st.form("absa_review_form", clear_on_submit=False):
        review = st.text_area("Restaurant review (English)", value=DEFAULT_REVIEW,
                              height=150, key="absa_input_review", help="Không sửa dấu câu hoặc khoảng trắng khi tính offsets.")
        submitted = st.form_submit_button("Analyze review", type="primary", use_container_width=True)

    if submitted:
        # Validation before model loading: no stripping/normalizing the input.
        if not isinstance(review, str) or not review.strip():
            st.session_state.pop("absa_last_report", None)
            st.session_state.pop("absa_last_device", None)
            st.warning("Review không được để trống.")
        else:
            st.session_state.pop("absa_last_report", None)
            st.session_state.pop("absa_last_device", None)
            try:
                with st.spinner("Đang tải / sử dụng model đã cache và phân tích review…"):
                    pipeline = get_cached_pipeline(str(ATE_DIR), str(ATSC_DIR),
                                                   str(PREPROCESSING_DIR), device)
                    report = pipeline.predict(review)
                if not basic_report_ok(report, review):
                    raise ValueError("Pipeline returned a malformed report or modified raw review")
                # Store only output JSON, never model objects in session_state.
                st.session_state["absa_last_report"] = dict(report)
                st.session_state["absa_last_device"] = device
            except Exception as exc:
                st.error(f"Không thể chạy ABSA: {type(exc).__name__}: {exc}")
                st.caption("Kiểm tra đường dẫn best_model, tokenizer và thiết bị; kết quả cũ đã được xóa.")

    previous = st.session_state.get("absa_last_report")
    if isinstance(previous, Mapping):
        if not submitted:
            if previous.get("review") != review or st.session_state.get("absa_last_device", device) != device:
                st.warning("Nội dung review hoặc thiết bị đã thay đổi. Nhấn Analyze review để cập nhật kết quả.")
            else:
                st.caption("Đang hiển thị kết quả của lần Analyze gần nhất.")
        render_report(previous)
    st.caption("Demo inference cục bộ. Softmax chưa hiệu chuẩn; không train, không đánh giá lại gold test và không tự ghi review ra file.")


if __name__ == "__main__":
    main()
