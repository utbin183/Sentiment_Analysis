
# Restaurant Aspect-Based Sentiment Analysis

An Aspect-Based Sentiment Analysis (ABSA) system built using
DistilBERT and Streamlit.

## Features

- Aspect Term Extraction (ATE)
- Aspect Term Sentiment Classification (ATSC)
- Sentiment classification: Negative, Neutral, Positive
- Aspect highlighting
- Interactive Streamlit application

## Architecture

Review
→ ATE (DistilBERT)
→ Offset-preserving Adapter
→ ATSC (DistilBERT)
→ Sentiment Results

## Installation

Python 3.12 recommended.

```bash
pip install -r requirements.txt
```

## Run locally

```bash
streamlit run app.py
```

## Models

Fine-tuned DistilBERT models are stored under:

- artifacts/ate_training/best_model
- artifacts/atsc_training/best_model

## Dataset

thainq107/abte-restaurants

## Notes

The application is designed for English restaurant reviews.
Model predictions are not guaranteed to be correct.
