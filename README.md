# Actuarial Pricing Engine

An interactive Streamlit dashboard for actuarial pricing automation with end-to-end insurance claims cost prediction.

## Features

- **Data Upload**: Upload train/test CSV files
- **Feature Engineering**: Auto-extract date features (Age, Reporting Lag, Seasonality)
- **Model Training**: XGBoost, LightGBM, CatBoost ensemble with cross-validation
- **Dynamic Pricing**: Adjustable expense loading, profit margin, contingency margin
- **Sensitivity Analysis**: See how pricing parameters affect premiums
- **Export**: Download submission CSV

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Files

| File | Description |
|------|-------------|
| `app.py` | Streamlit dashboard |
| `pricing_engine.py` | Core actuarial logic |
| `requirements.txt` | Dependencies |

## Egyptian UHI Context

This tool supports the mandatory 4-year actuarial review per Law 2/2018 for the Egyptian Universal Health Insurance system.

## Live Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)
