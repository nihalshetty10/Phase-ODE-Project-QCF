# Neural ODE SPY Forecasting and Backtest

An out-of-sample test of whether a compact Neural ODE can forecast the next daily return of SPY. The model is compared with an LSTM, a ridge autoregression, buy-and-hold, and cash.

**Interactive result:** [nihalshetty10.github.io/Phase-ODE-Project-QCF](https://nihalshetty10.github.io/Phase-ODE-Project-QCF/)

## Result

The Neural ODE earned 13.3% in the locked 2023–2024 test period, compared with 57.6% for buy-and-hold. Its maximum drawdown was 3.7%, but average gross exposure was only 19%. The result is therefore useful as a forecasting and risk-control experiment, not as evidence of market outperformance.

- Neural ODE: 13.3% return, 1.03 Sharpe, -3.7% maximum drawdown
- LSTM: no trades at its validation-selected signal threshold
- Ridge AR(20): -20.0% return
- Buy-and-hold: 57.6% return, 1.85 Sharpe, -10.0% maximum drawdown

## Experimental design

- **Data:** SPY auto-adjusted close from Yahoo Finance
- **Features:** 20 trailing daily log returns
- **Training period:** February 3, 2015 through December 31, 2021
- **Validation period:** calendar year 2022
- **Locked test period:** January 3, 2023 through December 31, 2024
- **Execution convention:** information through close *t* determines the position for return *t* to *t + 1*
- **Trading cost:** 5 basis points for each unit of turnover
- **Reproducibility:** fixed seed and deterministic PyTorch operations

Validation data controls early stopping and each model's neutral signal threshold. Test observations are not used for training or model selection.

## Repository layout

~~~text
.
├── assets/
│   ├── css/research-note.css
│   ├── data/backtest-results.json
│   └── js/backtest-dashboard.js
├── data/
│   └── spy_adjusted_close_2015_2024.csv
├── src/
│   └── build_backtest_results.py
├── tests/
│   └── test_backtest_pipeline.py
├── index.html
└── requirements.txt
~~~

The static website reads only the generated JSON artifact. It does not create sample returns or metrics in the browser.

## Reproduce the analysis

~~~powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
python src/build_backtest_results.py
python -m unittest discover -s tests -v
python -m http.server 8000
~~~

Then open `http://localhost:8000`.

The default build uses the checked-in data snapshot. To download the same fixed date range again:

~~~powershell
python src/build_backtest_results.py --refresh-data
~~~

## Model definitions

**Neural ODE:** A linear encoder maps the 20-return window to a 16-dimensional latent state. A learned vector field evolves that state over continuous depth using four differentiable RK4 steps. A final linear layer predicts the next log return.

**LSTM:** A one-layer recurrent network with 16 hidden units trained on the same input window and target.

**Ridge AR(20):** A regularized linear model using the same lagged returns. Its regularization strength is selected using validation error.

## Limitations

This is one ETF and one two-year test regime. The backtest omits bid-ask spread, market impact, financing, short-borrow constraints, and taxes. The fixed transaction charge is an approximation. No statistical-significance or live-performance claim is made.

## Authors

Nihal Shetty and Neil Mascarenhas.
