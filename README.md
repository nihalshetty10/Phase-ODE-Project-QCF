# Advanced Financial Trading Strategies Comparison

## Overview

This project implements and compares three advanced financial trading strategies:
1.  **ARCH/GARCH Models**: Utilizes Autoregressive Conditional Heteroskedasticity models to forecast volatility and generate trading signals based on volatility regimes.
2.  **LSTM (Long Short-Term Memory) Networks**: Employs recurrent neural networks to predict future price movements based on historical price sequences.
3.  **Neural Ordinary Differential Equations (Neural ODEs)**: Leverages continuous-depth neural networks to model the underlying dynamics of financial time series for price forecasting. This approach is inspired by the paper "Phase Space Reconstructed Neural Ordinary Differential Equations Model for Stock Price Forecasting" by Nguyen et al.

The project includes individual backtesting for each strategy and a comprehensive comparison framework, visualized through a web interface.

## Features

* **Multiple Models**: Implementation of ARCH/GARCH, LSTM, and Neural ODEs for stock price/direction forecasting.
* **Backtesting Engine**: Robust backtesting functionality to simulate trading strategies and evaluate performance.
* **Performance Metrics**: Calculation of a wide range of metrics, including:
    * Total & Annualized Return
    * Volatility (Daily & Annualized)
    * Sharpe Ratio
    * Maximum Drawdown
    * Win Rate & Profit Factor
    * Sortino Ratio
    * Calmar Ratio
* **Comparative Analysis**: A dedicated script to run all strategies on the same dataset and compare their performance side-by-side.
* **Interactive Web Visualization**: A user-friendly web interface (`web_interface/index.html`) to display:
    * Price predictions vs. actual prices.
    * Trading signals on price charts.
    * Portfolio performance against a Buy & Hold strategy.
    * Drawdown analysis.
    * Distribution of daily returns.
    * Rolling Sharpe ratios.
    * Tabular comparison of all key performance metrics.

## Technologies Used

* **Python 3.x**
* **Core Libraries**:
    * `pandas` for data manipulation.
    * `numpy` for numerical operations.
    * `matplotlib` for plotting.
    * `yfinance` for downloading stock data.
    * `scikit-learn` for data preprocessing (e.g., `MinMaxScaler`).
* **Modeling Libraries**:
    * `arch` for ARCH/GARCH models.
    * `torch` (PyTorch) for LSTM and Neural ODE implementations.
    * `torchdiffeq` for Neural ODE solver.
* **Web Interface**:
    * HTML, CSS, JavaScript
    * `Chart.js` for interactive charts.
    * `Bootstrap` for styling.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/nihalshetty10/Phase-ODE-Project-QCF.git
    cd Phase-ODE-Project-QCF
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Running Individual Strategies

You can run each trading strategy independently:

* **ARCH/GARCH Strategy:**
    ```bash
    python arch_trading_example.py
    ```
    This will load data for 'SPY', fit an ARCH/GARCH model, generate signals, visualize performance, and save results to `arch_trading_signals.csv` and `arch_results.pkl`.

* **LSTM Strategy:**
    ```bash
    python lstm_trading_example.py
    ```
    This will train an LSTM model on 'SPY' data, generate predictions, visualize performance, and save results to `lstm_trading_signals.csv` and `lstm_results.pkl`.

* **Neural ODE Strategy:**
    *(Assuming you've refactored `trading_strategy.py` to `neural_ode_core.py` and potentially created a `run_neural_ode_strategy.py`)*
    ```bash
    # If you have a dedicated runner script:
    python run_neural_ode_strategy.py 
    # Alternatively, the core logic might be directly in neural_ode_core.py:
    # python neural_ode_core.py 
    ```
    This script should handle training or loading a pre-trained Neural ODE model, making predictions, and saving its results (e.g., `neural_ode_results.pkl`). 
    *(You'll need to clarify how this is run after refactoring)*

### Running Model Comparison

To compare all models:
```bash
python model_comparison_example.py