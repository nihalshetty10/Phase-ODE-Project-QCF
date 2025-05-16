# Trading Strategy Model Comparison

This project implements a comparison between Neural Ordinary Differential Equations (Neural ODEs) and Long Short-Term Memory (LSTM) networks for stock price prediction and trading signal generation.

## Overview

The system includes implementations of both Neural ODE and LSTM models for time series prediction, along with a trading strategy framework that generates trading signals based on the predictions. The web interface allows users to compare the performance of both models side by side.

## Models

### Neural ODE Trading Strategy

Neural ODEs extend traditional neural networks by defining continuous-depth models. Instead of specifying discrete layers, Neural ODEs define the derivative of the hidden state with respect to depth, allowing for more flexible modeling of dynamical systems like stock price movements.

Key features:
- Continuous time modeling
- Parameter efficiency
- Adaptive computation
- Memory efficiency during training

### LSTM Trading Strategy

LSTMs are a type of recurrent neural network designed to address the vanishing gradient problem, making them effective for modeling sequential data with long-term dependencies.

Key features:
- Long-term dependency modeling
- Selective memory through gates
- Effective handling of time series data
- Robust to noise and variations in time lags

## Trading Strategy Rules

Both models use the same trading strategy rules:

1. If no holdings and model predicts a price increase: **BUY**
2. If holding a long position and model predicts a price increase: **HOLD**
3. If holding a long position and model predicts a price decrease: **SELL**
4. If no holdings and model predicts a price decrease: **SHORT**
5. If holding a short position and model predicts a price increase: **BUY TO COVER**
6. If holding a short position and model predicts a price decrease: **HOLD**

## Web Interface

The web interface provides a comprehensive comparison between the two models:

- Side-by-side visualization of price predictions
- Portfolio performance charts
- Trading signal visualization
- Drawdown analysis
- Return distribution comparison
- Rolling Sharpe ratio
- Comprehensive performance metrics

## Usage

### Running the Comparison

To run a model comparison:

```python
python model_comparison_example.py
```

This will:
1. Load historical price data for SPY
2. Train both Neural ODE and LSTM models
3. Generate trading signals for both models
4. Calculate performance metrics
5. Create visualization charts
6. Save the results for use in the web interface

### Web Interface

To use the web interface:

1. Start the web server:
   ```
   python -m http.server
   ```

2. Open a browser and navigate to `http://localhost:8000`

3. Enter a ticker symbol and click "Analyze"

4. View the side-by-side comparison of model performance

## Performance Metrics

The system calculates the following performance metrics for each model:

- Total Return (%)
- Annualized Return (%)
- Mean Daily Return (%)
- Daily Volatility (%)
- Annualized Volatility (%)
- Sharpe Ratio
- Maximum Drawdown (%)
- Win Rate (%)
- Profit Factor

Advanced metrics include:

- Calmar Ratio
- Sortino Ratio
- Positive Days (%)
- Max Consecutive Wins
- Max Consecutive Losses

## File Structure

- `neural_ode_trading_strategy.py`: Implementation of the Neural ODE trading strategy
- `lstm_trading_strategy.py`: Implementation of the LSTM trading strategy
- `model_comparison_example.py`: Script to run model comparison
- `index.html`: Web interface
- `script.js`: JavaScript code for the web interface
- `*.pkl`: Saved model comparison results

## Dependencies

- PyTorch
- torchdiffeq (for Neural ODEs)
- pandas
- numpy
- matplotlib
- yfinance
- scikit-learn

## Future Improvements

Potential improvements to the system include:

1. Addition of more advanced models (Transformers, GRUs, etc.)
2. Hyperparameter optimization for each model
3. Ensemble methods combining multiple models
4. More sophisticated trading strategies
5. Risk management features
6. Support for multi-asset portfolio optimization 