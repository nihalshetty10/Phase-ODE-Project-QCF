# Neural ODE Trading Strategy

This implementation extends the Neural ODE model with a trading strategy that generates buy, hold, sell, and short signals based on the model's price predictions.

## How It Works

The trading strategy follows these rules:

1. **If we have no holdings and the model predicts the stock will go up:** BUY
2. **If we have a holding and the model predicts up:** HOLD
3. **If we have a holding and the model predicts down:** SELL
4. **If the model predicts down and we have no holdings:** SHORT
5. **If we are short and the model predicts up:** BUY_TO_COVER (close short position)

## Files

- `neural_ode_trading_strategy.py`: The main module implementing the trading logic
- `neural_ode_trading_example.py`: Example code to integrate with your Neural ODE notebook

## How to Use

1. Save both Python files in the same directory as your Neural ODE notebook
2. Add the example code cells from `neural_ode_trading_example.py` to your notebook after your prediction code
3. Run the cells to generate and analyze the trading signals

## Features

- Generates trading signals based on Neural ODE predictions
- Calculates portfolio value over time
- Compares strategy performance with buy-and-hold
- Visualizes results with detailed charts
- Backtests different threshold values for signal generation
- Analyzes trade statistics and drawdowns
- Saves trading signals to CSV for further analysis

## Example Output

The visualization will show:
- Actual vs. predicted prices
- Trading signals plotted on the price chart
- Portfolio value compared to buy-and-hold

The statistics will include:
- Strategy return
- Buy-and-hold return
- Alpha (excess return)
- Total trades
- Win rate
- Average profit per trade
- Maximum drawdown

## Advanced Usage

You can experiment with different thresholds for signal generation:

```python
# Backtest different threshold values
thresholds = [0.0, 0.1, 0.2, 0.5, 1.0]
backtest_results = ts.backtest_parameters(actual_prices, predicted_prices, thresholds)
print(backtest_results)
```

A threshold of 0.5 means the model must predict at least a 0.5% price change to generate a signal.

## Note on Short Selling

The implementation assumes simple short selling with 1x leverage. In a real trading environment, you would need to account for margin requirements, borrowing costs, and other factors related to short selling. 