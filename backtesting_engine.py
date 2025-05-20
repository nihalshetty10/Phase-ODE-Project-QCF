# backtesting_engine.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def generate_trading_signals(actual_prices, predicted_prices):
    """
    Generate trading signals based on Neural ODE predictions.
    
    Args:
        actual_prices: Array of actual stock prices
        predicted_prices: Array of predicted prices from Neural ODE model
        
    Returns:
        List of trading signals (BUY, SELL, SHORT, BUY_TO_COVER, HOLD)
    """
    signals = []
    position = 0  # 0: no position, 1: long, -1: short
    
    # For the first day, we don't have a prediction yet
    signals.append('HOLD')
    
    for i in range(1, len(predicted_prices)):
        # Predict price change direction for the next day
        price_change = predicted_prices[i] - actual_prices[i-1]
        
        if position == 0:  # No position
            if price_change > 0:  # Model predicts price will go up
                signals.append('BUY')
                position = 1
            elif price_change < 0:  # Model predicts price will go down
                signals.append('SHORT')
                position = -1
            else:  # No change predicted
                signals.append('HOLD')
                
        elif position == 1:  # Long position
            if price_change > 0:  # Model predicts price will go up
                signals.append('HOLD')
            else:  # Model predicts price will go down
                signals.append('SELL')
                position = 0
                
        elif position == -1:  # Short position
            if price_change < 0:  # Model predicts price will go down
                signals.append('HOLD')
            else:  # Model predicts price will go up
                signals.append('BUY_TO_COVER')
                position = 0
    
    return signals

def calculate_portfolio_value(actual_prices, signals, initial_capital=10000.0):
    """
    Calculate portfolio value based on trading signals.
    
    Args:
        actual_prices: Array of actual stock prices
        signals: List of trading signals
        initial_capital: Initial capital amount
        
    Returns:
        Array of portfolio values
    """
    portfolio_value = [initial_capital]
    position = 0  # 0: no position, 1: long, -1: short
    shares = 0
    cash = initial_capital
    
    for i in range(1, len(signals)):
        signal = signals[i]
        price = actual_prices[i]
        
        if signal == 'BUY':
            # Calculate number of shares we can buy with all available cash
            shares = cash / price
            cash = 0
            position = 1
        elif signal == 'SELL' and position == 1:
            # Sell all shares
            cash = shares * price
            shares = 0
            position = 0
        elif signal == 'SHORT':
            # Short selling (borrow shares equal to cash value and sell them)
            # For simplicity, we'll assume short selling 1x leverage
            shares = -cash / price
            cash = cash * 2  # Cash + proceeds from short sale
            position = -1
        elif signal == 'BUY_TO_COVER' and position == -1:
            # Buy to cover short position
            cash = cash - (-shares * price)  # Subtract cost of buying back shares
            shares = 0
            position = 0
            
        # Calculate portfolio value
        if position == 0:
            portfolio_value.append(cash)
        elif position == 1:
            portfolio_value.append(cash + shares * price)
        elif position == -1:
            portfolio_value.append(cash + shares * price)  # Short position value
    
    return portfolio_value

def visualize_trading_strategy(actual_prices, predicted_prices, initial_capital=10000.0):
    """
    Visualize trading strategy performance.
    
    Args:
        actual_prices: Array of actual stock prices
        predicted_prices: Array of predicted prices from Neural ODE model
        initial_capital: Initial capital amount
        
    Returns:
        Dictionary with trading results
    """
    # Generate trading signals
    signals = generate_trading_signals(actual_prices, predicted_prices)
    
    # Calculate portfolio value
    portfolio_value = calculate_portfolio_value(actual_prices, signals, initial_capital)
    
    # Calculate buy and hold strategy
    buy_and_hold = [initial_capital]
    for i in range(1, len(actual_prices)):
        buy_and_hold.append(initial_capital * (actual_prices[i] / actual_prices[0]))
    
    # Calculate strategy performance metrics
    total_return = (portfolio_value[-1] / portfolio_value[0] - 1) * 100
    buy_and_hold_return = (buy_and_hold[-1] / buy_and_hold[0] - 1) * 100
    total_trades = sum(1 for signal in signals if signal in ['BUY', 'SELL', 'SHORT', 'BUY_TO_COVER'])
    
    # Plot strategy performance
    plt.figure(figsize=(14, 10))
    
    # Plot actual vs predicted prices
    plt.subplot(3, 1, 1)
    plt.plot(actual_prices, label='Actual Prices', color='blue')
    plt.plot(predicted_prices, label='Predicted Prices', color='green', linestyle='--')
    plt.title('Actual vs Predicted Prices')
    plt.legend()
    plt.grid(True)
    
    # Plot trading signals on the price chart
    plt.subplot(3, 1, 2)
    plt.plot(actual_prices, color='blue')
    
    for i in range(len(signals)):
        if signals[i] == 'BUY':
            plt.plot(i, actual_prices[i], '^', markersize=10, color='green')
        elif signals[i] == 'SELL':
            plt.plot(i, actual_prices[i], 'v', markersize=10, color='red')
        elif signals[i] == 'SHORT':
            plt.plot(i, actual_prices[i], 'v', markersize=10, color='purple')
        elif signals[i] == 'BUY_TO_COVER':
            plt.plot(i, actual_prices[i], '^', markersize=10, color='orange')
    
    plt.title('Trading Signals')
    plt.grid(True)
    
    # Plot portfolio value vs buy and hold
    plt.subplot(3, 1, 3)
    plt.plot(portfolio_value, label=f'Strategy ({total_return:.2f}%)', color='green')
    plt.plot(buy_and_hold, label=f'Buy and Hold ({buy_and_hold_return:.2f}%)', color='blue', linestyle='--')
    plt.title('Portfolio Value vs Buy and Hold')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print(f"Strategy Return: {total_return:.2f}%")
    print(f"Buy and Hold Return: {buy_and_hold_return:.2f}%")
    print(f"Alpha: {total_return - buy_and_hold_return:.2f}%")
    print(f"Total Trades: {total_trades}")
    
    results = {
        'signals': signals,
        'portfolio_value': portfolio_value,
        'buy_and_hold': buy_and_hold,
        'total_return': total_return,
        'buy_and_hold_return': buy_and_hold_return,
        'total_trades': total_trades
    }
    
    return results

def save_trading_signals(actual_prices, predicted_prices, signals, portfolio_value, file_name='trading_signals.csv'):
    """
    Save trading signals and performance to a CSV file.
    
    Args:
        actual_prices: Array of actual stock prices
        predicted_prices: Array of predicted prices
        signals: List of trading signals
        portfolio_value: List of portfolio values
        file_name: Output file name
    """
    # Create a DataFrame with all the data
    data = {
        'Actual_Price': actual_prices,
        'Predicted_Price': predicted_prices,
        'Signal': signals,
        'Portfolio_Value': portfolio_value
    }
    
    # Create dummy dates if needed
    dates = [datetime.now().strftime('%Y-%m-%d')] * len(actual_prices)
    
    # Create DataFrame
    df = pd.DataFrame(data, index=dates)
    
    # Save to CSV
    df.to_csv(file_name)
    print(f"Trading signals saved to {file_name}")

def backtest_parameters(actual_prices, predicted_prices, threshold_values):
    """
    Backtest different threshold values for trading signals.
    
    Args:
        actual_prices: Array of actual stock prices
        predicted_prices: Array of predicted prices
        threshold_values: List of threshold values to test
        
    Returns:
        DataFrame with backtest results
    """
    results = []
    
    for threshold in threshold_values:
        # Generate trading signals with threshold
        signals = generate_trading_signals_with_threshold(actual_prices, predicted_prices, threshold)
        
        # Calculate portfolio value
        portfolio_value = calculate_portfolio_value(actual_prices, signals)
        
        # Calculate strategy performance metrics
        total_return = (portfolio_value[-1] / portfolio_value[0] - 1) * 100
        total_trades = sum(1 for signal in signals if signal in ['BUY', 'SELL', 'SHORT', 'BUY_TO_COVER'])
        
        # Calculate buy and hold return
        buy_and_hold_return = (actual_prices[-1] / actual_prices[0] - 1) * 100
        
        # Calculate drawdown
        max_drawdown = calculate_max_drawdown(portfolio_value)
        
        results.append({
            'Threshold': threshold,
            'Total_Return': total_return,
            'Buy_and_Hold_Return': buy_and_hold_return,
            'Alpha': total_return - buy_and_hold_return,
            'Max_Drawdown': max_drawdown,
            'Total_Trades': total_trades
        })
    
    return pd.DataFrame(results)

def generate_trading_signals_with_threshold(actual_prices, predicted_prices, threshold=0.0):
    """
    Generate trading signals based on Neural ODE predictions with a threshold.
    
    Args:
        actual_prices: Array of actual stock prices
        predicted_prices: Array of predicted prices from Neural ODE model
        threshold: Minimum percentage change required to trigger a signal
        
    Returns:
        List of trading signals (BUY, SELL, SHORT, BUY_TO_COVER, HOLD)
    """
    signals = []
    position = 0  # 0: no position, 1: long, -1: short
    
    # For the first day, we don't have a prediction yet
    signals.append('HOLD')
    
    for i in range(1, len(predicted_prices)):
        # Predict price change direction and percentage for the next day
        price_change_pct = (predicted_prices[i] - actual_prices[i-1]) / actual_prices[i-1] * 100
        
        if position == 0:  # No position
            if price_change_pct > threshold:  # Model predicts price will go up beyond threshold
                signals.append('BUY')
                position = 1
            elif price_change_pct < -threshold:  # Model predicts price will go down beyond threshold
                signals.append('SHORT')
                position = -1
            else:  # Change not significant enough
                signals.append('HOLD')
                
        elif position == 1:  # Long position
            if price_change_pct > -threshold:  # Model doesn't predict significant downside
                signals.append('HOLD')
            else:  # Model predicts significant downside
                signals.append('SELL')
                position = 0
                
        elif position == -1:  # Short position
            if price_change_pct < threshold:  # Model doesn't predict significant upside
                signals.append('HOLD')
            else:  # Model predicts significant upside
                signals.append('BUY_TO_COVER')
                position = 0
    
    return signals

def calculate_max_drawdown(portfolio_values):
    """
    Calculate the maximum drawdown from a series of portfolio values.
    
    Args:
        portfolio_values: List of portfolio values
        
    Returns:
        Maximum drawdown as a percentage
    """
    # Convert to numpy array if it's not already
    values = np.array(portfolio_values)
    
    # Calculate the running maximum
    running_max = np.maximum.accumulate(values)
    
    # Calculate the drawdown
    drawdown = (values - running_max) / running_max * 100
    
    # Get the maximum drawdown
    max_drawdown = np.min(drawdown)
    
    return max_drawdown

if __name__ == "__main__":
    # Example usage
    import numpy as np
    
    # Create dummy data for testing
    days = 100
    actual_prices = np.linspace(100, 150, days) + np.random.normal(0, 5, days)
    # Create predictions with some noise and bias
    predicted_prices = np.roll(actual_prices, -1)  # Perfect predictions but 1 day ahead
    predicted_prices[-1] = predicted_prices[-2] * 1.01  # Last day prediction
    
    # Add some noise to predictions
    predicted_prices = predicted_prices + np.random.normal(0, 2, days)
    
    # Visualize strategy
    results = visualize_trading_strategy(actual_prices, predicted_prices)
    
    # Save trading signals
    save_trading_signals(actual_prices, predicted_prices, results['signals'], 
                        results['portfolio_value'], 'test_trading_signals.csv')
    
    # Backtest different threshold values
    thresholds = [0.0, 0.1, 0.2, 0.5, 1.0]
    backtest_results = backtest_parameters(actual_prices, predicted_prices, thresholds)
    print(backtest_results) 