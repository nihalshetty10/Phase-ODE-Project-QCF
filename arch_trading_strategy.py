import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from arch import arch_model
from sklearn.preprocessing import StandardScaler

def fit_arch_model(returns, p=1, q=1, mean='Zero', vol='GARCH', dist='normal'):
    """
    Fit ARCH/GARCH model to return series
    
    Args:
        returns: Series of asset returns
        p: Order of the symmetric innovation
        q: Order of lagged volatility 
        mean: Mean model ('Zero', 'Constant', 'AR', etc.)
        vol: Volatility model ('ARCH', 'GARCH', 'EGARCH', etc.)
        dist: Error distribution ('normal', 'studentst', 'skewt', etc.)
        
    Returns:
        Fitted ARCH model
    """
    # Create and fit the model
    model = arch_model(returns, p=p, q=q, mean=mean, vol=vol, dist=dist)
    result = model.fit(disp='off')
    
    return result

def prepare_returns(prices):
    """
    Calculate returns from price series
    
    Args:
        prices: Array of asset prices
        
    Returns:
        Array of returns
    """
    # Calculate percentage returns
    returns = np.diff(prices) / prices[:-1]
    
    return returns

def generate_predictions(model_result, returns, forecast_horizon=1):
    """
    Generate volatility forecasts from fitted ARCH model
    
    Args:
        model_result: Fitted ARCH model result
        returns: Series of asset returns
        forecast_horizon: Number of steps to forecast
        
    Returns:
        Forecasted volatility and directional predictions
    """
    # Forecast volatility
    forecasts = model_result.forecast(horizon=forecast_horizon)
    
    # Get the variance forecasts
    forecast_variance = forecasts.variance.values[-1]
    
    # Estimate price direction based on volatility trend
    # If volatility is decreasing, predict price increase
    # If volatility is increasing, predict price decrease
    
    # Get historical conditional volatility
    historical_vol = np.sqrt(model_result.conditional_volatility)
    
    # Initialize directional predictions
    direction_predictions = np.zeros(len(returns) + 1)
    
    # For the first day, there is no prediction
    direction_predictions[0] = 0
    
    # For subsequent days, predict based on volatility trend
    for i in range(1, len(returns)):
        vol_change = historical_vol[i] - historical_vol[i-1]
        if vol_change < 0:
            # Volatility decreasing, predict price increase
            direction_predictions[i] = 1
        else:
            # Volatility increasing, predict price decrease
            direction_predictions[i] = -1
    
    # Add forecasted direction for the next day
    if len(historical_vol) > 1 and forecast_variance is not None:
        last_vol = historical_vol[-1]
        forecasted_vol = np.sqrt(forecast_variance[0])
        if forecasted_vol < last_vol:
            # Forecasted volatility is lower, predict price increase
            direction_predictions[-1] = 1
        else:
            # Forecasted volatility is higher, predict price decrease
            direction_predictions[-1] = -1
    
    return direction_predictions, forecast_variance

def generate_trading_signals(actual_prices, direction_predictions):
    """
    Generate trading signals based on directional predictions
    
    Args:
        actual_prices: Array of actual prices
        direction_predictions: Array of directional predictions (1: up, -1: down, 0: no change)
        
    Returns:
        List of trading signals
    """
    signals = []
    position = 0  # 0: no position, 1: long, -1: short
    
    # For the first day, we don't have a meaningful prediction
    signals.append('HOLD')
    
    for i in range(1, len(direction_predictions)):
        # Get the current direction prediction
        direction = direction_predictions[i]
        
        if position == 0:  # No position
            if direction > 0:  # Model predicts price will go up
                signals.append('BUY')
                position = 1
            elif direction < 0:  # Model predicts price will go down
                signals.append('SHORT')
                position = -1
            else:  # No change predicted
                signals.append('HOLD')
                
        elif position == 1:  # Long position
            if direction > 0:  # Model predicts price will go up
                signals.append('HOLD')
            else:  # Model predicts price will go down
                signals.append('SELL')
                position = 0
                
        elif position == -1:  # Short position
            if direction < 0:  # Model predicts price will go down
                signals.append('HOLD')
            else:  # Model predicts price will go up
                signals.append('BUY_TO_COVER')
                position = 0
    
    return signals

def calculate_portfolio_value(actual_prices, signals, initial_capital=10000.0):
    """
    Calculate portfolio value based on trading signals
    
    Args:
        actual_prices: Array of actual prices
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

def visualize_trading_strategy(actual_prices, direction_predictions, initial_capital=10000.0):
    """
    Visualize ARCH trading strategy performance
    
    Args:
        actual_prices: Array of actual prices
        direction_predictions: Array of directional predictions
        initial_capital: Initial capital amount
        
    Returns:
        Dictionary with trading results
    """
    # Generate trading signals
    signals = generate_trading_signals(actual_prices, direction_predictions)
    
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
    
    # Plot actual prices and indicators of predictions
    plt.subplot(3, 1, 1)
    plt.plot(actual_prices, label='Actual Prices', color='blue')
    
    # Create prediction visualization
    predicted_up = np.where(direction_predictions > 0, actual_prices, np.nan)
    predicted_down = np.where(direction_predictions < 0, actual_prices, np.nan)
    
    plt.scatter(range(len(predicted_up)), predicted_up, color='green', alpha=0.5, 
                label='Predict Up', marker='^')
    plt.scatter(range(len(predicted_down)), predicted_down, color='red', alpha=0.5, 
                label='Predict Down', marker='v')
    
    plt.title('Actual Prices and ARCH Predictions')
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
    
    plt.title('ARCH Trading Signals')
    plt.grid(True)
    
    # Plot portfolio value vs buy and hold
    plt.subplot(3, 1, 3)
    plt.plot(portfolio_value, label=f'ARCH Strategy ({total_return:.2f}%)', color='green')
    plt.plot(buy_and_hold, label=f'Buy and Hold ({buy_and_hold_return:.2f}%)', color='blue', linestyle='--')
    plt.title('Portfolio Value vs Buy and Hold')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print(f"ARCH Strategy Return: {total_return:.2f}%")
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

def save_trading_signals(actual_prices, direction_predictions, signals, portfolio_value, file_name='arch_trading_signals.csv'):
    """
    Save ARCH trading signals and performance to a CSV file
    
    Args:
        actual_prices: Array of actual prices
        direction_predictions: Array of direction predictions
        signals: List of trading signals
        portfolio_value: List of portfolio values
        file_name: Output file name
    """
    # Create a DataFrame with all the data
    data = {
        'Actual_Price': actual_prices,
        'Direction_Prediction': direction_predictions,
        'Signal': signals,
        'Portfolio_Value': portfolio_value
    }
    
    # Create dummy dates if needed
    dates = [datetime.now().strftime('%Y-%m-%d')] * len(actual_prices)
    
    # Create DataFrame
    df = pd.DataFrame(data, index=dates)
    
    # Save to CSV
    df.to_csv(file_name)
    print(f"ARCH trading signals saved to {file_name}")

def calculate_performance_metrics(portfolio_values, buy_and_hold_values):
    """
    Calculate comprehensive performance metrics for the ARCH trading strategy
    
    Args:
        portfolio_values: Array of portfolio values
        buy_and_hold_values: Array of buy and hold values
        
    Returns:
        Dictionary with performance metrics
    """
    # Calculate daily returns
    strategy_returns = np.diff(portfolio_values) / portfolio_values[:-1]
    bh_returns = np.diff(buy_and_hold_values) / buy_and_hold_values[:-1]
    
    # Calculate total return
    total_return = (portfolio_values[-1] / portfolio_values[0] - 1) * 100
    bh_return = (buy_and_hold_values[-1] / buy_and_hold_values[0] - 1) * 100
    
    # Annualized return (assuming 252 trading days)
    n_days = len(portfolio_values)
    annualized_return = ((1 + total_return/100) ** (252/n_days) - 1) * 100
    bh_annualized_return = ((1 + bh_return/100) ** (252/n_days) - 1) * 100
    
    # Mean daily return
    mean_daily_return = np.mean(strategy_returns) * 100
    bh_mean_daily_return = np.mean(bh_returns) * 100
    
    # Volatility
    daily_volatility = np.std(strategy_returns) * 100
    bh_daily_volatility = np.std(bh_returns) * 100
    
    # Annualized volatility
    annualized_volatility = daily_volatility * np.sqrt(252)
    bh_annualized_volatility = bh_daily_volatility * np.sqrt(252)
    
    # Sharpe ratio (assuming risk-free rate = 0 for simplicity)
    sharpe_ratio = (mean_daily_return / daily_volatility) * np.sqrt(252)
    bh_sharpe_ratio = (bh_mean_daily_return / bh_daily_volatility) * np.sqrt(252)
    
    # Maximum drawdown
    cumulative_returns = np.array(portfolio_values) / portfolio_values[0]
    bh_cumulative_returns = np.array(buy_and_hold_values) / buy_and_hold_values[0]
    
    running_max = np.maximum.accumulate(cumulative_returns)
    bh_running_max = np.maximum.accumulate(bh_cumulative_returns)
    
    drawdown = (cumulative_returns - running_max) / running_max * 100
    bh_drawdown = (bh_cumulative_returns - bh_running_max) / bh_running_max * 100
    
    max_drawdown = np.min(drawdown)
    bh_max_drawdown = np.min(bh_drawdown)
    
    # Win rate
    positive_returns = np.sum(strategy_returns > 0)
    win_rate = positive_returns / len(strategy_returns) * 100
    
    # Profit factor
    gain = np.sum(strategy_returns[strategy_returns > 0])
    loss = np.abs(np.sum(strategy_returns[strategy_returns < 0]))
    profit_factor = gain / loss if loss != 0 else float('inf')
    
    # Create metrics dictionary
    metrics = {
        'Total Return (%)': {
            'strategy': f'{total_return:.2f}',
            'buyhold': f'{bh_return:.2f}'
        },
        'Annualized Return (%)': {
            'strategy': f'{annualized_return:.2f}',
            'buyhold': f'{bh_annualized_return:.2f}'
        },
        'Mean Daily Return (%)': {
            'strategy': f'{mean_daily_return:.4f}',
            'buyhold': f'{bh_mean_daily_return:.4f}'
        },
        'Daily Volatility (%)': {
            'strategy': f'{daily_volatility:.4f}',
            'buyhold': f'{bh_daily_volatility:.4f}'
        },
        'Annualized Volatility (%)': {
            'strategy': f'{annualized_volatility:.2f}',
            'buyhold': f'{bh_annualized_volatility:.2f}'
        },
        'Sharpe Ratio': {
            'strategy': f'{sharpe_ratio:.2f}',
            'buyhold': f'{bh_sharpe_ratio:.2f}'
        },
        'Maximum Drawdown (%)': {
            'strategy': f'{max_drawdown:.2f}',
            'buyhold': f'{bh_max_drawdown:.2f}'
        },
        'Win Rate (%)': {
            'strategy': f'{win_rate:.2f}',
            'buyhold': 'N/A'
        },
        'Profit Factor': {
            'strategy': f'{profit_factor:.2f}',
            'buyhold': 'N/A'
        }
    }
    
    # Advanced metrics
    # Sortino ratio (downside risk only)
    negative_returns = strategy_returns[strategy_returns < 0]
    bh_negative_returns = bh_returns[bh_returns < 0]
    
    downside_deviation = np.std(negative_returns) * 100 if len(negative_returns) > 0 else 1e-6
    bh_downside_deviation = np.std(bh_negative_returns) * 100 if len(bh_negative_returns) > 0 else 1e-6
    
    sortino_ratio = (mean_daily_return / downside_deviation) * np.sqrt(252)
    bh_sortino_ratio = (bh_mean_daily_return / bh_downside_deviation) * np.sqrt(252)
    
    # Calmar ratio (return / max drawdown)
    calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else float('inf')
    bh_calmar_ratio = bh_annualized_return / abs(bh_max_drawdown) if bh_max_drawdown != 0 else float('inf')
    
    # Consecutive wins/losses
    win_streak = 0
    max_win_streak = 0
    loss_streak = 0
    max_loss_streak = 0
    
    for ret in strategy_returns:
        if ret > 0:
            win_streak += 1
            loss_streak = 0
            max_win_streak = max(max_win_streak, win_streak)
        elif ret < 0:
            loss_streak += 1
            win_streak = 0
            max_loss_streak = max(max_loss_streak, loss_streak)
        else:
            win_streak = 0
            loss_streak = 0
    
    # Positive days percentage
    positive_days = np.sum(strategy_returns > 0) / len(strategy_returns) * 100
    bh_positive_days = np.sum(bh_returns > 0) / len(bh_returns) * 100
    
    # Add advanced metrics
    advanced_metrics = {
        'Calmar Ratio': {
            'strategy': f'{calmar_ratio:.2f}',
            'buyhold': f'{bh_calmar_ratio:.2f}'
        },
        'Sortino Ratio': {
            'strategy': f'{sortino_ratio:.2f}',
            'buyhold': f'{bh_sortino_ratio:.2f}'
        },
        'Positive Days (%)': {
            'strategy': f'{positive_days:.2f}',
            'buyhold': f'{bh_positive_days:.2f}'
        },
        'Max Consecutive Wins': {
            'strategy': f'{max_win_streak}',
            'buyhold': 'N/A'
        },
        'Max Consecutive Losses': {
            'strategy': f'{max_loss_streak}',
            'buyhold': 'N/A'
        }
    }
    
    return metrics, advanced_metrics 