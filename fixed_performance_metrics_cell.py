#############################################################################
# Cell: Calculate and Display Advanced Performance Metrics
#############################################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

# Calculate daily returns from portfolio values
portfolio_values = np.array(results['portfolio_value'])
daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]

# Calculate buy and hold values if not already in results
if 'buy_and_hold' not in results:
    # Get the first and last price from the data used to generate signals
    # This assumes we have access to the original prices
    try:
        # If we used neural_ode_trading_strategy.py implementation
        first_price = actual_prices[0]
        prices = actual_prices
    except NameError:
        # If we're using the original implementation
        # Look for the prices in the scope
        if 'prices' in globals():
            first_price = prices[0]
        else:
            print("Warning: Cannot find price data. Using a dummy buy-and-hold curve.")
            first_price = portfolio_values[0] / 10000  # Approximate
            prices = np.linspace(first_price, first_price * portfolio_values[-1] / portfolio_values[0], len(portfolio_values))
    
    # Calculate buy and hold values
    initial_capital = portfolio_values[0]
    buy_and_hold = [initial_capital]
    for i in range(1, len(prices)):
        buy_and_hold.append(initial_capital * (prices[i] / prices[0]))
    
    # Create buy_and_hold key in results
    results['buy_and_hold'] = buy_and_hold
    
    # Calculate buy and hold return
    if 'buy_and_hold_return' not in results:
        results['buy_and_hold_return'] = (buy_and_hold[-1] / buy_and_hold[0] - 1) * 100

# Get buy and hold values
buy_hold_values = np.array(results['buy_and_hold'])
buy_hold_daily_returns = np.diff(buy_hold_values) / buy_hold_values[:-1]

# Calculate performance metrics
mean_daily_return = np.mean(daily_returns) * 100
std_daily_return = np.std(daily_returns) * 100
annualized_return = ((1 + mean_daily_return/100) ** 252 - 1) * 100
annualized_volatility = std_daily_return * np.sqrt(252)
sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility != 0 else 0

# Calculate maximum drawdown
def calculate_max_drawdown(portfolio_values):
    # Calculate the running maximum
    running_max = np.maximum.accumulate(portfolio_values)
    # Calculate drawdown in percentage terms
    drawdown = (portfolio_values - running_max) / running_max * 100
    # Get the maximum drawdown and its index
    max_drawdown = np.min(drawdown)
    max_drawdown_idx = np.argmin(drawdown)
    # Find the peak before the maximum drawdown
    peak_idx = np.argmax(portfolio_values[:max_drawdown_idx+1])
    
    return max_drawdown, peak_idx, max_drawdown_idx

max_drawdown, peak_idx, trough_idx = calculate_max_drawdown(portfolio_values)
max_dd_duration = trough_idx - peak_idx

# Calculate the same metrics for buy and hold
bh_mean_daily_return = np.mean(buy_hold_daily_returns) * 100
bh_std_daily_return = np.std(buy_hold_daily_returns) * 100
bh_annualized_return = ((1 + bh_mean_daily_return/100) ** 252 - 1) * 100
bh_annualized_volatility = bh_std_daily_return * np.sqrt(252)
bh_sharpe_ratio = bh_annualized_return / bh_annualized_volatility if bh_annualized_volatility != 0 else 0
bh_max_drawdown, bh_peak_idx, bh_trough_idx = calculate_max_drawdown(buy_hold_values)
bh_max_dd_duration = bh_trough_idx - bh_peak_idx

# Check if we have trades_df
if 'trades_df' not in locals() and 'trades_df' not in globals():
    # If we don't have trades_df, try to create one from signals
    try:
        trades = []
        signals = results['signals']
        prices_for_trades = prices[:len(signals)]  # Make sure we don't go out of bounds
        
        current_position = 'NONE'
        entry_price = 0
        entry_index = 0
        
        for i in range(len(signals)):
            signal = signals[i]
            if i < len(prices_for_trades):
                price = prices_for_trades[i]
            else:
                break
                
            if signal == 'BUY':
                current_position = 'LONG'
                entry_price = price
                entry_index = i
            elif signal == 'SHORT':
                current_position = 'SHORT'
                entry_price = price
                entry_index = i
            elif signal == 'SELL' and current_position == 'LONG':
                profit = (price - entry_price) / entry_price * 100
                trades.append({
                    'Type': 'LONG',
                    'Entry': entry_index,
                    'Exit': i,
                    'Entry_Price': entry_price,
                    'Exit_Price': price,
                    'Profit_Pct': profit,
                    'Duration': i - entry_index
                })
                current_position = 'NONE'
            elif signal == 'BUY_TO_COVER' and current_position == 'SHORT':
                profit = (entry_price - price) / entry_price * 100
                trades.append({
                    'Type': 'SHORT',
                    'Entry': entry_index,
                    'Exit': i,
                    'Entry_Price': entry_price,
                    'Exit_Price': price,
                    'Profit_Pct': profit,
                    'Duration': i - entry_index
                })
                current_position = 'NONE'

        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    except:
        # If that fails, create an empty DataFrame
        trades_df = pd.DataFrame()

# Calculate win rate and profit factor
if len(trades_df) > 0:
    win_rate = sum(trades_df['Profit_Pct'] > 0) / len(trades_df) * 100
    total_profits = trades_df[trades_df['Profit_Pct'] > 0]['Profit_Pct'].sum()
    total_losses = abs(trades_df[trades_df['Profit_Pct'] < 0]['Profit_Pct'].sum())
    profit_factor = total_profits / total_losses if total_losses != 0 else float('inf')
else:
    win_rate = 0
    profit_factor = 0

# Print performance metrics
print("\n" + "="*50)
print("PERFORMANCE METRICS COMPARISON")
print("="*50)

metrics_table = pd.DataFrame({
    'Metric': [
        'Total Return (%)', 
        'Annualized Return (%)', 
        'Mean Daily Return (%)', 
        'Daily Volatility (%)', 
        'Annualized Volatility (%)', 
        'Sharpe Ratio',
        'Maximum Drawdown (%)', 
        'Max Drawdown Duration (days)',
        'Win Rate (%)',
        'Profit Factor'
    ],
    'Trading Strategy': [
        f"{results['total_return']:.2f}",
        f"{annualized_return:.2f}",
        f"{mean_daily_return:.4f}",
        f"{std_daily_return:.4f}",
        f"{annualized_volatility:.2f}",
        f"{sharpe_ratio:.2f}",
        f"{max_drawdown:.2f}",
        f"{max_dd_duration}",
        f"{win_rate:.2f}" if len(trades_df) > 0 else "N/A",
        f"{profit_factor:.2f}" if len(trades_df) > 0 else "N/A"
    ],
    'Buy and Hold': [
        f"{results['buy_and_hold_return']:.2f}",
        f"{bh_annualized_return:.2f}",
        f"{bh_mean_daily_return:.4f}",
        f"{bh_std_daily_return:.4f}",
        f"{bh_annualized_volatility:.2f}",
        f"{bh_sharpe_ratio:.2f}",
        f"{bh_max_drawdown:.2f}",
        f"{bh_max_dd_duration}",
        "N/A",
        "N/A"
    ]
})

# Display metrics table
print(metrics_table.to_string(index=False))

# Visualize the return distribution
plt.figure(figsize=(15, 10))

# Plot 1: Return distribution
plt.subplot(2, 2, 1)
plt.hist(daily_returns * 100, bins=50, alpha=0.6, color='green', label='Strategy')
plt.hist(buy_hold_daily_returns * 100, bins=50, alpha=0.6, color='blue', label='Buy & Hold')
plt.axvline(0, color='black', linestyle='--')
plt.title('Daily Return Distribution')
plt.xlabel('Daily Return (%)')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Drawdown over time
plt.subplot(2, 2, 2)
running_max_strategy = np.maximum.accumulate(portfolio_values)
drawdown_strategy = (portfolio_values - running_max_strategy) / running_max_strategy * 100

running_max_bh = np.maximum.accumulate(buy_hold_values)
drawdown_bh = (buy_hold_values - running_max_bh) / running_max_bh * 100

plt.plot(drawdown_strategy, color='green', label='Strategy')
plt.plot(drawdown_bh, color='blue', label='Buy & Hold')
plt.title('Drawdown Over Time')
plt.xlabel('Days')
plt.ylabel('Drawdown (%)')
plt.gca().yaxis.set_major_formatter(PercentFormatter())
plt.grid(True, alpha=0.3)
plt.legend()

# Plot 3: Rolling Sharpe Ratio (30-day window)
plt.subplot(2, 2, 3)
rolling_returns = pd.Series(daily_returns)
rolling_std = rolling_returns.rolling(window=30).std() * np.sqrt(252)
rolling_mean = rolling_returns.rolling(window=30).mean() * 252
rolling_sharpe = rolling_mean / rolling_std

rolling_returns_bh = pd.Series(buy_hold_daily_returns)
rolling_std_bh = rolling_returns_bh.rolling(window=30).std() * np.sqrt(252)
rolling_mean_bh = rolling_returns_bh.rolling(window=30).mean() * 252
rolling_sharpe_bh = rolling_mean_bh / rolling_std_bh

plt.plot(rolling_sharpe, color='green', label='Strategy')
plt.plot(rolling_sharpe_bh, color='blue', label='Buy & Hold')
plt.title('30-Day Rolling Sharpe Ratio')
plt.xlabel('Days')
plt.ylabel('Sharpe Ratio')
plt.grid(True, alpha=0.3)
plt.legend()

# Plot 4: Equity curve with drawdown periods highlighted
plt.subplot(2, 2, 4)
plt.plot(portfolio_values, color='green', label='Strategy')
plt.plot(buy_hold_values, color='blue', label='Buy & Hold')

# Highlight major drawdown periods for strategy
is_drawdown = drawdown_strategy < -5  # Drawdowns greater than 5%
in_drawdown = False
start_idx = 0

for i in range(len(is_drawdown)):
    if is_drawdown[i] and not in_drawdown:
        # Start of drawdown period
        in_drawdown = True
        start_idx = i
    elif not is_drawdown[i] and in_drawdown:
        # End of drawdown period
        plt.axvspan(start_idx, i, alpha=0.2, color='red')
        in_drawdown = False

# If still in drawdown at the end
if in_drawdown:
    plt.axvspan(start_idx, len(is_drawdown)-1, alpha=0.2, color='red')

plt.title('Equity Curve with Major Drawdown Periods Highlighted')
plt.xlabel('Days')
plt.ylabel('Portfolio Value')
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()

# Calculate additional metrics
# Calmar ratio (annualized return / max drawdown)
calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else float('inf')
bh_calmar_ratio = bh_annualized_return / abs(bh_max_drawdown) if bh_max_drawdown != 0 else float('inf')

# Sortino ratio (using downside deviation)
downside_returns = np.array([min(0, r) for r in daily_returns])
downside_deviation = np.std(downside_returns) * np.sqrt(252)
sortino_ratio = annualized_return / downside_deviation if downside_deviation != 0 else float('inf')

bh_downside_returns = np.array([min(0, r) for r in buy_hold_daily_returns])
bh_downside_deviation = np.std(bh_downside_returns) * np.sqrt(252)
bh_sortino_ratio = bh_annualized_return / bh_downside_deviation if bh_downside_deviation != 0 else float('inf')

# Calculate the percentage of positive days
positive_days_pct = np.sum(daily_returns > 0) / len(daily_returns) * 100
bh_positive_days_pct = np.sum(buy_hold_daily_returns > 0) / len(buy_hold_daily_returns) * 100

# Calculate the max consecutive wins and losses if we have trade data
if len(trades_df) > 0:
    trades_df['IsWin'] = trades_df['Profit_Pct'] > 0
    
    # Calculate consecutive wins and losses
    trades_df['ConsecutiveCount'] = 1
    for i in range(1, len(trades_df)):
        if trades_df['IsWin'].iloc[i] == trades_df['IsWin'].iloc[i-1]:
            trades_df.at[trades_df.index[i], 'ConsecutiveCount'] = trades_df['ConsecutiveCount'].iloc[i-1] + 1
    
    max_consecutive_wins = trades_df[trades_df['IsWin']]['ConsecutiveCount'].max() if not trades_df[trades_df['IsWin']].empty else 0
    max_consecutive_losses = trades_df[~trades_df['IsWin']]['ConsecutiveCount'].max() if not trades_df[~trades_df['IsWin']].empty else 0
else:
    max_consecutive_wins = 0
    max_consecutive_losses = 0

# Print advanced metrics
print("\n" + "="*50)
print("ADVANCED PERFORMANCE METRICS")
print("="*50)

advanced_metrics = pd.DataFrame({
    'Metric': [
        'Calmar Ratio', 
        'Sortino Ratio', 
        'Positive Days (%)', 
        'Max Consecutive Wins',
        'Max Consecutive Losses'
    ],
    'Trading Strategy': [
        f"{calmar_ratio:.2f}",
        f"{sortino_ratio:.2f}",
        f"{positive_days_pct:.2f}",
        f"{max_consecutive_wins}" if len(trades_df) > 0 else "N/A",
        f"{max_consecutive_losses}" if len(trades_df) > 0 else "N/A"
    ],
    'Buy and Hold': [
        f"{bh_calmar_ratio:.2f}",
        f"{bh_sortino_ratio:.2f}",
        f"{bh_positive_days_pct:.2f}",
        "N/A",
        "N/A"
    ]
})

# Display advanced metrics table
print(advanced_metrics.to_string(index=False)) 