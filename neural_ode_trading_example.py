# Add these cells to your existing Neural ODE notebook after you've made predictions

#############################################################################
# Cell 1: Import the trading strategy module
#############################################################################
# First, save the neural_ode_trading_strategy.py file in the same directory as your notebook
import neural_ode_trading_strategy as ts
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#############################################################################
# Cell 2: Use the Neural ODE predictions to generate trading signals
#############################################################################
# Assuming you already have:
# - test_target (actual prices)
# - forecasted_test_scaled (predicted prices from your Neural ODE model)
# - scaler_target (the MinMaxScaler used to scale target values)

# Get the unscaled (original) prices
actual_prices = scaler_target.inverse_transform(test_target_scaled.reshape(-1, 1)).flatten()
predicted_prices = scaler_target.inverse_transform(forecasted_test_scaled).flatten()

# Get trading signals and visualize the strategy
results = ts.visualize_trading_strategy(actual_prices, predicted_prices)

# Display performance metrics
print("\nSummary Statistics:")
print(f"Strategy Return: {results['total_return']:.2f}%")
print(f"Buy and Hold Return: {results['buy_and_hold_return']:.2f}%")
print(f"Alpha: {results['total_return'] - results['buy_and_hold_return']:.2f}%")
print(f"Total Trades: {results['total_trades']}")

# Save trading signals to CSV for further analysis
ts.save_trading_signals(
    actual_prices, 
    predicted_prices,
    results['signals'],
    results['portfolio_value'],
    file_name=f'{ticker}_trading_signals.csv'
)

#############################################################################
# Cell 3: Analyze specific trading periods (optional)
#############################################################################
# Create a DataFrame with all the data
trading_data = pd.DataFrame({
    'Price': actual_prices,
    'Predicted_Price': predicted_prices,
    'Signal': results['signals'],
    'Portfolio_Value': results['portfolio_value'],
})

# Analyze consecutive winning/losing trades
trading_data['Daily_Return'] = trading_data['Portfolio_Value'].pct_change()
trading_data['Trade'] = trading_data['Signal'].isin(['BUY', 'SELL', 'SHORT', 'BUY_TO_COVER']).astype(int)
trading_data['Trade_Number'] = trading_data['Trade'].cumsum()

# Identify winning trades
trades = []
current_position = 'NONE'
entry_price = 0
entry_index = 0

for i in range(len(trading_data)):
    signal = trading_data['Signal'].iloc[i]
    price = trading_data['Price'].iloc[i]
    
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

trades_df = pd.DataFrame(trades)
if len(trades_df) > 0:
    # Print trade statistics
    print("\nTrade Statistics:")
    print(f"Total Completed Trades: {len(trades_df)}")
    print(f"Winning Trades: {sum(trades_df['Profit_Pct'] > 0)}")
    print(f"Losing Trades: {sum(trades_df['Profit_Pct'] <= 0)}")
    print(f"Win Rate: {sum(trades_df['Profit_Pct'] > 0) / len(trades_df) * 100:.2f}%")
    print(f"Average Trade Profit: {trades_df['Profit_Pct'].mean():.2f}%")
    print(f"Average Winning Trade: {trades_df[trades_df['Profit_Pct'] > 0]['Profit_Pct'].mean():.2f}%")
    print(f"Average Losing Trade: {trades_df[trades_df['Profit_Pct'] <= 0]['Profit_Pct'].mean():.2f}%")
    print(f"Average Trade Duration: {trades_df['Duration'].mean():.1f} days")
    
    # Plot profit distribution
    plt.figure(figsize=(10, 6))
    plt.hist(trades_df['Profit_Pct'], bins=20)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.title('Distribution of Trade Profits')
    plt.xlabel('Profit %')
    plt.ylabel('Number of Trades')
    plt.grid(True)
    plt.show()

#############################################################################
# Cell 4: Monthly performance analysis (optional)
#############################################################################
# First create some dummy dates for this example
# In your real notebook, you should have actual dates from your data
import pandas as pd
from datetime import datetime, timedelta

# Create dummy dates (replace this with your actual dates if available)
start_date = datetime(2020, 1, 1)
dates = [start_date + timedelta(days=i) for i in range(len(actual_prices))]

# Create a DataFrame with all the data including dates
trading_data = pd.DataFrame({
    'Date': dates,
    'Price': actual_prices,
    'Predicted_Price': predicted_prices,
    'Signal': results['signals'],
    'Portfolio_Value': results['portfolio_value'],
})

# Set the date as index
trading_data['Date'] = pd.to_datetime(trading_data['Date'])
trading_data.set_index('Date', inplace=True)

# Calculate monthly returns
monthly_returns = trading_data['Portfolio_Value'].resample('M').last().pct_change() * 100
# Calculate buy-and-hold monthly returns for comparison
trading_data['Buy_Hold_Value'] = [10000 * (price / actual_prices[0]) for price in actual_prices]
buy_hold_monthly_returns = trading_data['Buy_Hold_Value'].resample('M').last().pct_change() * 100

# Combine the returns
monthly_comparison = pd.DataFrame({
    'Strategy': monthly_returns,
    'Buy_and_Hold': buy_hold_monthly_returns
})

# Drop the first row which is NaN
monthly_comparison = monthly_comparison.dropna()

# Print monthly performance table
print("\nMonthly Performance:")
print(monthly_comparison)

# Plot monthly returns comparison
plt.figure(figsize=(12, 6))
monthly_comparison.plot(kind='bar', figsize=(12, 6))
plt.title('Monthly Returns: Strategy vs Buy-and-Hold')
plt.ylabel('Return (%)')
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# Calculate and plot drawdowns
def calculate_drawdowns(equity_curve):
    """Calculate the drawdowns for an equity curve."""
    # Calculate running maximum
    running_max = equity_curve.cummax()
    # Calculate drawdown in percentage terms
    drawdown = (equity_curve / running_max - 1) * 100
    return drawdown

# Calculate drawdowns for both strategy and buy-and-hold
trading_data['Strategy_Drawdown'] = calculate_drawdowns(trading_data['Portfolio_Value'])
trading_data['Buy_Hold_Drawdown'] = calculate_drawdowns(trading_data['Buy_Hold_Value'])

# Plot drawdowns
plt.figure(figsize=(12, 6))
trading_data[['Strategy_Drawdown', 'Buy_Hold_Drawdown']].plot(figsize=(12, 6))
plt.title('Drawdowns: Strategy vs Buy-and-Hold')
plt.ylabel('Drawdown (%)')
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# Print maximum drawdown
print("\nDrawdown Analysis:")
print(f"Strategy Max Drawdown: {trading_data['Strategy_Drawdown'].min():.2f}%")
print(f"Buy and Hold Max Drawdown: {trading_data['Buy_Hold_Drawdown'].min():.2f}%") 