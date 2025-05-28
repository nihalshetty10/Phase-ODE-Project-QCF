# neural_ode_core.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import copy
from torchdiffeq import odeint

# Define the ODE function and Neural ODE model (same as in the notebook)
class ODEFunc(nn.Module):
    def __init__(self, dim):
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, dim),
        )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        return self.net(y)

class NeuralODEModel(nn.Module):
    def __init__(self, ode_func, dim):
        super(NeuralODEModel, self).__init__()
        self.ode_func = ode_func
        self.fc = nn.Linear(dim, 1)  # Output layer to map ODE solution to a single value

    def forward(self, x):
        out = odeint(self.ode_func, x, torch.tensor([0, 1.0]), method='dopri5')
        out = out[1]  # Take the solution at the end time (t=1.0)
        out = self.fc(out)
        return out

# Function to load and preprocess data
def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    # Assuming 'Close' is the column name for closing prices
    prices = data['Close'].values.reshape(-1, 1)
    
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_prices = scaler.fit_transform(prices)
    
    return data, scaled_prices, scaler

# Function to prepare data for Neural ODE (similar to the notebook phase_space_reconstruction)
def phase_space_reconstruction(data, delay, embedding_dim):
    n_samples = len(data) - delay * (embedding_dim - 1)
    reconstructed = np.zeros((n_samples, embedding_dim))
    
    for i in range(n_samples):
        for j in range(embedding_dim):
            reconstructed[i, j] = data[i + j * delay]
    
    return reconstructed

# Function to generate trading signals based on model predictions
def generate_trading_signals(actual_prices, predicted_prices):
    signals = []
    position = 0  # 0: no position, 1: long, -1: short
    
    # For the first day, we don't have a prediction yet
    signals.append('HOLD')
    
    for i in range(1, len(predicted_prices)):
        price_change = predicted_prices[i] - actual_prices[i-1]
        
        if position == 0:  # No current position
            if price_change > 0:
                signals.append('BUY')
                position = 1
            elif price_change < 0:
                signals.append('SHORT')
                position = -1
            else:
                signals.append('HOLD')
        
        elif position == 1:  # Currently long
            if price_change > 0:
                signals.append('HOLD')
            else:
                signals.append('SELL')
                position = 0
        
        elif position == -1:  # Currently short
            if price_change < 0:
                signals.append('HOLD')
            else:
                signals.append('BUY_TO_COVER')
                position = 0
    
    return signals

# Function to calculate portfolio value based on trading signals
def calculate_portfolio_value(prices, signals, initial_capital=10000):
    portfolio_value = [initial_capital]
    position = 0  # 0: no position, 1: long, -1: short
    shares = 0
    
    for i in range(1, len(signals)):
        current_value = portfolio_value[-1]
        
        if signals[i] == 'BUY':
            position = 1
            shares = current_value / prices[i]
            portfolio_value.append(current_value)  # Value doesn't change on buy day
        
        elif signals[i] == 'SELL':
            position = 0
            current_value = shares * prices[i]
            shares = 0
            portfolio_value.append(current_value)
        
        elif signals[i] == 'SHORT':
            position = -1
            shares = current_value / prices[i]
            portfolio_value.append(current_value)  # Value doesn't change on short day
        
        elif signals[i] == 'BUY_TO_COVER':
            position = 0
            current_value = current_value + (shares * (prices[i-1] - prices[i]))
            shares = 0
            portfolio_value.append(current_value)
        
        else:  # HOLD
            if position == 1:
                portfolio_value.append(shares * prices[i])
            elif position == -1:
                profit_or_loss = shares * (prices[i-1] - prices[i])
                portfolio_value.append(portfolio_value[-1] + profit_or_loss)
            else:
                portfolio_value.append(current_value)
    
    return portfolio_value

# Function to visualize trading results
def visualize_trading_results(data, signals, portfolio_value):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
    
    # Plot stock prices and signals
    ax1.plot(data.index, data['Close'], label='Stock Price')
    
    # Add markers for buy and sell signals
    buy_signals = [i for i in range(len(signals)) if signals[i] == 'BUY']
    sell_signals = [i for i in range(len(signals)) if signals[i] == 'SELL']
    short_signals = [i for i in range(len(signals)) if signals[i] == 'SHORT']
    cover_signals = [i for i in range(len(signals)) if signals[i] == 'BUY_TO_COVER']
    
    if buy_signals:
        ax1.scatter(data.index[buy_signals], data['Close'].iloc[buy_signals], 
                    marker='^', color='green', s=100, label='Buy')
    if sell_signals:
        ax1.scatter(data.index[sell_signals], data['Close'].iloc[sell_signals], 
                    marker='v', color='red', s=100, label='Sell')
    if short_signals:
        ax1.scatter(data.index[short_signals], data['Close'].iloc[short_signals], 
                    marker='v', color='purple', s=100, label='Short')
    if cover_signals:
        ax1.scatter(data.index[cover_signals], data['Close'].iloc[cover_signals], 
                    marker='^', color='orange', s=100, label='Cover')
    
    ax1.set_title('Stock Price and Trading Signals')
    ax1.set_ylabel('Price')
    ax1.grid(True)
    ax1.legend()
    
    # Plot portfolio value
    ax2.plot(data.index[:len(portfolio_value)], portfolio_value, label='Portfolio Value', color='blue')
    ax2.set_title('Portfolio Value Over Time')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Value ($)')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('trading_strategy_results.png')
    plt.show()

def main():
    # Load the saved model
    model_path = 'neural_ode_model.pth'
    data_path = 'SPX_test.csv'  # Update with your actual data path
    
    # Load and preprocess data
    data, scaled_prices, scaler = load_and_preprocess_data(data_path)
    
    # Parameters for phase space reconstruction (same as in the notebook)
    delay = 1
    embedding_dim = 3
    
    # Prepare data
    reconstructed_features = phase_space_reconstruction(scaled_prices, delay, embedding_dim)
    
    # Load the model
    input_dim = reconstructed_features.shape[1]
    ode_func = ODEFunc(input_dim)
    model = NeuralODEModel(ode_func, input_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Generate predictions
    with torch.no_grad():
        tensor_input = torch.from_numpy(reconstructed_features).float()
        predictions = model(tensor_input).numpy()
    
    # Inverse transform predictions to get actual prices
    predicted_prices = scaler.inverse_transform(predictions)
    actual_prices = data['Close'].values[delay * (embedding_dim - 1):]
    
    # Generate trading signals
    signals = generate_trading_signals(actual_prices, predicted_prices.flatten())
    
    # Calculate portfolio value
    portfolio_value = calculate_portfolio_value(actual_prices, signals)
    
    # Visualize results
    data_subset = data.iloc[delay * (embedding_dim - 1):].reset_index(drop=True)
    visualize_trading_results(data_subset, signals, portfolio_value)
    
    # Print strategy performance metrics
    initial_capital = 10000
    final_value = portfolio_value[-1]
    total_return = (final_value - initial_capital) / initial_capital * 100
    
    print(f"Initial Capital: ${initial_capital:.2f}")
    print(f"Final Portfolio Value: ${final_value:.2f}")
    print(f"Total Return: {total_return:.2f}%")
    
    # Calculate and print additional metrics
    buy_and_hold_return = (actual_prices[-1] - actual_prices[0]) / actual_prices[0] * 100
    print(f"Buy and Hold Return: {buy_and_hold_return:.2f}%")
    
    # Count the number of trades
    buy_count = signals.count('BUY')
    sell_count = signals.count('SELL')
    short_count = signals.count('SHORT')
    cover_count = signals.count('BUY_TO_COVER')
    
    total_trades = buy_count + sell_count + short_count + cover_count
    print(f"Total Trades: {total_trades}")
    print(f"  Buy: {buy_count}, Sell: {sell_count}")
    print(f"  Short: {short_count}, Cover: {cover_count}")
    
    # Save trading signals to CSV
    signal_df = pd.DataFrame({
        'Date': data_subset.index,
        'Price': actual_prices,
        'Predicted_Price': predicted_prices.flatten(),
        'Signal': signals,
    })
    signal_df.to_csv('trading_signals.csv', index=False)
    print("Trading signals saved to 'trading_signals.csv'")

if __name__ == "__main__":
    main() 
