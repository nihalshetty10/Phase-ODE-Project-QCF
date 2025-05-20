# lstm_trading_strategy.py
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        """
        LSTM model for time series prediction
        
        Args:
            input_dim: Input dimension (features)
            hidden_dim: Hidden dimension size
            num_layers: Number of LSTM layers
            output_dim: Output dimension
        """
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, input_dim)
            
        Returns:
            Predictions of shape (batch_size, output_dim)
        """
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # LSTM forward pass
        out, _ = self.lstm(x, (h0, c0))
        
        # We only need the output from the last time step
        out = self.fc(out[:, -1, :])
        
        return out

class StockDataset(Dataset):
    def __init__(self, data, seq_length, scaler=None, train=True):
        """
        Stock price dataset for LSTM
        
        Args:
            data: DataFrame with stock prices
            seq_length: Sequence length for LSTM
            scaler: Optional MinMaxScaler for normalization
            train: Whether this is training data
        """
        self.seq_length = seq_length
        self.train = train
        
        # Preprocess data
        if scaler is None:
            self.scaler = MinMaxScaler(feature_range=(0, 1))
            self.scaled_data = self.scaler.fit_transform(data.values.reshape(-1, 1))
        else:
            self.scaler = scaler
            self.scaled_data = self.scaler.transform(data.values.reshape(-1, 1))
        
        # Create sequences
        self.create_sequences()
        
    def create_sequences(self):
        """Create sequences for LSTM model"""
        self.X, self.y = [], []
        
        for i in range(len(self.scaled_data) - self.seq_length):
            X_seq = self.scaled_data[i:i+self.seq_length]
            y_seq = self.scaled_data[i+self.seq_length]
            
            self.X.append(X_seq)
            self.y.append(y_seq)
            
        self.X = torch.FloatTensor(np.array(self.X))
        self.y = torch.FloatTensor(np.array(self.y))
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def train_lstm_model(data, seq_length=20, hidden_dim=64, num_layers=2, batch_size=32, num_epochs=100, learning_rate=0.001):
    """
    Train LSTM model on stock price data
    
    Args:
        data: DataFrame with stock prices
        seq_length: Sequence length for LSTM model
        hidden_dim: Hidden dimension size
        num_layers: Number of LSTM layers
        batch_size: Batch size for training
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        
    Returns:
        Trained LSTM model and data scaler
    """
    # Prepare dataset
    dataset = StockDataset(data, seq_length)
    
    # Split into train and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model, loss function, and optimizer
    input_dim = 1  # Single feature (price)
    output_dim = 1  # Single output (next price)
    
    model = LSTMModel(input_dim, hidden_dim, num_layers, output_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train the model
    train_losses, val_losses = [], []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        
        for X_batch, y_batch in train_loader:
            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
    
    return model, dataset.scaler

def generate_predictions(model, data, scaler, seq_length=20):
    """
    Generate predictions using trained LSTM model
    
    Args:
        model: Trained LSTM model
        data: DataFrame with stock prices
        scaler: MinMaxScaler used for normalization
        seq_length: Sequence length used for LSTM model
        
    Returns:
        Array of predicted prices
    """
    # Prepare data
    scaled_data = scaler.transform(data.values.reshape(-1, 1))
    
    # Create sequences
    X = []
    for i in range(len(scaled_data) - seq_length):
        X.append(scaled_data[i:i+seq_length])
    
    X = torch.FloatTensor(np.array(X))
    
    # Generate predictions
    model.eval()
    with torch.no_grad():
        predictions = model(X)
    
    # Inverse transform
    predictions = scaler.inverse_transform(predictions.numpy())
    
    # Shift predictions to align with actual data
    full_predictions = np.zeros(len(data))
    full_predictions[:seq_length] = data.values[:seq_length].flatten()
    full_predictions[seq_length:] = predictions.flatten()
    
    return full_predictions

def generate_trading_signals(actual_prices, predicted_prices):
    """
    Generate trading signals based on LSTM predictions.
    
    Args:
        actual_prices: Array of actual stock prices
        predicted_prices: Array of predicted prices from LSTM model
        
    Returns:
        List of trading signals (BUY, SELL, SHORT, BUY_TO_COVER, HOLD)
    """
    signals = []
    position = 0  # 0: no position, 1: long, -1: short
    
    # For the first day, we don't have a meaningful prediction yet
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
    Visualize LSTM trading strategy performance.
    
    Args:
        actual_prices: Array of actual stock prices
        predicted_prices: Array of predicted prices from LSTM model
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
    plt.plot(predicted_prices, label='Predicted Prices (LSTM)', color='green', linestyle='--')
    plt.title('Actual vs Predicted Prices (LSTM)')
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
    
    plt.title('LSTM Trading Signals')
    plt.grid(True)
    
    # Plot portfolio value vs buy and hold
    plt.subplot(3, 1, 3)
    plt.plot(portfolio_value, label=f'LSTM Strategy ({total_return:.2f}%)', color='green')
    plt.plot(buy_and_hold, label=f'Buy and Hold ({buy_and_hold_return:.2f}%)', color='blue', linestyle='--')
    plt.title('Portfolio Value vs Buy and Hold')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print(f"LSTM Strategy Return: {total_return:.2f}%")
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

def save_trading_signals(actual_prices, predicted_prices, signals, portfolio_value, file_name='lstm_trading_signals.csv'):
    """
    Save LSTM trading signals and performance to a CSV file.
    
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
    print(f"LSTM trading signals saved to {file_name}")

def calculate_performance_metrics(portfolio_values, buy_and_hold_values):
    """
    Calculate comprehensive performance metrics for the LSTM trading strategy.
    
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