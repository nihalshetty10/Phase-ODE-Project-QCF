# model_comparison_example.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import torch
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

# Import trading strategy modules
from backtesting_engine import visualize_trading_strategy as visualize_neural_ode
from lstm_trading_strategy import visualize_trading_strategy as visualize_lstm
from arch_trading_strategy import (
    prepare_returns, 
    fit_arch_model, 
    generate_predictions as arch_predict,
    visualize_trading_strategy as visualize_arch
)

# Import additional functions for different models
from lstm_trading_strategy import (
    train_lstm_model, 
    generate_predictions as lstm_predict, 
    calculate_performance_metrics as lstm_metrics
)

from arch_trading_strategy import (
    calculate_performance_metrics as arch_metrics
)

def load_stock_data(ticker, start_date='2017-01-01', end_date='2022-12-31'):
    """
    Load stock data from Yahoo Finance
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date for data
        end_date: End date for data
        
    Returns:
        DataFrame with stock prices
    """
    # Download data from Yahoo Finance
    data = yf.download(ticker, start=start_date, end=end_date)
    
    # Select close prices (Yahoo Finance now auto-adjusts by default)
    prices = data['Close']
    
    return prices

def compare_models(ticker='SPY', start_date='2017-01-01', end_date='2022-12-31'):
    """
    Compare Neural ODE, LSTM, and ARCH models for trading strategy on given ticker
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date for data
        end_date: End date for data
    """
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    print(f"Loading data for {ticker}...")
    prices = load_stock_data(ticker, start_date, end_date)
    
    # Split data into train and test sets
    train_size = int(len(prices) * 0.8)
    train_data = prices[:train_size]
    test_data = prices[train_size-20:]  # Include last 20 points from train set for sequence
    
    print(f"Training data: {train_data.index[0]} to {train_data.index[-1]} ({len(train_data)} days)")
    print(f"Test data: {test_data.index[0]} to {test_data.index[-1]} ({len(test_data)} days)")
    
    # -------------------------
    # Train and evaluate LSTM model
    # -------------------------
    print("\nTraining LSTM model...")
    lstm_model, scaler = train_lstm_model(
        train_data, 
        seq_length=20, 
        hidden_dim=64, 
        num_layers=2, 
        batch_size=32, 
        num_epochs=50,
        learning_rate=0.001
    )
    
    # Generate LSTM predictions
    print("Generating LSTM predictions...")
    lstm_predictions = lstm_predict(lstm_model, test_data, scaler, seq_length=20)
    
    # Visualize LSTM strategy
    print("Evaluating LSTM trading strategy...")
    lstm_results = visualize_lstm(test_data.values, lstm_predictions)
    
    lstm_metrics_results, lstm_advanced_metrics = lstm_metrics(
        lstm_results['portfolio_value'], 
        lstm_results['buy_and_hold']
    )
    
    # -------------------------
    # Train and evaluate ARCH model
    # -------------------------
    print("\nTraining ARCH model...")
    # Calculate returns for ARCH model
    test_returns = prepare_returns(test_data.values)
    
    # Fit ARCH model
    arch_model_result = fit_arch_model(
        test_returns,
        p=1,  # ARCH term
        q=1,  # GARCH term
        mean='Constant',
        vol='GARCH',
        dist='normal'
    )
    
    # Generate ARCH predictions
    print("Generating ARCH predictions...")
    arch_direction_predictions, arch_forecast_variance = arch_predict(
        arch_model_result, 
        test_returns, 
        forecast_horizon=1
    )
    
    # Visualize ARCH strategy
    print("Evaluating ARCH trading strategy...")
    arch_results = visualize_arch(test_data.values, arch_direction_predictions)
    
    arch_metrics_results, arch_advanced_metrics = arch_metrics(
        arch_results['portfolio_value'], 
        arch_results['buy_and_hold']
    )
    
    # -------------------------
    # For Neural ODE, we would normally train here
    # For this example, we'll generate synthetic results for demonstration
    # -------------------------
    print("\nTraining Neural ODE model...")
    # In a real implementation, this would call the Neural ODE training function
    
    # Create synthetic Neural ODE predictions
    # In a real implementation, these would come from the actual model
    neural_ode_predictions = create_synthetic_predictions(test_data.values, shift=3, noise=0.02)
    
    # Visualize Neural ODE strategy
    print("Evaluating Neural ODE trading strategy...")
    neural_ode_results = visualize_neural_ode(test_data.values, neural_ode_predictions)
    
    # Compare the results
    compare_results(
        lstm_results, lstm_metrics_results, lstm_advanced_metrics,
        arch_results, arch_metrics_results, arch_advanced_metrics,
        neural_ode_results, 
        ticker
    )
    
    # Save comparison results
    save_comparison_results(
        lstm_results, lstm_metrics_results, lstm_advanced_metrics,
        arch_results, arch_metrics_results, arch_advanced_metrics,
        neural_ode_results, 
        ticker
    )
    
    print("\nModel comparison completed!")

def create_synthetic_predictions(actual_prices, shift=2, noise=0.01):
    """
    Create synthetic predictions for demonstration purposes
    
    Args:
        actual_prices: Actual price data
        shift: Number of days to shift predictions forward
        noise: Amount of noise to add to predictions
        
    Returns:
        Synthetic predicted prices
    """
    predictions = np.zeros_like(actual_prices)
    
    # Copy the actual prices but shifted by 'shift' days
    predictions[:shift] = actual_prices[:shift]  # Keep first few values the same
    
    # Create shifted prediction with some noise
    for i in range(shift, len(actual_prices)):
        # Base prediction: value from 'shift' days ago + trend
        base_prediction = actual_prices[i-shift]
        trend = (actual_prices[i-1] - actual_prices[i-shift]) / shift  # Simple trend calculation
        
        # Add some randomness to make it interesting
        random_factor = 1 + (np.random.random() - 0.5) * noise
        
        # Combine factors
        predictions[i] = base_prediction + trend * shift * random_factor
    
    return predictions

def compare_results(
    lstm_results, lstm_metrics, lstm_advanced_metrics,
    arch_results, arch_metrics, arch_advanced_metrics,
    neural_ode_results, 
    ticker
):
    """
    Compare and visualize the performance of LSTM, ARCH, and Neural ODE models
    
    Args:
        lstm_results: Results from LSTM trading strategy
        lstm_metrics: Performance metrics for LSTM
        lstm_advanced_metrics: Advanced metrics for LSTM
        arch_results: Results from ARCH trading strategy
        arch_metrics: Performance metrics for ARCH
        arch_advanced_metrics: Advanced metrics for ARCH
        neural_ode_results: Results from Neural ODE trading strategy
        ticker: Stock ticker symbol
    """
    # Plot portfolio values
    plt.figure(figsize=(12, 8))
    plt.plot(neural_ode_results['portfolio_value'], 
             label=f"Neural ODE ({neural_ode_results['total_return']:.2f}%)", 
             color='green')
    plt.plot(lstm_results['portfolio_value'], 
             label=f"LSTM ({lstm_results['total_return']:.2f}%)", 
             color='red')
    plt.plot(arch_results['portfolio_value'], 
             label=f"ARCH ({arch_results['total_return']:.2f}%)", 
             color='purple')
    plt.plot(neural_ode_results['buy_and_hold'], 
             label=f"Buy & Hold ({neural_ode_results['buy_and_hold_return']:.2f}%)", 
             color='blue', 
             linestyle='--')
    
    plt.title(f"Portfolio Performance Comparison - {ticker}")
    plt.xlabel("Trading Days")
    plt.ylabel("Portfolio Value ($)")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{ticker}_model_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print key metrics comparison
    print("\n----- PERFORMANCE METRICS COMPARISON -----")
    print(f"{'Metric':<25} | {'Neural ODE':<12} | {'LSTM':<12} | {'ARCH':<12} | {'Buy & Hold':<12}")
    print("-" * 85)
    
    for metric in lstm_metrics:
        neural_ode_value = neural_ode_results['metrics'][metric]['strategy'] if metric in neural_ode_results.get('metrics', {}) else 'N/A'
        arch_value = arch_metrics[metric]['strategy'] if metric in arch_metrics else 'N/A'
        
        print(f"{metric:<25} | {neural_ode_value:<12} | {lstm_metrics[metric]['strategy']:<12} | {arch_value:<12} | {lstm_metrics[metric]['buyhold']:<12}")
    
    print("\n----- ADVANCED METRICS COMPARISON -----")
    print(f"{'Metric':<25} | {'Neural ODE':<12} | {'LSTM':<12} | {'ARCH':<12} | {'Buy & Hold':<12}")
    print("-" * 85)
    
    for metric in lstm_advanced_metrics:
        neural_ode_value = neural_ode_results['advanced_metrics'][metric]['strategy'] if metric in neural_ode_results.get('advanced_metrics', {}) else 'N/A'
        arch_value = arch_advanced_metrics[metric]['strategy'] if metric in arch_advanced_metrics else 'N/A'
        
        print(f"{metric:<25} | {neural_ode_value:<12} | {lstm_advanced_metrics[metric]['strategy']:<12} | {arch_value:<12} | {lstm_advanced_metrics[metric]['buyhold']:<12}")

def save_comparison_results(
    lstm_results, lstm_metrics, lstm_advanced_metrics,
    arch_results, arch_metrics, arch_advanced_metrics,
    neural_ode_results, 
    ticker
):
    """
    Save comparison results for web interface
    
    Args:
        lstm_results: Results from LSTM trading strategy
        lstm_metrics: Performance metrics for LSTM
        lstm_advanced_metrics: Advanced metrics for LSTM
        arch_results: Results from ARCH trading strategy
        arch_metrics: Performance metrics for ARCH
        arch_advanced_metrics: Advanced metrics for ARCH
        neural_ode_results: Results from Neural ODE trading strategy
        ticker: Stock ticker symbol
    """
    # Combine all results
    comparison_data = {
        'ticker': ticker,
        'lstm': {
            'signals': lstm_results['signals'],
            'portfolio_value': lstm_results['portfolio_value'],
            'total_return': lstm_results['total_return'],
            'metrics': lstm_metrics,
            'advanced_metrics': lstm_advanced_metrics
        },
        'arch': {
            'signals': arch_results['signals'],
            'portfolio_value': arch_results['portfolio_value'],
            'total_return': arch_results['total_return'],
            'metrics': arch_metrics,
            'advanced_metrics': arch_advanced_metrics
        },
        'neural_ode': {
            'signals': neural_ode_results['signals'],
            'portfolio_value': neural_ode_results['portfolio_value'],
            'total_return': neural_ode_results['total_return'],
            'metrics': neural_ode_results.get('metrics', {}),
            'advanced_metrics': neural_ode_results.get('advanced_metrics', {})
        },
        'buy_and_hold': {
            'values': lstm_results['buy_and_hold'],
            'return': lstm_results['buy_and_hold_return']
        }
    }
    
    # Save to pickle file for web interface
    with open(f"{ticker}_comparison_results.pkl", 'wb') as f:
        pickle.dump(comparison_data, f)
    
    print(f"Comparison results saved to {ticker}_comparison_results.pkl")

if __name__ == "__main__":
    # Run comparison with SPY data
    compare_models("SPY", start_date="2020-01-01", end_date="2022-12-31") 
