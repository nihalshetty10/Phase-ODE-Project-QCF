import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import yfinance as yf
from lstm_trading_strategy import (
    train_lstm_model, 
    generate_predictions, 
    visualize_trading_strategy, 
    save_trading_signals,
    calculate_performance_metrics
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
    
    # Select adjusted close prices
    prices = data['Adj Close']
    
    return prices

def main():
    """Main function to demonstrate LSTM trading strategy"""
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    print("Loading stock data...")
    ticker = 'SPY'  # S&P 500 ETF
    prices = load_stock_data(ticker)
    
    # Split data into train and test sets
    train_size = int(len(prices) * 0.8)
    train_data = prices[:train_size]
    test_data = prices[train_size-20:]  # Include last 20 points from train set for sequence
    
    print(f"Training data size: {len(train_data)}")
    print(f"Test data size: {len(test_data)}")
    
    # Train LSTM model
    print("Training LSTM model...")
    lstm_model, scaler = train_lstm_model(
        train_data, 
        seq_length=20, 
        hidden_dim=64, 
        num_layers=2, 
        batch_size=32, 
        num_epochs=50,  # Reduced for example
        learning_rate=0.001
    )
    
    # Generate predictions
    print("Generating predictions...")
    predicted_prices = generate_predictions(lstm_model, test_data, scaler, seq_length=20)
    
    # Get actual prices from test set
    actual_prices = test_data.values
    
    # Visualize trading strategy
    print("Visualizing trading strategy...")
    results = visualize_trading_strategy(actual_prices, predicted_prices)
    
    # Save trading signals
    save_trading_signals(
        actual_prices, 
        predicted_prices, 
        results['signals'], 
        results['portfolio_value'], 
        'lstm_trading_signals.csv'
    )
    
    # Calculate detailed performance metrics
    metrics, advanced_metrics = calculate_performance_metrics(
        results['portfolio_value'], 
        results['buy_and_hold']
    )
    
    # Print detailed metrics
    print("\nPerformance Metrics:")
    for metric, values in metrics.items():
        print(f"{metric}: Strategy = {values['strategy']}, Buy & Hold = {values['buyhold']}")
    
    print("\nAdvanced Metrics:")
    for metric, values in advanced_metrics.items():
        print(f"{metric}: Strategy = {values['strategy']}, Buy & Hold = {values['buyhold']}")
    
    # Compare with results data
    combined_results = {
        'lstm': {
            'signals': results['signals'],
            'portfolio_value': results['portfolio_value'],
            'buy_and_hold': results['buy_and_hold'],
            'total_return': results['total_return'],
            'buy_and_hold_return': results['buy_and_hold_return'],
            'metrics': metrics,
            'advanced_metrics': advanced_metrics
        }
    }
    
    # Save combined results (can be loaded for comparison with neural ODE)
    import pickle
    with open('lstm_results.pkl', 'wb') as f:
        pickle.dump(combined_results, f)
    
    print("LSTM trading strategy analysis complete!")

if __name__ == "__main__":
    main() 