import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import yfinance as yf
import pickle
from arch_trading_strategy import (
    prepare_returns,
    fit_arch_model,
    generate_predictions,
    generate_trading_signals,
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
    """Main function to demonstrate ARCH trading strategy"""
    # Set random seed for reproducibility
    np.random.seed(42)
    
    print("Loading stock data...")
    ticker = 'SPY'  # S&P 500 ETF
    prices = load_stock_data(ticker, start_date='2018-01-01', end_date='2022-12-31')
    
    # Split data into train and test sets
    train_size = int(len(prices) * 0.8)
    train_data = prices[:train_size]
    test_data = prices[train_size:]
    
    print(f"Training data size: {len(train_data)}")
    print(f"Test data size: {len(test_data)}")
    
    # Calculate returns for ARCH model
    print("Calculating returns...")
    train_returns = prepare_returns(train_data.values)
    test_returns = prepare_returns(test_data.values)
    
    # Fit ARCH model
    print("Fitting ARCH/GARCH model...")
    # Using GARCH(1,1) with a constant mean
    model_result = fit_arch_model(
        train_returns,
        p=1,  # ARCH term
        q=1,  # GARCH term
        mean='Constant',
        vol='GARCH',
        dist='normal'
    )
    
    print("\nModel Summary:")
    print(model_result.summary().tables[0].as_text())
    
    # Generate directional predictions
    print("\nGenerating volatility forecasts and directional predictions...")
    direction_predictions, forecast_variance = generate_predictions(
        model_result, 
        train_returns, 
        forecast_horizon=1
    )
    
    # Fit another model on all data for testing
    full_returns = prepare_returns(test_data.values)
    test_model = fit_arch_model(
        full_returns,
        p=1,
        q=1,
        mean='Constant',
        vol='GARCH',
        dist='normal'
    )
    
    # Generate predictions for test set
    test_direction_predictions, test_forecast_variance = generate_predictions(
        test_model,
        full_returns,
        forecast_horizon=1
    )
    
    # Visualize trading strategy
    print("Visualizing ARCH trading strategy...")
    results = visualize_trading_strategy(test_data.values, test_direction_predictions)
    
    # Save trading signals
    save_trading_signals(
        test_data.values,
        test_direction_predictions,
        results['signals'],
        results['portfolio_value'],
        'arch_trading_signals.csv'
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
    
    # Save combined results (can be loaded for comparison with other models)
    combined_results = {
        'arch': {
            'signals': results['signals'],
            'portfolio_value': results['portfolio_value'],
            'buy_and_hold': results['buy_and_hold'],
            'total_return': results['total_return'],
            'buy_and_hold_return': results['buy_and_hold_return'],
            'metrics': metrics,
            'advanced_metrics': advanced_metrics
        }
    }
    
    with open('arch_results.pkl', 'wb') as f:
        pickle.dump(combined_results, f)
    
    print("ARCH trading strategy analysis complete!")

if __name__ == "__main__":
    main() 