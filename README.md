# Neural ODE Trading Strategy

A web-based visualization tool for demonstrating trading strategies based on Neural Ordinary Differential Equations for price prediction.

## Overview

This project showcases a trading strategy that uses Neural ODEs to generate trading signals (BUY, SELL, SHORT, and COVER) based on price predictions. The website allows users to input a stock ticker symbol and view the performance of the trading strategy compared to a simple buy-and-hold approach.

## Live Demo

You can access the live demo at: [https://yourusername.github.io/neural-ode-trading](https://yourusername.github.io/neural-ode-trading)

## Features

- Interactive web interface for testing the Neural ODE trading strategy
- Visualization of price predictions and trading signals
- Performance comparison with buy-and-hold strategy
- Comprehensive performance metrics including:
  - Maximum drawdown
  - Sharpe ratio
  - Standard deviation
  - Mean return
  - Win rate
  - Profit factor
  - Calmar ratio
  - Sortino ratio
- Visual analysis of drawdowns, return distribution, and rolling Sharpe ratio

## Technologies Used

- **Frontend**: HTML, CSS, JavaScript
- **Visualization**: Chart.js
- **Styling**: Bootstrap 5
- **Hosting**: GitHub Pages

## How It Works

The Neural ODE trading strategy follows these rules:

1. If we have no holdings and the model predicts the stock will go up: BUY
2. If we have a holding and the model predicts up: HOLD
3. If we have a holding and the model predicts down: SELL
4. If the model predicts down and we have no holdings: SHORT
5. If we are short and the model predicts up: BUY_TO_COVER (close short position)

## Repository Structure

```
.
├── index.html          # Main HTML file
├── script.js           # JavaScript for handling user interaction and visualization
├── README.md           # This file
└── LICENSE             # License file
```

## Background on Neural ODEs

Neural Ordinary Differential Equations (Neural ODEs) represent a novel approach to modeling continuous-time dynamics of data. Unlike traditional neural networks that update states in discrete layers, Neural ODEs define the derivative of the hidden state with respect to time. This continuous formulation allows the model to learn complex temporal dependencies in time series data, making them particularly suitable for financial time series prediction.

The core idea can be expressed as:

```
dh(t)/dt = f(h(t), t, θ)
```

Where:
- h(t) is the hidden state at time t
- f is a neural network that computes the derivative
- θ represents the model parameters

For our trading strategy, we use Neural ODEs to predict future price movements, which then inform our trading decisions.

## Local Development

To run this project locally:

1. Clone the repository:
```
git clone https://github.com/yourusername/neural-ode-trading.git
cd neural-ode-trading
```

2. Open `index.html` in your web browser or use a local server:
```
# If you have Python installed:
python -m http.server
```

3. Access the site at `http://localhost:8000`

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This project is based on research in Neural ODEs for financial time series prediction
- The trading strategy logic is inspired by quantitative finance principles
- Chart.js for providing excellent visualization capabilities