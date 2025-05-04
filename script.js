// Chart.js instances
let priceChart, portfolioChart, drawdownChart, returnDistChart, sharpeChart;

// Sample data for visualization (this would normally come from your backend)
const sampleData = {
    ticker: 'SPY',
    dates: Array.from({length: 100}, (_, i) => {
        const date = new Date('2022-01-01');
        date.setDate(date.getDate() + i);
        return date.toISOString().split('T')[0];
    }),
    prices: Array.from({length: 100}, (_, i) => 100 + Math.sin(i/10) * 10 + i/5),
    predicted_prices: Array.from({length: 100}, (_, i) => 100 + Math.sin((i+3)/10) * 10 + i/5 + (Math.random() - 0.5) * 2),
    signals: Array.from({length: 100}, (_, i) => {
        if (i % 20 === 5) return 'BUY';
        if (i % 20 === 15) return 'SELL';
        if (i % 30 === 8) return 'SHORT';
        if (i % 30 === 18) return 'BUY_TO_COVER';
        return 'HOLD';
    }),
    portfolio_value: Array.from({length: 100}, (_, i) => 10000 * (1 + (Math.sin(i/15) * 0.05 + i/100))),
    buy_and_hold: Array.from({length: 100}, (_, i) => 10000 * (1 + i/100)),
    drawdown_strategy: Array.from({length: 100}, (_, i) => -Math.abs(Math.sin(i/20) * 8)),
    drawdown_bh: Array.from({length: 100}, (_, i) => -Math.abs(Math.sin(i/30) * 6)),
    daily_returns: Array.from({length: 99}, () => (Math.random() - 0.4) * 2),
    bh_daily_returns: Array.from({length: 99}, () => (Math.random() - 0.5) * 1.5),
    rolling_sharpe: Array.from({length: 70}, (_, i) => Math.sin(i/10) + 1.5 + (Math.random() - 0.5) * 0.5),
    rolling_sharpe_bh: Array.from({length: 70}, (_, i) => Math.sin(i/10) + 1 + (Math.random() - 0.5) * 0.5),
    metrics: {
        'Total Return (%)': {strategy: '45.32', buyhold: '32.15'},
        'Annualized Return (%)': {strategy: '18.76', buyhold: '14.22'},
        'Mean Daily Return (%)': {strategy: '0.0845', buyhold: '0.0654'},
        'Daily Volatility (%)': {strategy: '1.2375', buyhold: '1.4211'},
        'Annualized Volatility (%)': {strategy: '19.64', buyhold: '22.56'},
        'Sharpe Ratio': {strategy: '0.96', buyhold: '0.63'},
        'Maximum Drawdown (%)': {strategy: '-15.42', buyhold: '-18.76'},
        'Max Drawdown Duration (days)': {strategy: '37', buyhold: '42'},
        'Win Rate (%)': {strategy: '58.33', buyhold: 'N/A'},
        'Profit Factor': {strategy: '1.87', buyhold: 'N/A'}
    },
    advanced_metrics: {
        'Calmar Ratio': {strategy: '1.22', buyhold: '0.76'},
        'Sortino Ratio': {strategy: '1.38', buyhold: '0.92'},
        'Positive Days (%)': {strategy: '54.55', buyhold: '51.52'},
        'Max Consecutive Wins': {strategy: '7', buyhold: 'N/A'},
        'Max Consecutive Losses': {strategy: '4', buyhold: 'N/A'}
    }
};

// Initialize the page when DOM is fully loaded
document.addEventListener('DOMContentLoaded', function() {
    // Form submission handler
    document.getElementById('tickerForm').addEventListener('submit', function(e) {
        e.preventDefault();
        
        const ticker = document.getElementById('ticker').value.toUpperCase();
        if (!ticker) return;
        
        // Show loading spinner
        document.getElementById('loadingSpinner').style.display = 'inline-block';
        
        // In a real app, we would fetch data from a backend here
        // For this demo, we'll use sample data with a delay to simulate loading
        setTimeout(() => {
            displayResults(ticker);
            document.getElementById('loadingSpinner').style.display = 'none';
        }, 1500);
    });
});

// Display results based on ticker
function displayResults(ticker) {
    // Update ticker display
    document.getElementById('tickerDisplay').textContent = ticker;
    
    // Show results section
    document.getElementById('resultsSection').style.display = 'block';
    
    // Update the sample data with the new ticker
    const data = {...sampleData, ticker};
    
    // Render all charts and tables
    createPriceChart(data);
    createPortfolioChart(data);
    createDrawdownChart(data);
    createReturnDistChart(data);
    createSharpeChart(data);
    populateMetricsTable(data);
    populateAdvancedMetricsTable(data);
    
    // Scroll to results
    document.getElementById('resultsSection').scrollIntoView({behavior: 'smooth'});
}

// Create Price Chart with Signals
function createPriceChart(data) {
    const ctx = document.getElementById('priceChart').getContext('2d');
    
    // Destroy previous chart if it exists
    if (priceChart) priceChart.destroy();
    
    // Prepare signal points for scatter plot
    const buyPoints = [], sellPoints = [], shortPoints = [], coverPoints = [];
    
    data.signals.forEach((signal, idx) => {
        if (signal === 'BUY') {
            buyPoints.push({x: data.dates[idx], y: data.prices[idx]});
        } else if (signal === 'SELL') {
            sellPoints.push({x: data.dates[idx], y: data.prices[idx]});
        } else if (signal === 'SHORT') {
            shortPoints.push({x: data.dates[idx], y: data.prices[idx]});
        } else if (signal === 'BUY_TO_COVER') {
            coverPoints.push({x: data.dates[idx], y: data.prices[idx]});
        }
    });
    
    priceChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.dates,
            datasets: [
                {
                    label: 'Actual Price',
                    data: data.prices,
                    borderColor: 'rgba(0, 123, 255, 1)',
                    backgroundColor: 'rgba(0, 123, 255, 0.1)',
                    fill: false,
                    tension: 0.1
                },
                {
                    label: 'Predicted Price',
                    data: data.predicted_prices,
                    borderColor: 'rgba(40, 167, 69, 1)',
                    borderDash: [5, 5],
                    backgroundColor: 'rgba(40, 167, 69, 0.1)',
                    fill: false,
                    tension: 0.1
                },
                {
                    label: 'Buy Signals',
                    data: buyPoints,
                    borderColor: 'rgba(40, 167, 69, 1)',
                    backgroundColor: 'rgba(40, 167, 69, 1)',
                    pointRadius: 6,
                    pointStyle: 'triangle',
                    showLine: false
                },
                {
                    label: 'Sell Signals',
                    data: sellPoints,
                    borderColor: 'rgba(220, 53, 69, 1)',
                    backgroundColor: 'rgba(220, 53, 69, 1)',
                    pointRadius: 6,
                    pointStyle: 'triangle',
                    rotation: 180,
                    showLine: false
                },
                {
                    label: 'Short Signals',
                    data: shortPoints,
                    borderColor: 'rgba(111, 66, 193, 1)',
                    backgroundColor: 'rgba(111, 66, 193, 1)',
                    pointRadius: 6,
                    pointStyle: 'triangle',
                    rotation: 180,
                    showLine: false
                },
                {
                    label: 'Cover Signals',
                    data: coverPoints,
                    borderColor: 'rgba(255, 153, 0, 1)',
                    backgroundColor: 'rgba(255, 153, 0, 1)',
                    pointRadius: 6,
                    pointStyle: 'triangle',
                    showLine: false
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    type: 'category',
                    title: {
                        display: true,
                        text: 'Date'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Price'
                    }
                }
            },
            plugins: {
                title: {
                    display: true,
                    text: `${data.ticker} Price and Neural ODE Predictions`
                },
                tooltip: {
                    mode: 'index',
                    intersect: false
                }
            }
        }
    });
}

// Create Portfolio Performance Chart
function createPortfolioChart(data) {
    const ctx = document.getElementById('portfolioChart').getContext('2d');
    
    // Destroy previous chart if it exists
    if (portfolioChart) portfolioChart.destroy();
    
    portfolioChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.dates,
            datasets: [
                {
                    label: 'Trading Strategy',
                    data: data.portfolio_value,
                    borderColor: 'rgba(40, 167, 69, 1)',
                    backgroundColor: 'rgba(40, 167, 69, 0.1)',
                    fill: false
                },
                {
                    label: 'Buy and Hold',
                    data: data.buy_and_hold,
                    borderColor: 'rgba(0, 123, 255, 1)',
                    backgroundColor: 'rgba(0, 123, 255, 0.1)',
                    borderDash: [5, 5],
                    fill: false
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Date'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Portfolio Value ($)'
                    }
                }
            },
            plugins: {
                title: {
                    display: true,
                    text: 'Portfolio Performance Comparison'
                },
                tooltip: {
                    mode: 'index',
                    intersect: false
                }
            }
        }
    });
}

// Create Drawdown Chart
function createDrawdownChart(data) {
    const ctx = document.getElementById('drawdownChart').getContext('2d');
    
    // Destroy previous chart if it exists
    if (drawdownChart) drawdownChart.destroy();
    
    drawdownChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.dates,
            datasets: [
                {
                    label: 'Trading Strategy',
                    data: data.drawdown_strategy,
                    borderColor: 'rgba(40, 167, 69, 1)',
                    backgroundColor: 'rgba(40, 167, 69, 0.1)',
                    fill: false
                },
                {
                    label: 'Buy and Hold',
                    data: data.drawdown_bh,
                    borderColor: 'rgba(0, 123, 255, 1)',
                    backgroundColor: 'rgba(0, 123, 255, 0.1)',
                    fill: false
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Date'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Drawdown (%)'
                    },
                    ticks: {
                        callback: function(value) {
                            return value + '%';
                        }
                    }
                }
            },
            plugins: {
                title: {
                    display: true,
                    text: 'Drawdown Analysis'
                },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                    callbacks: {
                        label: function(context) {
                            return context.dataset.label + ': ' + context.parsed.y + '%';
                        }
                    }
                }
            }
        }
    });
}

// Create Return Distribution Chart
function createReturnDistChart(data) {
    const ctx = document.getElementById('returnDistChart').getContext('2d');
    
    // Destroy previous chart if it exists
    if (returnDistChart) returnDistChart.destroy();
    
    // Calculate bins for histogram
    const allReturns = [...data.daily_returns, ...data.bh_daily_returns];
    const min = Math.min(...allReturns);
    const max = Math.max(...allReturns);
    const binWidth = (max - min) / 20;
    
    // Create bins
    const bins = Array.from({length: 20}, (_, i) => min + i * binWidth);
    
    // Count values in each bin
    const strategyCounts = Array(20).fill(0);
    const bhCounts = Array(20).fill(0);
    
    data.daily_returns.forEach(val => {
        const binIndex = Math.min(Math.floor((val - min) / binWidth), 19);
        strategyCounts[binIndex]++;
    });
    
    data.bh_daily_returns.forEach(val => {
        const binIndex = Math.min(Math.floor((val - min) / binWidth), 19);
        bhCounts[binIndex]++;
    });
    
    returnDistChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: bins.map(b => b.toFixed(2)),
            datasets: [
                {
                    label: 'Trading Strategy',
                    data: strategyCounts,
                    backgroundColor: 'rgba(40, 167, 69, 0.6)',
                    borderColor: 'rgba(40, 167, 69, 1)',
                    borderWidth: 1
                },
                {
                    label: 'Buy and Hold',
                    data: bhCounts,
                    backgroundColor: 'rgba(0, 123, 255, 0.6)',
                    borderColor: 'rgba(0, 123, 255, 1)',
                    borderWidth: 1
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Daily Return (%)'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Frequency'
                    }
                }
            },
            plugins: {
                title: {
                    display: true,
                    text: 'Daily Return Distribution'
                },
                tooltip: {
                    callbacks: {
                        title: function(context) {
                            const index = context[0].dataIndex;
                            const binStart = parseFloat(bins[index].toFixed(2));
                            const binEnd = parseFloat((bins[index] + binWidth).toFixed(2));
                            return `Return: ${binStart}% to ${binEnd}%`;
                        }
                    }
                }
            }
        }
    });
}

// Create Rolling Sharpe Ratio Chart
function createSharpeChart(data) {
    const ctx = document.getElementById('sharpeChart').getContext('2d');
    
    // Destroy previous chart if it exists
    if (sharpeChart) sharpeChart.destroy();
    
    // We start at day 30 (since we use a 30-day window)
    const rollingDates = data.dates.slice(30);
    
    sharpeChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: rollingDates,
            datasets: [
                {
                    label: 'Trading Strategy',
                    data: data.rolling_sharpe,
                    borderColor: 'rgba(40, 167, 69, 1)',
                    backgroundColor: 'rgba(40, 167, 69, 0.1)',
                    fill: false
                },
                {
                    label: 'Buy and Hold',
                    data: data.rolling_sharpe_bh,
                    borderColor: 'rgba(0, 123, 255, 1)',
                    backgroundColor: 'rgba(0, 123, 255, 0.1)',
                    fill: false
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Date'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Sharpe Ratio (30-day window)'
                    }
                }
            },
            plugins: {
                title: {
                    display: true,
                    text: '30-Day Rolling Sharpe Ratio'
                },
                tooltip: {
                    mode: 'index',
                    intersect: false
                }
            }
        }
    });
}

// Populate Metrics Table
function populateMetricsTable(data) {
    const tableBody = document.getElementById('metricsTableBody');
    tableBody.innerHTML = '';
    
    for (const [metric, values] of Object.entries(data.metrics)) {
        const row = document.createElement('tr');
        
        const metricCell = document.createElement('td');
        metricCell.textContent = metric;
        
        const strategyCell = document.createElement('td');
        strategyCell.textContent = values.strategy;
        
        const buyholdCell = document.createElement('td');
        buyholdCell.textContent = values.buyhold;
        
        row.appendChild(metricCell);
        row.appendChild(strategyCell);
        row.appendChild(buyholdCell);
        
        tableBody.appendChild(row);
    }
}

// Populate Advanced Metrics Table
function populateAdvancedMetricsTable(data) {
    const tableBody = document.getElementById('advancedMetricsTableBody');
    tableBody.innerHTML = '';
    
    for (const [metric, values] of Object.entries(data.advanced_metrics)) {
        const row = document.createElement('tr');
        
        const metricCell = document.createElement('td');
        metricCell.textContent = metric;
        
        const strategyCell = document.createElement('td');
        strategyCell.textContent = values.strategy;
        
        const buyholdCell = document.createElement('td');
        buyholdCell.textContent = values.buyhold;
        
        row.appendChild(metricCell);
        row.appendChild(strategyCell);
        row.appendChild(buyholdCell);
        
        tableBody.appendChild(row);
    }
} 