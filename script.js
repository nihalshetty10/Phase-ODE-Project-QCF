// Chart.js instances
let priceChart, portfolioChart, drawdownChart, returnDistChart, sharpeChart;

// Current active model view
let activeView = 'compare'; // 'compare', 'neuralode', or 'lstm'

// Sample data for visualization (this would normally come from your backend)
const sampleData = {
    ticker: 'SPY',
    dates: Array.from({length: 100}, (_, i) => {
        const date = new Date('2022-01-01');
        date.setDate(date.getDate() + i);
        return date.toISOString().split('T')[0];
    }),
    prices: Array.from({length: 100}, (_, i) => 100 + Math.sin(i/10) * 10 + i/5),
    
    // Neural ODE model data
    neuralode: {
        predicted_prices: Array.from({length: 100}, (_, i) => 100 + Math.sin((i+3)/10) * 10 + i/5 + (Math.random() - 0.5) * 2),
        signals: Array.from({length: 100}, (_, i) => {
            if (i % 20 === 5) return 'BUY';
            if (i % 20 === 15) return 'SELL';
            if (i % 30 === 8) return 'SHORT';
            if (i % 30 === 18) return 'BUY_TO_COVER';
            return 'HOLD';
        }),
        portfolio_value: Array.from({length: 100}, (_, i) => 10000 * (1 + (Math.sin(i/15) * 0.05 + i/100))),
        drawdown: Array.from({length: 100}, (_, i) => -Math.abs(Math.sin(i/20) * 8)),
        daily_returns: Array.from({length: 99}, () => (Math.random() - 0.4) * 2),
        rolling_sharpe: Array.from({length: 70}, (_, i) => Math.sin(i/10) + 1.5 + (Math.random() - 0.5) * 0.5),
    },
    
    // LSTM model data (with slightly different characteristics)
    lstm: {
        predicted_prices: Array.from({length: 100}, (_, i) => 100 + Math.sin((i+2)/10) * 10 + i/5 + (Math.random() - 0.5) * 2.5),
        signals: Array.from({length: 100}, (_, i) => {
            if (i % 18 === 3) return 'BUY';
            if (i % 18 === 12) return 'SELL';
            if (i % 25 === 6) return 'SHORT';
            if (i % 25 === 16) return 'BUY_TO_COVER';
            return 'HOLD';
        }),
        portfolio_value: Array.from({length: 100}, (_, i) => 10000 * (1 + (Math.sin(i/12) * 0.06 + i/95))),
        drawdown: Array.from({length: 100}, (_, i) => -Math.abs(Math.sin(i/18) * 9)),
        daily_returns: Array.from({length: 99}, () => (Math.random() - 0.38) * 2.2),
        rolling_sharpe: Array.from({length: 70}, (_, i) => Math.sin(i/12) + 1.3 + (Math.random() - 0.5) * 0.6),
    },
    
    // Buy and hold strategy
    buy_and_hold: Array.from({length: 100}, (_, i) => 10000 * (1 + i/100)),
    drawdown_bh: Array.from({length: 100}, (_, i) => -Math.abs(Math.sin(i/30) * 6)),
    bh_daily_returns: Array.from({length: 99}, () => (Math.random() - 0.5) * 1.5),
    rolling_sharpe_bh: Array.from({length: 70}, (_, i) => Math.sin(i/10) + 1 + (Math.random() - 0.5) * 0.5),
    
    // Performance metrics
    metrics: {
        'Total Return (%)': {neuralode: '45.32', lstm: '47.88', buyhold: '32.15'},
        'Annualized Return (%)': {neuralode: '18.76', lstm: '19.54', buyhold: '14.22'},
        'Mean Daily Return (%)': {neuralode: '0.0845', lstm: '0.0892', buyhold: '0.0654'},
        'Daily Volatility (%)': {neuralode: '1.2375', lstm: '1.4125', buyhold: '1.4211'},
        'Annualized Volatility (%)': {neuralode: '19.64', lstm: '22.42', buyhold: '22.56'},
        'Sharpe Ratio': {neuralode: '0.96', lstm: '0.87', buyhold: '0.63'},
        'Maximum Drawdown (%)': {neuralode: '-15.42', lstm: '-17.65', buyhold: '-18.76'},
        'Win Rate (%)': {neuralode: '58.33', lstm: '56.78', buyhold: 'N/A'},
        'Profit Factor': {neuralode: '1.87', lstm: '1.74', buyhold: 'N/A'}
    },
    
    // Advanced metrics
    advanced_metrics: {
        'Calmar Ratio': {neuralode: '1.22', lstm: '1.11', buyhold: '0.76'},
        'Sortino Ratio': {neuralode: '1.38', lstm: '1.25', buyhold: '0.92'},
        'Positive Days (%)': {neuralode: '54.55', lstm: '53.21', buyhold: '51.52'},
        'Max Consecutive Wins': {neuralode: '7', lstm: '6', buyhold: 'N/A'},
        'Max Consecutive Losses': {neuralode: '4', lstm: '5', buyhold: 'N/A'}
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
    
    // Add event listeners for model selection buttons
    document.getElementById('btnCompare').addEventListener('click', function() {
        setActiveView('compare');
    });
    
    document.getElementById('btnNeuralODE').addEventListener('click', function() {
        setActiveView('neuralode');
    });
    
    document.getElementById('btnLSTM').addEventListener('click', function() {
        setActiveView('lstm');
    });
});

// Set the active view and update UI
function setActiveView(view) {
    activeView = view;
    
    // Update button styles
    document.querySelectorAll('.model-selector .btn').forEach(btn => {
        btn.classList.remove('active');
    });
    
    const activeButton = document.getElementById(`btn${view.charAt(0).toUpperCase() + view.slice(1)}`);
    if (activeButton) {
        activeButton.classList.add('active');
    }
    
    // Update charts based on the selected view
    const ticker = document.getElementById('tickerDisplay').textContent;
    displayResults(ticker);
}

// Display results based on ticker
function displayResults(ticker) {
    // Update ticker display
    document.getElementById('tickerDisplay').textContent = ticker;
    
    // Show results section
    document.getElementById('resultsSection').style.display = 'block';
    
    // Update the sample data with the new ticker
    const data = {...sampleData, ticker};
    
    // Render all charts and tables based on active view
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
    
    // Datasets based on active view
    const datasets = [];
    
    // Always include actual price
    datasets.push({
        label: 'Actual Price',
        data: data.prices,
        borderColor: 'rgba(0, 123, 255, 1)',
        backgroundColor: 'rgba(0, 123, 255, 0.1)',
        fill: false,
        tension: 0.1
    });
    
    // Prepare signal points for scatter plot
    let neuralodesignals = [], lstmsignals = [];
    
    if (activeView === 'compare' || activeView === 'neuralode') {
        // Add Neural ODE predictions
        datasets.push({
            label: 'Neural ODE Predictions',
            data: data.neuralode.predicted_prices,
            borderColor: 'rgba(40, 167, 69, 1)',
            borderDash: [5, 5],
            backgroundColor: 'rgba(40, 167, 69, 0.1)',
            fill: false,
            tension: 0.1
        });
        
        // Prepare Neural ODE signal points
        neuralodesignals = prepareTradingSignals(data.dates, data.prices, data.neuralode.signals);
    }
    
    if (activeView === 'compare' || activeView === 'lstm') {
        // Add LSTM predictions
        datasets.push({
            label: 'LSTM Predictions',
            data: data.lstm.predicted_prices,
            borderColor: 'rgba(220, 53, 69, 1)',
            borderDash: [5, 5],
            backgroundColor: 'rgba(220, 53, 69, 0.1)',
            fill: false,
            tension: 0.1
        });
        
        // Prepare LSTM signal points
        lstmsignals = prepareTradingSignals(data.dates, data.prices, data.lstm.signals);
    }
    
    // Add signal datasets based on active view
    if (activeView === 'neuralode' || activeView === 'compare') {
        datasets.push(...createSignalDatasets(neuralodesignals, 'Neural ODE'));
    }
    
    if (activeView === 'lstm' || activeView === 'compare') {
        datasets.push(...createSignalDatasets(lstmsignals, 'LSTM'));
    }
    
    priceChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.dates,
            datasets: datasets
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
                    text: `${data.ticker} Price and Model Predictions`
                },
                tooltip: {
                    mode: 'index',
                    intersect: false
                }
            }
        }
    });
}

// Helper function to prepare trading signals
function prepareTradingSignals(dates, prices, signals) {
    const buyPoints = [], sellPoints = [], shortPoints = [], coverPoints = [];
    
    signals.forEach((signal, idx) => {
        if (signal === 'BUY') {
            buyPoints.push({x: dates[idx], y: prices[idx]});
        } else if (signal === 'SELL') {
            sellPoints.push({x: dates[idx], y: prices[idx]});
        } else if (signal === 'SHORT') {
            shortPoints.push({x: dates[idx], y: prices[idx]});
        } else if (signal === 'BUY_TO_COVER') {
            coverPoints.push({x: dates[idx], y: prices[idx]});
        }
    });
    
    return { buyPoints, sellPoints, shortPoints, coverPoints };
}

// Helper function to create signal datasets
function createSignalDatasets(signals, prefix) {
    const { buyPoints, sellPoints, shortPoints, coverPoints } = signals;
    
    // Color variations based on model
    const colors = prefix === 'Neural ODE' 
        ? { buy: 'rgba(40, 167, 69, 1)', sell: 'rgba(220, 53, 69, 1)', 
            short: 'rgba(111, 66, 193, 1)', cover: 'rgba(255, 153, 0, 1)' }
        : { buy: 'rgba(20, 120, 80, 1)', sell: 'rgba(180, 30, 45, 1)', 
            short: 'rgba(90, 50, 150, 1)', cover: 'rgba(230, 130, 20, 1)' };
    
    return [
        {
            label: `${prefix} Buy Signals`,
            data: buyPoints,
            borderColor: colors.buy,
            backgroundColor: colors.buy,
            pointRadius: 6,
            pointStyle: 'triangle',
            showLine: false
        },
        {
            label: `${prefix} Sell Signals`,
            data: sellPoints,
            borderColor: colors.sell,
            backgroundColor: colors.sell,
            pointRadius: 6,
            pointStyle: 'triangle',
            rotation: 180,
            showLine: false
        },
        {
            label: `${prefix} Short Signals`,
            data: shortPoints,
            borderColor: colors.short,
            backgroundColor: colors.short,
            pointRadius: 6,
            pointStyle: 'triangle',
            rotation: 180,
            showLine: false
        },
        {
            label: `${prefix} Cover Signals`,
            data: coverPoints,
            borderColor: colors.cover,
            backgroundColor: colors.cover,
            pointRadius: 6,
            pointStyle: 'triangle',
            showLine: false
        }
    ];
}

// Create Portfolio Performance Chart
function createPortfolioChart(data) {
    const ctx = document.getElementById('portfolioChart').getContext('2d');
    
    // Destroy previous chart if it exists
    if (portfolioChart) portfolioChart.destroy();
    
    // Datasets based on active view
    const datasets = [];
    
    if (activeView === 'compare' || activeView === 'neuralode') {
        datasets.push({
            label: 'Neural ODE Strategy',
            data: data.neuralode.portfolio_value,
            borderColor: 'rgba(40, 167, 69, 1)',
            backgroundColor: 'rgba(40, 167, 69, 0.1)',
            fill: false
        });
    }
    
    if (activeView === 'compare' || activeView === 'lstm') {
        datasets.push({
            label: 'LSTM Strategy',
            data: data.lstm.portfolio_value,
            borderColor: 'rgba(220, 53, 69, 1)',
            backgroundColor: 'rgba(220, 53, 69, 0.1)',
            fill: false
        });
    }
    
    // Always include buy and hold
    datasets.push({
        label: 'Buy and Hold',
        data: data.buy_and_hold,
        borderColor: 'rgba(0, 123, 255, 1)',
        backgroundColor: 'rgba(0, 123, 255, 0.1)',
        borderDash: [5, 5],
        fill: false
    });
    
    portfolioChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.dates,
            datasets: datasets
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
    
    // Datasets based on active view
    const datasets = [];
    
    if (activeView === 'compare' || activeView === 'neuralode') {
        datasets.push({
            label: 'Neural ODE Strategy',
            data: data.neuralode.drawdown,
            borderColor: 'rgba(40, 167, 69, 1)',
            backgroundColor: 'rgba(40, 167, 69, 0.1)',
            fill: false
        });
    }
    
    if (activeView === 'compare' || activeView === 'lstm') {
        datasets.push({
            label: 'LSTM Strategy',
            data: data.lstm.drawdown,
            borderColor: 'rgba(220, 53, 69, 1)',
            backgroundColor: 'rgba(220, 53, 69, 0.1)',
            fill: false
        });
    }
    
    // Always include buy and hold
    datasets.push({
        label: 'Buy and Hold',
        data: data.drawdown_bh,
        borderColor: 'rgba(0, 123, 255, 1)',
        backgroundColor: 'rgba(0, 123, 255, 0.1)',
        borderDash: [5, 5],
        fill: false
    });
    
    drawdownChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.dates,
            datasets: datasets
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
                    intersect: false
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
    
    // Prepare bins for histogram
    const bins = Array.from({length: 20}, (_, i) => -2 + i * 0.2);
    
    // Datasets based on active view
    const datasets = [];
    let histogramData = [];
    
    if (activeView === 'compare' || activeView === 'neuralode') {
        // Neural ODE histogram data
        const neuralodeHistogram = createHistogram(data.neuralode.daily_returns, bins);
        datasets.push({
            label: 'Neural ODE Strategy',
            data: neuralodeHistogram,
            backgroundColor: 'rgba(40, 167, 69, 0.5)',
            borderColor: 'rgba(40, 167, 69, 1)',
            borderWidth: 1
        });
        histogramData.push(neuralodeHistogram);
    }
    
    if (activeView === 'compare' || activeView === 'lstm') {
        // LSTM histogram data
        const lstmHistogram = createHistogram(data.lstm.daily_returns, bins);
        datasets.push({
            label: 'LSTM Strategy',
            data: lstmHistogram,
            backgroundColor: 'rgba(220, 53, 69, 0.5)',
            borderColor: 'rgba(220, 53, 69, 1)',
            borderWidth: 1
        });
        histogramData.push(lstmHistogram);
    }
    
    // Always include buy and hold
    const bhHistogram = createHistogram(data.bh_daily_returns, bins);
    datasets.push({
        label: 'Buy and Hold',
        data: bhHistogram,
        backgroundColor: 'rgba(0, 123, 255, 0.5)',
        borderColor: 'rgba(0, 123, 255, 1)',
        borderWidth: 1
    });
    histogramData.push(bhHistogram);
    
    // Find maximum Y value across all datasets
    const maxY = Math.max(...histogramData.flat());
    
    returnDistChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: bins.map(b => b.toFixed(1)),
            datasets: datasets
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
                    },
                    max: maxY + 2 // Add some padding
                }
            },
            plugins: {
                title: {
                    display: true,
                    text: 'Daily Return Distribution'
                },
                tooltip: {
                    mode: 'index',
                    intersect: false
                }
            }
        }
    });
}

// Helper function to create histogram data
function createHistogram(data, bins) {
    const histogram = Array(bins.length).fill(0);
    
    data.forEach(value => {
        for (let i = 0; i < bins.length; i++) {
            if (value < bins[i] || i === bins.length - 1) {
                histogram[i]++;
                break;
            }
        }
    });
    
    return histogram;
}

// Create Sharpe Ratio Chart
function createSharpeChart(data) {
    const ctx = document.getElementById('sharpeChart').getContext('2d');
    
    // Destroy previous chart if it exists
    if (sharpeChart) sharpeChart.destroy();
    
    // Create dates for rolling sharpe (subset of main dates)
    const sharpeDates = data.dates.slice(30, 100);
    
    // Datasets based on active view
    const datasets = [];
    
    if (activeView === 'compare' || activeView === 'neuralode') {
        datasets.push({
            label: 'Neural ODE Strategy',
            data: data.neuralode.rolling_sharpe,
            borderColor: 'rgba(40, 167, 69, 1)',
            backgroundColor: 'rgba(40, 167, 69, 0.1)',
            fill: false
        });
    }
    
    if (activeView === 'compare' || activeView === 'lstm') {
        datasets.push({
            label: 'LSTM Strategy',
            data: data.lstm.rolling_sharpe,
            borderColor: 'rgba(220, 53, 69, 1)',
            backgroundColor: 'rgba(220, 53, 69, 0.1)',
            fill: false
        });
    }
    
    // Always include buy and hold
    datasets.push({
        label: 'Buy and Hold',
        data: data.rolling_sharpe_bh,
        borderColor: 'rgba(0, 123, 255, 1)',
        backgroundColor: 'rgba(0, 123, 255, 0.1)',
        borderDash: [5, 5],
        fill: false
    });
    
    sharpeChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: sharpeDates,
            datasets: datasets
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
                        text: 'Rolling Sharpe (30-day)'
                    }
                }
            },
            plugins: {
                title: {
                    display: true,
                    text: 'Rolling Sharpe Ratio'
                },
                tooltip: {
                    mode: 'index',
                    intersect: false
                }
            }
        }
    });
}

// Populate metrics table
function populateMetricsTable(data) {
    const tbody = document.getElementById('metricsTableBody');
    tbody.innerHTML = '';
    
    for (const [metric, values] of Object.entries(data.metrics)) {
        const row = document.createElement('tr');
        
        // Add metric name
        const metricCell = document.createElement('td');
        metricCell.textContent = metric;
        row.appendChild(metricCell);
        
        // Add values based on active view
        if (activeView === 'compare') {
            // Neural ODE
            const neuralodeCell = document.createElement('td');
            neuralodeCell.textContent = values.neuralode;
            row.appendChild(neuralodeCell);
            
            // LSTM
            const lstmCell = document.createElement('td');
            lstmCell.textContent = values.lstm;
            row.appendChild(lstmCell);
            
            // Buy and Hold
            const buyHoldCell = document.createElement('td');
            buyHoldCell.textContent = values.buyhold;
            row.appendChild(buyHoldCell);
        } else if (activeView === 'neuralode') {
            // Neural ODE
            const neuralodeCell = document.createElement('td');
            neuralodeCell.textContent = values.neuralode;
            row.appendChild(neuralodeCell);
            
            // Buy and Hold
            const buyHoldCell = document.createElement('td');
            buyHoldCell.textContent = values.buyhold;
            row.appendChild(buyHoldCell);
        } else if (activeView === 'lstm') {
            // LSTM
            const lstmCell = document.createElement('td');
            lstmCell.textContent = values.lstm;
            row.appendChild(lstmCell);
            
            // Buy and Hold
            const buyHoldCell = document.createElement('td');
            buyHoldCell.textContent = values.buyhold;
            row.appendChild(buyHoldCell);
        }
        
        tbody.appendChild(row);
    }
}

// Populate advanced metrics table
function populateAdvancedMetricsTable(data) {
    const tbody = document.getElementById('advancedMetricsTableBody');
    tbody.innerHTML = '';
    
    for (const [metric, values] of Object.entries(data.advanced_metrics)) {
        const row = document.createElement('tr');
        
        // Add metric name
        const metricCell = document.createElement('td');
        metricCell.textContent = metric;
        row.appendChild(metricCell);
        
        // Add values based on active view
        if (activeView === 'compare') {
            // Neural ODE
            const neuralodeCell = document.createElement('td');
            neuralodeCell.textContent = values.neuralode;
            row.appendChild(neuralodeCell);
            
            // LSTM
            const lstmCell = document.createElement('td');
            lstmCell.textContent = values.lstm;
            row.appendChild(lstmCell);
            
            // Buy and Hold
            const buyHoldCell = document.createElement('td');
            buyHoldCell.textContent = values.buyhold;
            row.appendChild(buyHoldCell);
        } else if (activeView === 'neuralode') {
            // Neural ODE
            const neuralodeCell = document.createElement('td');
            neuralodeCell.textContent = values.neuralode;
            row.appendChild(neuralodeCell);
            
            // Buy and Hold
            const buyHoldCell = document.createElement('td');
            buyHoldCell.textContent = values.buyhold;
            row.appendChild(buyHoldCell);
        } else if (activeView === 'lstm') {
            // LSTM
            const lstmCell = document.createElement('td');
            lstmCell.textContent = values.lstm;
            row.appendChild(lstmCell);
            
            // Buy and Hold
            const buyHoldCell = document.createElement('td');
            buyHoldCell.textContent = values.buyhold;
            row.appendChild(buyHoldCell);
        }
        
        tbody.appendChild(row);
    }
} 