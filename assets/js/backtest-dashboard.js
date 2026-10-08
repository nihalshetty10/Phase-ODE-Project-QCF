const BACKTEST_COLORS = {
  neural_ode: '#315a7d',
  lstm: '#a65b45',
  ridge_ar: '#7a6f99',
  buy_hold: '#6e8798',
  cash: '#999999',
  actual: '#222222'
};

const MODEL_LABELS = {
  neural_ode: 'Neural ODE',
  lstm: 'LSTM',
  ridge_ar: 'Ridge AR(20)',
  buy_hold: 'Buy & hold',
  cash: 'Cash'
};

let backtestData;
let selectedView = 'compare';
let chartInstances = {};

function formatPercent(value, digits = 1) {
  return value == null ? '—' : `${(value * 100).toFixed(digits)}%`;
}

function formatNumber(value, digits = 2) {
  return value == null ? '—' : Number(value).toFixed(digits);
}

function formatMoney(value) {
  if (value == null) return '—';
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    maximumFractionDigits: 0
  }).format(value);
}

function formatShortDate(isoDate) {
  return new Date(`${isoDate}T00:00:00`).toLocaleDateString('en-US', {
    month: 'short',
    year: '2-digit'
  });
}

function createLineDataset(modelKey, values, options = {}) {
  return {
    label: MODEL_LABELS[modelKey] || modelKey,
    data: values,
    borderColor: BACKTEST_COLORS[modelKey] || BACKTEST_COLORS.actual,
    backgroundColor: options.fill ? `${BACKTEST_COLORS[modelKey]}18` : 'transparent',
    pointRadius: 0,
    borderWidth: options.width || 2,
    borderDash: options.dash || [],
    fill: Boolean(options.fill),
    tension: 0.08,
    spanGaps: true
  };
}

function createChartOptions(formatYAxis) {
  return {
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: 'index', intersect: false },
    plugins: {
      legend: {
        labels: {
          color: '#555555',
          usePointStyle: true,
          pointStyle: 'line',
          padding: 18,
          font: { size: 11 }
        }
      },
      tooltip: {
        backgroundColor: '#ffffff',
        titleColor: '#222222',
        bodyColor: '#444444',
        borderColor: '#cccccc',
        borderWidth: 1,
        padding: 12
      }
    },
    scales: {
      x: {
        grid: { color: 'rgba(0,0,0,.05)' },
        ticks: {
          color: '#777777',
          maxTicksLimit: 7,
          callback: (_, index) => formatShortDate(backtestData.dates[index])
        }
      },
      y: {
        grid: { color: 'rgba(0,0,0,.07)' },
        ticks: { color: '#777777', callback: formatYAxis }
      }
    }
  };
}

function destroyExistingCharts() {
  Object.values(chartInstances).forEach((chart) => chart.destroy());
  chartInstances = {};
}

function getVisibleStrategyKeys() {
  if (selectedView === 'compare') {
    return ['neural_ode', 'lstm', 'ridge_ar', 'buy_hold', 'cash'];
  }
  return [selectedView, 'buy_hold'];
}

function renderInteractiveCharts() {
  destroyExistingCharts();
  const visibleStrategies = getVisibleStrategyKeys();

  chartInstances.equity = new Chart(document.getElementById('equityChart'), {
    type: 'line',
    data: {
      labels: backtestData.dates,
      datasets: visibleStrategies.map((modelKey) =>
        createLineDataset(modelKey, backtestData.models[modelKey].equity, {
          width: modelKey === 'cash' ? 1 : 2,
          dash: modelKey === 'cash' ? [4, 5] : []
        })
      )
    },
    options: createChartOptions((value) => `$${(value / 1000).toFixed(0)}k`)
  });

  const forecastModelKeys =
    selectedView === 'compare' ? ['neural_ode', 'lstm', 'ridge_ar'] : [selectedView];
  const forecastDatasets = [
    createLineDataset('actual', backtestData.actual_price, { width: 2 })
  ];
  forecastDatasets[0].label = 'Actual adjusted close';
  forecastModelKeys.forEach((modelKey) => {
    forecastDatasets.push(
      createLineDataset(modelKey, backtestData.models[modelKey].predicted_price, { width: 1 })
    );
  });

  chartInstances.price = new Chart(document.getElementById('priceChart'), {
    type: 'line',
    data: { labels: backtestData.dates, datasets: forecastDatasets },
    options: createChartOptions((value) => `$${Number(value).toFixed(0)}`)
  });

  chartInstances.drawdown = new Chart(document.getElementById('drawdownChart'), {
    type: 'line',
    data: {
      labels: backtestData.dates,
      datasets: visibleStrategies
        .filter((modelKey) => modelKey !== 'cash')
        .map((modelKey) =>
          createLineDataset(modelKey, backtestData.models[modelKey].drawdown, {
            fill: selectedView !== 'compare' && modelKey === selectedView
          })
        )
    },
    options: createChartOptions((value) => formatPercent(value, 0))
  });
}

function renderPerformanceTable() {
  const tableBody = document.getElementById('metricsBody');
  tableBody.innerHTML = Object.keys(MODEL_LABELS)
    .map((modelKey) => {
      const modelMetrics = backtestData.models[modelKey].metrics;
      const isForecastModel = ['neural_ode', 'lstm', 'ridge_ar'].includes(modelKey);
      const exposure = isForecastModel
        ? formatPercent(modelMetrics.gross_exposure, 0)
        : modelKey === 'buy_hold'
          ? '100%'
          : '0%';

      return `<tr>
        <th><span class="strategy-dot" style="background:${BACKTEST_COLORS[modelKey]}"></span>${MODEL_LABELS[modelKey]}</th>
        <td>${formatPercent(modelMetrics.total_return)}</td>
        <td>${formatPercent(modelMetrics.cagr)}</td>
        <td>${formatNumber(modelMetrics.sharpe)}</td>
        <td>${formatPercent(modelMetrics.annual_volatility)}</td>
        <td>${formatPercent(modelMetrics.max_drawdown)}</td>
        <td>${formatNumber(modelMetrics.sortino)}</td>
        <td>${isForecastModel ? modelMetrics.trades : '—'}</td>
        <td>${exposure}</td>
      </tr>`;
    })
    .join('');
}

function renderSummaryText() {
  const metadata = backtestData.metadata;
  const neuralOdeMetrics = backtestData.models.neural_ode.metrics;
  const buyAndHoldMetrics = backtestData.models.buy_hold.metrics;
  const returnGap = neuralOdeMetrics.total_return - buyAndHoldMetrics.total_return;

  document.getElementById('verdictValue').textContent = formatPercent(
    neuralOdeMetrics.total_return
  );
  document.getElementById('verdictText').textContent =
    `Buy-and-hold returned ${formatPercent(buyAndHoldMetrics.total_return)} over the same period.`;
  document.getElementById('testRange').textContent = metadata.test_period;
  document.getElementById('instrument').textContent = `${metadata.symbol} · adjusted`;
  document.getElementById('observations').textContent =
    metadata.test_observations.toLocaleString();
  document.getElementById('cost').textContent = `${metadata.transaction_cost_bps} bps / turn`;
  document.getElementById('seed').textContent = metadata.seed;

  const headlineStatistics = [
    [
      'Buy & hold return',
      formatPercent(buyAndHoldMetrics.total_return),
      `Final value ${formatMoney(buyAndHoldMetrics.final_value)}`,
      'buy_hold'
    ],
    [
      'Neural ODE Sharpe',
      formatNumber(neuralOdeMetrics.sharpe),
      `Buy-and-hold: ${formatNumber(buyAndHoldMetrics.sharpe)}`,
      'neural_ode'
    ],
    [
      'Neural ODE drawdown',
      formatPercent(neuralOdeMetrics.max_drawdown),
      `Buy-and-hold: ${formatPercent(buyAndHoldMetrics.max_drawdown)}`,
      'neural_ode'
    ],
    [
      'Neural ODE exposure',
      formatPercent(neuralOdeMetrics.gross_exposure, 0),
      `${neuralOdeMetrics.trades} position changes`,
      'neural_ode'
    ]
  ];

  document.getElementById('statGrid').innerHTML = headlineStatistics
    .map(
      ([label, value, detail, modelKey]) =>
        `<article style="--accent:${BACKTEST_COLORS[modelKey]}">
          <span>${label}</span><strong>${value}</strong><small>${detail}</small>
        </article>`
    )
    .join('');

  const neuralOdeForecast = backtestData.models.neural_ode.forecast_metrics;
  const lstmForecast = backtestData.models.lstm.forecast_metrics;
  const returnGapInPoints = (Math.abs(returnGap) * 100).toFixed(1);
  const lstmThreshold = formatNumber(backtestData.models.lstm.threshold * 100, 2);

  document.getElementById('findings').innerHTML = `
    <p><strong>Buy-and-hold did better.</strong> It returned ${formatPercent(
      buyAndHoldMetrics.total_return
    )}, compared with ${formatPercent(
      neuralOdeMetrics.total_return
    )} for the Neural ODE. The difference was ${returnGapInPoints} percentage points.</p>
    <p><strong>The Neural ODE was in the market less often.</strong> Average gross exposure was ${formatPercent(
      neuralOdeMetrics.gross_exposure,
      0
    )}. Its ${formatPercent(
      Math.abs(neuralOdeMetrics.max_drawdown)
    )} maximum drawdown should be read in that context.</p>
    <p><strong>The LSTM made no test trades.</strong> Its 2022 validation threshold was ${lstmThreshold}%. Although it got the return sign right on ${formatPercent(
      lstmForecast.directional_accuracy
    )} of test days, its forecasts never crossed that threshold. Neural ODE directional accuracy was ${formatPercent(
      neuralOdeForecast.directional_accuracy
    )}.</p>`;

  document.getElementById('trainPeriod').textContent = metadata.train_period;
  document.getElementById('validationPeriod').textContent = metadata.validation_period;
  document.getElementById('testPeriod').textContent = metadata.test_period;
  document.getElementById('dataRange').textContent =
    `${metadata.data_snapshot_start} through ${metadata.data_snapshot_end}`;
  document.getElementById('generatedAt').textContent =
    `generated ${new Date(metadata.generated_at).toLocaleString()}`;
}

function bindModelTabs() {
  document.querySelectorAll('.tab').forEach((button) => {
    button.addEventListener('click', () => {
      selectedView = button.dataset.view;
      document.querySelectorAll('.tab').forEach((tab) => {
        const isSelected = tab === button;
        tab.classList.toggle('active', isSelected);
        tab.setAttribute('aria-selected', String(isSelected));
      });
      renderInteractiveCharts();
    });
  });
}

async function initializeDashboard() {
  try {
    const response = await fetch('assets/data/backtest-results.json', { cache: 'no-store' });
    if (!response.ok) {
      throw new Error(`backtest-results.json returned ${response.status}`);
    }

    backtestData = await response.json();
    renderSummaryText();
    renderPerformanceTable();
    renderInteractiveCharts();
    bindModelTabs();
  } catch (error) {
    const errorBanner = document.getElementById('errorBanner');
    errorBanner.hidden = false;
    errorBanner.textContent =
      `Unable to load the saved backtest results: ${error.message}. ` +
      'Serve this directory over HTTP instead of opening index.html directly.';
    document.getElementById('verdictValue').textContent = 'Data unavailable';
  }
}

initializeDashboard();
