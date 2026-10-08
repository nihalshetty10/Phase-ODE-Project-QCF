const COLORS = { neural_ode:'#315a7d', lstm:'#a65b45', ridge_ar:'#7a6f99', buy_hold:'#6e8798', cash:'#999999', actual:'#222222' };
const LABELS = { neural_ode:'Neural ODE', lstm:'LSTM', ridge_ar:'Ridge AR(20)', buy_hold:'Buy & hold', cash:'Cash' };
let results, activeView = 'compare', charts = {};
const percent = (v, d=1) => v == null ? '—' : `${(v*100).toFixed(d)}%`;
const number = (v, d=2) => v == null ? '—' : Number(v).toFixed(d);
const money = v => v == null ? '—' : new Intl.NumberFormat('en-US',{style:'currency',currency:'USD',maximumFractionDigits:0}).format(v);
const shortDate = v => new Date(`${v}T00:00:00`).toLocaleDateString('en-US',{month:'short',year:'2-digit'});

function lineDataset(key, values, options={}) {
  return { label:LABELS[key]||key, data:values, borderColor:COLORS[key]||COLORS.actual, backgroundColor:options.fill?`${COLORS[key]}18`:'transparent', pointRadius:0, borderWidth:options.width||2, borderDash:options.dash||[], fill:Boolean(options.fill), tension:.08, spanGaps:true };
}
function baseOptions(yFormatter) {
  return { responsive:true, maintainAspectRatio:false, interaction:{mode:'index',intersect:false}, plugins:{legend:{labels:{color:'#555555',usePointStyle:true,pointStyle:'line',padding:18,font:{size:11}}},tooltip:{backgroundColor:'#ffffff',titleColor:'#222222',bodyColor:'#444444',borderColor:'#cccccc',borderWidth:1,padding:12}}, scales:{x:{grid:{color:'rgba(0,0,0,.05)'},ticks:{color:'#777777',maxTicksLimit:7,callback:(_,i)=>shortDate(results.dates[i])}},y:{grid:{color:'rgba(0,0,0,.07)'},ticks:{color:'#777777',callback:yFormatter}}} };
}
function destroyCharts(){ Object.values(charts).forEach(c=>c.destroy()); charts={}; }
function chartKeys(){ return activeView==='compare' ? ['neural_ode','lstm','ridge_ar','buy_hold','cash'] : [activeView,'buy_hold']; }
function renderCharts(){
  destroyCharts(); const keys=chartKeys();
  charts.equity=new Chart(document.getElementById('equityChart'),{type:'line',data:{labels:results.dates,datasets:keys.map(k=>lineDataset(k,results.models[k].equity,{width:k==='cash'?1:2,dash:k==='cash'?[4,5]:[]}))},options:baseOptions(v=>`$${(v/1000).toFixed(0)}k`)});
  const forecastKeys=activeView==='compare'?['neural_ode','lstm','ridge_ar']:[activeView];
  const forecastData=[lineDataset('actual',results.actual_price,{width:2})]; forecastData[0].label='Actual adjusted close';
  forecastKeys.forEach(k=>forecastData.push(lineDataset(k,results.models[k].predicted_price,{width:1})));
  charts.price=new Chart(document.getElementById('priceChart'),{type:'line',data:{labels:results.dates,datasets:forecastData},options:baseOptions(v=>`$${Number(v).toFixed(0)}`)});
  charts.drawdown=new Chart(document.getElementById('drawdownChart'),{type:'line',data:{labels:results.dates,datasets:keys.filter(k=>k!=='cash').map(k=>lineDataset(k,results.models[k].drawdown,{fill:activeView!=='compare'&&k===activeView}))},options:baseOptions(v=>percent(v,0))});
}
function renderMetrics(){
  document.getElementById('metricsBody').innerHTML=Object.keys(LABELS).map(k=>{const m=results.models[k].metrics,isModel=['neural_ode','lstm','ridge_ar'].includes(k);return `<tr><th><span class="strategy-dot" style="background:${COLORS[k]}"></span>${LABELS[k]}</th><td>${percent(m.total_return)}</td><td>${percent(m.cagr)}</td><td>${number(m.sharpe)}</td><td>${percent(m.annual_volatility)}</td><td>${percent(m.max_drawdown)}</td><td>${number(m.sortino)}</td><td>${isModel?m.trades:'—'}</td><td>${isModel?percent(m.gross_exposure,0):(k==='buy_hold'?'100%':'0%')}</td></tr>`}).join('');
}
function renderSummary(){
  const meta=results.metadata,ode=results.models.neural_ode.metrics,passive=results.models.buy_hold.metrics,gap=ode.total_return-passive.total_return;
  document.getElementById('verdictValue').textContent=percent(ode.total_return);
  document.getElementById('verdictText').textContent=`Buy-and-hold returned ${percent(passive.total_return)} over the same period.`;
  document.getElementById('testRange').textContent=meta.test_period;
  document.getElementById('instrument').textContent=`${meta.symbol} · adjusted`; document.getElementById('observations').textContent=meta.test_observations.toLocaleString(); document.getElementById('cost').textContent=`${meta.transaction_cost_bps} bps / turn`; document.getElementById('seed').textContent=meta.seed;
  const stats=[['Buy & hold return',percent(passive.total_return),`Final value ${money(passive.final_value)}`,'buy_hold'],['Neural ODE Sharpe',number(ode.sharpe),`Buy-and-hold: ${number(passive.sharpe)}`,'neural_ode'],['Neural ODE drawdown',percent(ode.max_drawdown),`Buy-and-hold: ${percent(passive.max_drawdown)}`,'neural_ode'],['Neural ODE exposure',percent(ode.gross_exposure,0),`${ode.trades} position changes`,'neural_ode']];
  document.getElementById('statGrid').innerHTML=stats.map(([l,v,d,k])=>`<article style="--accent:${COLORS[k]}"><span>${l}</span><strong>${v}</strong><small>${d}</small></article>`).join('');
  const nf=results.models.neural_ode.forecast_metrics,lf=results.models.lstm.forecast_metrics;
  document.getElementById('findings').innerHTML=`<p><strong>Buy-and-hold did better.</strong> It returned ${percent(passive.total_return)}, compared with ${percent(ode.total_return)} for the Neural ODE. The difference was ${(Math.abs(gap)*100).toFixed(1)} percentage points.</p><p><strong>The Neural ODE was in the market less often.</strong> Average gross exposure was ${percent(ode.gross_exposure,0)}. Its ${percent(Math.abs(ode.max_drawdown))} maximum drawdown should be read in that context.</p><p><strong>The LSTM made no test trades.</strong> Its 2022 validation threshold was ${number(results.models.lstm.threshold*100,2)}%. Although it got the return sign right on ${percent(lf.directional_accuracy)} of test days, its forecasts never crossed that threshold. Neural ODE directional accuracy was ${percent(nf.directional_accuracy)}.</p>`;
  document.getElementById('trainPeriod').textContent=meta.train_period; document.getElementById('validationPeriod').textContent=meta.validation_period; document.getElementById('testPeriod').textContent=meta.test_period; document.getElementById('dataRange').textContent=`${meta.data_snapshot_start} through ${meta.data_snapshot_end}`; document.getElementById('generatedAt').textContent=`Generated ${new Date(meta.generated_at).toLocaleString()}`;
}
function attachTabs(){ document.querySelectorAll('.tab').forEach(b=>b.addEventListener('click',()=>{activeView=b.dataset.view;document.querySelectorAll('.tab').forEach(t=>{const a=t===b;t.classList.toggle('active',a);t.setAttribute('aria-selected',String(a))});renderCharts()})); }
async function boot(){try{const response=await fetch('results.json',{cache:'no-store'});if(!response.ok)throw new Error(`results.json returned ${response.status}`);results=await response.json();renderSummary();renderMetrics();renderCharts();attachTabs()}catch(error){const banner=document.getElementById('errorBanner');banner.hidden=false;banner.textContent=`Unable to load the backtest artifact: ${error.message}. Serve this directory over HTTP instead of opening index.html directly.`;document.getElementById('verdictValue').textContent='Data unavailable';}}
boot();
