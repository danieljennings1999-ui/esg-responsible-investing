"""
Capstone Analysis — Environmental Investing and Financial Returns
Imperial College Business School Executive Education — Data Analytics Capstone

Research Question: Can investors achieve competitive financial returns by
investing only in environmentally responsible companies?

Data sources:
    1. Refinitiv (LSEG) Workspace export — Environmental Pillar ESG Score,
       Social/Governance Pillar Scores, Controversies Score, GICS Industry,
       Market Cap for 1,496 US companies (GridExport_August_23_2026_11_6_48.csv)
    2. Refinitiv (LSEG) Workspace export — Emissions Score for 1,495 US
       companies (GridExport_August_29_2026_8_47_34.xlsx)
    3. Yahoo Finance via yfinance — daily closing prices, Jan 2022 to Aug 2025,
       for 1,531 US tickers (sp500_prices_expanded.csv)
    4. Yahoo Finance via yfinance — daily closing price for ESGU (iShares ESG
       Aware MSCI USA ETF), used as a real-world benchmark (esgu_prices.csv)

Outputs:
    7 charts (PNG) + full console log of all statistics reported in the
    accompanying report.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
from pypfopt import EfficientFrontier, expected_returns, risk_models

np.random.seed(42)  # reproducibility for Monte Carlo simulation

COLORS = {'Green': '#2E7D32', 'Brown': '#8D5524', 'Market': '#1565C0'}


# =========================================================================
# STEP 1 — LOAD & CLEAN ESG + EMISSIONS DATA
# =========================================================================
print("=" * 70)
print("STEP 1: Loading and cleaning ESG data")
print("=" * 70)

esg = pd.read_csv("GridExport_August_23_2026_11_6_48.csv")
esg.columns = ['Identifier', 'Company', 'EnvScore', 'Country', 'RIC',
               'Industry', 'MarketCap', 'SocialScore', 'GovScore', 'Controversies']
esg = esg.dropna(subset=['EnvScore']).copy()
print(f"Companies with a valid Environmental Pillar Score: {len(esg)}")

esg['Ticker'] = esg['RIC'].str.split('.').str[0]
esg['Exchange'] = esg['RIC'].str.split('.').str[1]
esg['MarketCapNum'] = esg['MarketCap'].str.replace(',', '').astype(float)

# Some tickers collide after stripping the exchange suffix (e.g. BYD.N, a US
# stock, vs BYD.TO, a Canadian stock). Keep the US-listed entity in each case.
us_exch = {'N', 'OQ', 'A', 'K'}
esg = esg.sort_values('Exchange', key=lambda s: ~s.isin(us_exch))
esg = esg.drop_duplicates(subset='Ticker', keep='first')
print(f"Companies after resolving ticker collisions: {len(esg)}")

# Merge in the separately-exported Emissions Score (0-100, higher = better
# emissions management), matched by Identifier before any further filtering.
emissions = pd.read_excel("GridExport_August_29_2026_8_47_34.xlsx")
emissions.columns = ['Identifier', 'Company', 'EnvScoreCheck', 'Country', 'EmissionsScore']
emissions = emissions[['Identifier', 'EmissionsScore']]
esg = esg.merge(emissions, on='Identifier', how='left')
print(f"Companies with an Emissions Score: {esg['EmissionsScore'].notna().sum()} of {len(esg)}")


# =========================================================================
# STEP 2 — LOAD PRICE DATA & BUILD THE INVESTABLE UNIVERSE
# =========================================================================
print("\n" + "=" * 70)
print("STEP 2: Building the investable universe")
print("=" * 70)

prices = pd.read_csv("sp500_prices_expanded.csv", parse_dates=['Date']).set_index('Date')

fully_empty = prices.columns[prices.isna().sum() == len(prices)].tolist()
prices = prices.drop(columns=fully_empty)
print(f"Dropped {len(fully_empty)} price columns with no data at all")

esgu = pd.read_csv("esgu_prices.csv", parse_dates=['Date']).set_index('Date')

# Restrict to companies with usable price data BEFORE computing any
# Green/Brown percentile thresholds. Doing this the other way around
# (thresholds first, then filtering to available prices) was tried and
# produces a severely unbalanced sample, because low-scoring companies in
# this dataset skew toward smaller, less liquid firms that are more likely
# to be missing from the price file.
investable_tickers = set(prices.columns) & set(esg['Ticker'])
investable = esg[esg['Ticker'].isin(investable_tickers)].copy()
investable = investable[investable['MarketCapNum'] >= 1e8]  # drop sub-$100m companies
print(f"Investable universe after price match + market cap floor: {len(investable)}")

# Forward-fill short data gaps (stray non-trading days, feed hiccups) rather
# than treating every missing day as fatal. Then exclude any ticker that
# still has more than 5 missing days in the test window even after
# forward-filling — this correctly catches companies that were delisted or
# went bankrupt during the test period, while not needlessly discarding
# companies with only a handful of minor data gaps.
prices = prices.ffill()
test_range = prices.loc['2024-01-01':'2025-08-21']
max_gap = test_range.isna().sum()
tickers_with_test_data = set(max_gap[max_gap <= 5].index)
investable = investable[investable['Ticker'].isin(tickers_with_test_data)]
print(f"Investable universe after excluding delisted/incomplete tickers: {len(investable)}")


# =========================================================================
# STEP 3 — DEFINE GREEN / BROWN / MARKET PORTFOLIOS
# =========================================================================
print("\n" + "=" * 70)
print("STEP 3: Defining Green, Brown and Market portfolios")
print("=" * 70)

# Top/bottom 30% (tercile) of the investable universe by Environmental
# Pillar Score. Green additionally excludes high-controversy companies
# (Controversies >= 3) as a "greenwashing" screen.
green_thresh = investable['EnvScore'].quantile(0.70)
brown_thresh = investable['EnvScore'].quantile(0.30)

green_tickers = investable[(investable['EnvScore'] >= green_thresh) &
                            (investable['Controversies'] < 3)]['Ticker'].tolist()
brown_tickers = investable[investable['EnvScore'] <= brown_thresh]['Ticker'].tolist()
market_tickers = investable['Ticker'].tolist()

print(f"Green threshold: EnvScore >= {green_thresh:.2f} -> {len(green_tickers)} companies")
print(f"Brown threshold: EnvScore <= {brown_thresh:.2f} -> {len(brown_tickers)} companies")
print(f"Market: all investable companies -> {len(market_tickers)} companies")


# =========================================================================
# STEP 4 — BUILD PORTFOLIOS (MAX-SHARPE, TRAINED ON 2022-2023)
# =========================================================================
print("\n" + "=" * 70)
print("STEP 4: Building max-Sharpe portfolios (trained on 2022-2023)")
print("=" * 70)

train = prices.loc['2022-01-01':'2023-12-31']
test = prices.loc['2024-01-01':'2025-08-21']


def build_portfolio(tickers, price_df, max_weight=0.05):
    """Max-Sharpe portfolio via PyPortfolioOpt, capped at max_weight per holding.

    Uses CAPM-based expected returns (more robust to short-window noise than
    a raw historical mean — see report Methods/Limitations for the failure
    case this avoids) and Ledoit-Wolf covariance shrinkage (needed because
    the number of candidate assets is comparable to or exceeds the number of
    training observations, which makes the raw sample covariance matrix
    unstable).
    """
    sub = price_df[tickers].dropna(axis=1)
    sub = sub.loc[:, (sub > 0).all()]  # drop tickers with any non-positive price
    mu = expected_returns.capm_return(sub)
    S = risk_models.CovarianceShrinkage(sub).ledoit_wolf()
    ef = EfficientFrontier(mu, S, weight_bounds=(0, max_weight))
    ef.max_sharpe()
    weights = ef.clean_weights()
    perf = ef.portfolio_performance()
    return weights, perf


results = {}
for name, tickers in [('Green', green_tickers), ('Brown', brown_tickers), ('Market', market_tickers)]:
    weights, perf = build_portfolio(tickers, train)
    results[name] = weights
    ret, vol, sharpe = perf
    print(f"{name}: training return {ret*100:.1f}%, volatility {vol*100:.1f}%, Sharpe {sharpe:.2f}")


def build_equal_weight(tickers, price_df):
    """Equal-weight portfolio across all valid candidate tickers, as an
    alternative construction method with no optimization at all."""
    sub = price_df[tickers].dropna(axis=1)
    sub = sub.loc[:, (sub > 0).all()]
    n = len(sub.columns)
    return {ticker: 1 / n for ticker in sub.columns}


equal_results = {}
for name, tickers in [('Green', green_tickers), ('Brown', brown_tickers), ('Market', market_tickers)]:
    weights = build_equal_weight(tickers, train)
    equal_results[name] = weights
    print(f"{name} equal-weight: {len(weights)} holdings, each at {100/len(weights):.2f}%")


def backtest(weights, price_df, start_value=10000):
    tickers = [t for t, w in weights.items() if w > 0]
    w = np.array([weights[t] for t in tickers])
    sub = price_df[tickers]
    normed = sub / sub.iloc[0]
    growth = (normed * w).sum(axis=1)
    return growth * start_value


# =========================================================================
# STEP 5 — BACKTEST ON 2024-2025 (OUT-OF-SAMPLE)
# =========================================================================
print("\n" + "=" * 70)
print("STEP 5: Backtesting on 2024-2025 (out-of-sample)")
print("=" * 70)

backtest_values = {}
for name in ['Green', 'Brown', 'Market']:
    pv = backtest(results[name], test)
    backtest_values[name] = pv
    total_ret = (pv.iloc[-1] / 10000 - 1) * 100
    print(f"{name}: £10,000 -> £{pv.iloc[-1]:,.0f} ({total_ret:+.1f}%)")

esgu_test = esgu.loc['2024-01-01':'2025-08-21']
esgu_growth = (esgu_test / esgu_test.iloc[0]) * 10000
esgu_final = esgu_growth.iloc[-1].values[0]
esgu_return = (esgu_final / 10000 - 1) * 100
print(f"ESGU (real-world benchmark): £10,000 -> £{esgu_final:,.0f} ({esgu_return:+.1f}%)")

print("\nRobustness check — equal-weight construction (no optimizer):")
for name in ['Green', 'Brown', 'Market']:
    pv = backtest(equal_results[name], test)
    total_ret = (pv.iloc[-1] / 10000 - 1) * 100
    print(f"{name} equal-weight: £10,000 -> £{pv.iloc[-1]:,.0f} ({total_ret:+.1f}%)")


def max_drawdown(portfolio_values):
    running_max = portfolio_values.cummax()
    drawdown = (portfolio_values - running_max) / running_max
    return drawdown.min() * 100


print("\nMax drawdown (worst peak-to-trough fall during the test period):")
for name in ['Green', 'Brown', 'Market']:
    dd = max_drawdown(backtest_values[name])
    print(f"{name}: {dd:.1f}%")


# =========================================================================
# STEP 6 — ESG REGRESSION (SIMPLE + MULTIPLE + SIZE-BUCKETED)
# =========================================================================
print("\n" + "=" * 70)
print("STEP 6: Regression analysis")
print("=" * 70)

returns = {}
for t in investable['Ticker']:
    if t in test.columns:
        s = test[t].dropna()
        if len(s) > 1:
            returns[t] = (s.iloc[-1] / s.iloc[0] - 1) * 100

ret_df = pd.DataFrame(list(returns.items()), columns=['Ticker', 'Return2425'])
merged = investable.merge(ret_df, on='Ticker')

slope, intercept, r, p, se = stats.linregress(merged['EnvScore'], merged['Return2425'])
print(f"Simple regression (EnvScore -> Return): slope={slope:.2f}, "
      f"R²={r**2:.4f}, p={p:.3f}, n={len(merged)}")

# Multiple regression: does EnvScore still predict returns once market cap
# (a known confound - see descriptive stats) is controlled for?
X = merged[['EnvScore', 'MarketCapNum']].copy()
X['MarketCapNum'] = X['MarketCapNum'] / 1e9  # rescale to billions for readable coefficients
X = sm.add_constant(X)
y = merged['Return2425']
model = sm.OLS(y, X).fit()
print("\nMultiple regression (EnvScore + MarketCap -> Return):")
print(model.summary())

# Size-bucketed regression: reveals Simpson's Paradox — a negative EnvScore
# effect appears within every size tercile, despite no effect in the pooled
# (unconditional) regression above.
merged['SizeBucket'] = pd.qcut(merged['MarketCapNum'], q=3, labels=['Small', 'Mid', 'Large'])
print(f"\n{merged['SizeBucket'].value_counts()}")

print("\nRegression within each size bucket:")
for bucket in ['Small', 'Mid', 'Large']:
    sub = merged[merged['SizeBucket'] == bucket]
    b_slope, b_intercept, b_r, b_p, b_se = stats.linregress(sub['EnvScore'], sub['Return2425'])
    print(f"{bucket} (n={len(sub)}): slope={b_slope:.2f}, R²={b_r**2:.4f}, p={b_p:.3f}")


# =========================================================================
# STEP 7 — EMISSIONS COMPARISON
# =========================================================================
print("\n" + "=" * 70)
print("STEP 7: Environmental impact comparison")
print("=" * 70)

green_df_em = investable[(investable['EnvScore'] >= green_thresh) & (investable['Controversies'] < 3)]
brown_df_em = investable[investable['EnvScore'] <= brown_thresh]

print(f"Green candidate pool avg Emissions Score: {green_df_em['EmissionsScore'].mean():.1f}")
print(f"Brown candidate pool avg Emissions Score: {brown_df_em['EmissionsScore'].mean():.1f}")
print(f"Market candidate pool avg Emissions Score: {investable['EmissionsScore'].mean():.1f}")


def portfolio_weighted_emissions(weights, investable_df):
    """Emissions Score of the ACTUAL invested portfolio (weighted by real
    holdings), not just the average of the full candidate pool."""
    total_weight = 0
    weighted_sum = 0
    for ticker, weight in weights.items():
        if weight > 0:
            match = investable_df[investable_df['Ticker'] == ticker]
            if len(match) > 0 and not pd.isna(match['EmissionsScore'].values[0]):
                weighted_sum += weight * match['EmissionsScore'].values[0]
                total_weight += weight
    return weighted_sum / total_weight if total_weight > 0 else None


print("\nActual invested portfolio Emissions Score (weighted by real holdings):")
for name in ['Green', 'Brown', 'Market']:
    score = portfolio_weighted_emissions(results[name], investable)
    print(f"{name}: {score:.1f}")


# =========================================================================
# STEP 8 — ROBUSTNESS: ROLLING TIME WINDOWS
# =========================================================================
print("\n" + "=" * 70)
print("STEP 8: Robustness check — rolling time windows")
print("=" * 70)

windows = [
    ('2022-01-01', '2022-12-31', '2023-01-01', '2023-12-31'),
    ('2023-01-01', '2023-12-31', '2024-01-01', '2024-12-31'),
    ('2024-01-01', '2024-12-31', '2025-01-01', '2025-08-21'),
]

for train_start, train_end, test_start, test_end in windows:
    print(f"\nWindow: train {train_start} to {train_end}, test {test_start} to {test_end}")
    window_train = prices.loc[train_start:train_end]
    window_test = prices.loc[test_start:test_end]

    window_results = {}
    for name, tickers in [('Green', green_tickers), ('Brown', brown_tickers), ('Market', market_tickers)]:
        try:
            weights, perf = build_portfolio(tickers, window_train)
            window_results[name] = weights
        except Exception as e:
            print(f"  {name} failed to build: {e}")
            continue

    for name in ['Green', 'Brown', 'Market']:
        if name in window_results:
            pv = backtest(window_results[name], window_test)
            total_ret = (pv.iloc[-1] / 10000 - 1) * 100
            print(f"  {name}: £10,000 -> £{pv.iloc[-1]:,.0f} ({total_ret:+.1f}%)")


# =========================================================================
# STEP 9 — MONTE CARLO SIMULATION (3 YEARS FORWARD)
# =========================================================================
print("\n" + "=" * 70)
print("STEP 9: Monte Carlo simulation (1,000 sims, 3 years forward)")
print("=" * 70)

full_period = prices.loc['2022-01-01':'2025-08-21']
n_sims, n_days, start_value = 1000, 252 * 3, 15000


def portfolio_daily_returns(weights, price_df):
    tickers = [t for t, w in weights.items() if w > 0]
    w = np.array([weights[t] for t in tickers])
    sub = price_df[tickers].dropna()
    daily_ret = sub.pct_change().dropna()
    return (daily_ret * w).sum(axis=1)


mc_paths = {}
for name in ['Green', 'Brown', 'Market']:
    daily_rets = portfolio_daily_returns(results[name], full_period)
    sim_finals = np.zeros(n_sims)
    all_paths = np.zeros((n_sims, n_days))
    for i in range(n_sims):
        sampled = np.random.choice(daily_rets.values, size=n_days, replace=True)
        path = start_value * np.cumprod(1 + sampled)
        all_paths[i] = path
        sim_finals[i] = path[-1]
    mc_paths[name] = all_paths
    print(f"{name}: median £{np.median(sim_finals):,.0f}, "
          f"worst 10% £{np.percentile(sim_finals, 10):,.0f}, "
          f"best 10% £{np.percentile(sim_finals, 90):,.0f}")


def sortino_ratio(daily_returns, target=0):
    downside = daily_returns[daily_returns < target]
    downside_std = downside.std() * np.sqrt(252)
    annual_return = daily_returns.mean() * 252
    return annual_return / downside_std


print("\nSortino ratio (downside-risk-adjusted return):")
for name in ['Green', 'Brown', 'Market']:
    daily_rets = portfolio_daily_returns(results[name], full_period)
    print(f"{name}: {sortino_ratio(daily_rets):.2f}")

print("\n95% Value-at-Risk (3-year horizon, £15,000 start):")
for name in ['Green', 'Brown', 'Market']:
    paths = mc_paths[name]
    final_values = paths[:, -1]
    var_95 = start_value - np.percentile(final_values, 5)
    print(f"{name}: £{var_95:,.0f} potential loss")


# =========================================================================
# STEP 10 — STRESS TEST (REAL MARKET DIP, NOV 2024 – APR 2025)
# =========================================================================
print("\n" + "=" * 70)
print("STEP 10: Stress test against the real Nov 2024 - Apr 2025 market dip")
print("=" * 70)

market_pv = backtest_values['Market']
stress_window = market_pv.loc['2025-03-01':'2025-05-01']
trough_date = stress_window.idxmin()
peak_before = market_pv.loc[:trough_date].idxmax()
print(f"Peak before dip: {peak_before.date()} at £{market_pv[peak_before]:,.0f}")
print(f"Trough: {trough_date.date()} at £{market_pv[trough_date]:,.0f}")

print(f"\nStress test: {peak_before.date()} (peak) to {trough_date.date()} (trough)")
for name in ['Green', 'Brown', 'Market']:
    pv = backtest_values[name]
    stress_peak_val = pv[peak_before]
    stress_trough_val = pv[trough_date]
    decline = (stress_trough_val / stress_peak_val - 1) * 100
    print(f"{name}: £{stress_peak_val:,.0f} -> £{stress_trough_val:,.0f} ({decline:+.1f}% during stress event)")


# =========================================================================
# STEP 11 — SECTOR ANALYSIS
# =========================================================================
print("\n" + "=" * 70)
print("STEP 11: Sector analysis")
print("=" * 70)

sector_stats = investable.groupby('Industry')['EnvScore'].agg(['mean', 'count'])
sector_stats = sector_stats[sector_stats['count'] >= 3].sort_values('mean', ascending=False)
print("Greenest sectors:\n", sector_stats.head(5))
print("\nDirtiest sectors:\n", sector_stats.tail(5))


# =========================================================================
# STEP 12 — CHARTS
# =========================================================================
print("\n" + "=" * 70)
print("STEP 12: Generating charts")
print("=" * 70)

# Chart 1: Backtest (with ESGU real-world benchmark)
fig, ax = plt.subplots(figsize=(11, 6))
for name in ['Green', 'Brown', 'Market']:
    ax.plot(backtest_values[name].index, backtest_values[name], label=name, color=COLORS[name], linewidth=2)
ax.plot(esgu_growth.index, esgu_growth['ESGU'], label='ESGU (real-world ESG ETF)',
        color='dimgray', linewidth=1.5, linestyle=':')
ax.axhline(10000, color='gray', linestyle='--', linewidth=0.8)
ax.set_title('Backtest: £10,000 Invested, Jan 2024 – Aug 2025\n'
             'Constructed portfolios vs. a real-world ESG ETF benchmark')
ax.set_ylabel('Portfolio Value (£)')
ax.set_xlabel('Date')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('backtest_v2.png', dpi=150)
plt.close()
print("Saved backtest_v2.png")

# Chart 2: Regression
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(merged['EnvScore'], merged['Return2425'], alpha=0.5, s=25, color='#455A64')
xline = np.linspace(merged['EnvScore'].min(), merged['EnvScore'].max(), 100)
ax.plot(xline, intercept + slope * xline, color='#D32F2F', linewidth=2,
        label=f'slope={slope:.2f}, R²={r**2:.4f}, p={p:.3f}')
ax.set_title(f'Environmental Score vs 2024–2025 Return (n={len(merged)})')
ax.set_xlabel('Environmental Pillar Score')
ax.set_ylabel('Total Return 2024–2025 (%)')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('esg_regression_v2.png', dpi=150)
plt.close()
print("Saved esg_regression_v2.png")

# Chart 3: Monte Carlo
fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), sharey=True)
for ax, name in zip(axes, ['Green', 'Brown', 'Market']):
    paths = mc_paths[name]
    days = np.arange(paths.shape[1])
    p10, p25, p50 = np.percentile(paths, [10, 25, 50], axis=0)
    p75, p90 = np.percentile(paths, [75, 90], axis=0)
    ax.fill_between(days, p10, p90, color=COLORS[name], alpha=0.15, label='10th-90th pct')
    ax.fill_between(days, p25, p75, color=COLORS[name], alpha=0.3, label='25th-75th pct')
    ax.plot(days, p50, color=COLORS[name], linewidth=2, label='Median')
    ax.set_title(f'{name}\nMedian: £{np.median(paths[:, -1]):,.0f}')
    ax.set_xlabel('Trading days forward')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
axes[0].set_ylabel('Portfolio Value (£)')
fig.suptitle('Monte Carlo: 3-Year Forward Projection (1,000 simulations, £15,000 start)')
plt.tight_layout()
plt.savefig('monte_carlo_v2.png', dpi=150)
plt.close()
print("Saved monte_carlo_v2.png")

# Chart 4: Sector analysis
fig, ax = plt.subplots(figsize=(10, 12))
sector_sorted = sector_stats.sort_values('mean')
colors_bar = plt.cm.RdYlGn((sector_sorted['mean'] - sector_sorted['mean'].min()) /
                            (sector_sorted['mean'].max() - sector_sorted['mean'].min()))
ax.barh(sector_sorted.index, sector_sorted['mean'], color=colors_bar)
ax.set_xlabel('Average Environmental Pillar Score')
ax.set_title('Average Environmental Score by GICS Industry\n(sectors with >= 3 companies in investable universe)')
ax.grid(alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('sector_analysis.png', dpi=150)
plt.close()
print("Saved sector_analysis.png")

# Chart 5: Descriptive stats
green_df = investable[(investable['EnvScore'] >= green_thresh) & (investable['Controversies'] < 3)]
brown_df = investable[investable['EnvScore'] <= brown_thresh]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].hist(investable['EnvScore'], bins=25, color='#455A64', alpha=0.7, edgecolor='white')
axes[0].axvline(green_thresh, color=COLORS['Green'], linestyle='--', label=f'Green threshold ({green_thresh:.2f})')
axes[0].axvline(brown_thresh, color=COLORS['Brown'], linestyle='--', label=f'Brown threshold ({brown_thresh:.2f})')
axes[0].set_title('Distribution of Environmental Scores\n(Investable Universe)')
axes[0].set_xlabel('Environmental Pillar Score')
axes[0].set_ylabel('Number of Companies')
axes[0].legend(fontsize=9)
axes[0].grid(alpha=0.3)

box_data = [green_df['MarketCapNum'] / 1e9, brown_df['MarketCapNum'] / 1e9, investable['MarketCapNum'] / 1e9]
bp = axes[1].boxplot(box_data, tick_labels=['Green', 'Brown', 'Market'], patch_artist=True, showfliers=False)
for patch, name in zip(bp['boxes'], ['Green', 'Brown', 'Market']):
    patch.set_facecolor(COLORS[name])
    patch.set_alpha(0.5)
axes[1].set_title('Market Cap by Portfolio\n(log scale; outliers hidden for readability)')
axes[1].set_ylabel('Market Cap ($ billions)')
axes[1].set_yscale('log')
axes[1].grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('descriptive_stats.png', dpi=150)
plt.close()
print("Saved descriptive_stats.png")

# Chart 6: Simpson's Paradox
fig, ax = plt.subplots(figsize=(10, 7))
bucket_colors = {'Small': '#EF6C00', 'Mid': '#6A1B9A', 'Large': '#0277BD'}

for bucket in ['Small', 'Mid', 'Large']:
    sub = merged[merged['SizeBucket'] == bucket]
    ax.scatter(sub['EnvScore'], sub['Return2425'], alpha=0.3, s=20,
               color=bucket_colors[bucket], label=f'{bucket} cap')
    b_slope, b_intercept, b_r, b_p, b_se = stats.linregress(sub['EnvScore'], sub['Return2425'])
    xline_b = np.linspace(sub['EnvScore'].min(), sub['EnvScore'].max(), 100)
    ax.plot(xline_b, b_intercept + b_slope * xline_b, color=bucket_colors[bucket], linewidth=2.5)

xline_all = np.linspace(merged['EnvScore'].min(), merged['EnvScore'].max(), 100)
ax.plot(xline_all, intercept + slope * xline_all, color='black', linewidth=2.5,
        linestyle='--', label='Pooled (all companies)')

ax.set_title("Simpson's Paradox: Environmental Score vs Return\n"
             "Within each size group: negative. Pooled together: flat.")
ax.set_xlabel('Environmental Pillar Score')
ax.set_ylabel('Total Return 2024–2025 (%)')
ax.set_ylim(-100, 300)
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('simpsons_paradox.png', dpi=150)
plt.close()
print("Saved simpsons_paradox.png")

# Chart 7: Emissions comparison
fig, ax = plt.subplots(figsize=(8, 6))
portfolio_names = ['Green', 'Market', 'Brown']
emissions_values = [
    green_df_em['EmissionsScore'].mean(),
    investable['EmissionsScore'].mean(),
    brown_df_em['EmissionsScore'].mean()
]
bar_colors = [COLORS['Green'], COLORS['Market'], COLORS['Brown']]

bars = ax.bar(portfolio_names, emissions_values, color=bar_colors, alpha=0.8, width=0.6)
for bar, val in zip(bars, emissions_values):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 1.5, f'{val:.1f}',
            ha='center', fontsize=13, fontweight='bold')

ax.set_ylabel('Average Emissions Score (0-100, higher = better management)')
ax.set_title('Environmental Impact by Portfolio\n'
             'Green holdings score more than 2x higher than Brown on real emissions management')
ax.set_ylim(0, 100)
ax.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('emissions_comparison.png', dpi=150)
plt.close()
print("Saved emissions_comparison.png")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)