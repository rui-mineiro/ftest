import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from statsmodels import regression
import statsmodels.api as sm
import os

os.environ["QT_QPA_PLATFORM"] = "xcb"


# Define time period and assets
start_date = '2020-01-01'
end_date = '2023-12-31'

# Market portfolio (usually S&P 500) and risk-free rate (3-month T-bill)
market_symbol = '^GSPC'
risk_free_symbol = '^IRX'  # 3-month Treasury bill

# Portfolio assets (example with tech stocks)
portfolio_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']

# Download data
market_data = yf.download(market_symbol, start_date, end_date,auto_adjust=False)['Adj Close']
risk_free_data = yf.download(risk_free_symbol, start_date, end_date,auto_adjust=False)['Adj Close']
portfolio_data = yf.download(portfolio_symbols, start_date, end_date,auto_adjust=False)['Adj Close']

# Convert to daily returns
market_returns = market_data.pct_change().dropna()
risk_free_returns = risk_free_data.pct_change().dropna() / 100  # Convert from percentage
portfolio_returns = portfolio_data.pct_change().dropna()

# Align dates (since different assets might have slightly different trading days)
common_dates = market_returns.index.intersection(portfolio_returns.index)
risk_free_returns = risk_free_returns.loc[common_dates]
market_returns = market_returns.loc[common_dates]
portfolio_returns = portfolio_returns.loc[common_dates]

# Calculate excess returns
# excess_market_returns = market_returns - risk_free_returns.reindex(market_returns.index, method='ffill')
# excess_portfolio_returns = portfolio_returns.sub(risk_free_returns.reindex(portfolio_returns.index, method='ffill'), axis=0)

excess_market_returns    = market_returns    - risk_free_returns.values
excess_portfolio_returns = portfolio_returns - risk_free_returns.values



def estimate_capm(asset_returns, market_returns):
    """Estimate CAPM parameters for a single asset"""
    X = sm.add_constant(market_returns)  # Add constant for alpha
    model = sm.OLS(asset_returns, X).fit()
    alpha = model.params.iloc[0]
    beta = model.params.iloc[1]
    return alpha, beta

# Estimate for each asset in portfolio
capm_params = {}
for asset in portfolio_symbols:
    alpha, beta = estimate_capm(excess_portfolio_returns[asset], excess_market_returns)
    capm_params[asset] = {'Alpha': alpha, 'Beta': beta}
    
capm_df = pd.DataFrame(capm_params).T
print("CAPM Parameters for Portfolio Assets:")
print(capm_df)

def simulate_capm_returns(beta, alpha, market_returns, risk_free_returns, days=252):
    """Simulate returns based on CAPM"""
    simulated_returns = risk_free_returns.values + alpha + beta * (market_returns - risk_free_returns.values)
    return simulated_returns

# Simulate for each asset
simulated_returns = pd.DataFrame()
for asset in portfolio_symbols:
    beta = capm_params[asset]['Beta']
    alpha = capm_params[asset]['Alpha']
    simulated_returns[asset] = simulate_capm_returns(
        beta, alpha, market_returns, risk_free_returns.reindex(market_returns.index, method='ffill')
    )

# Calculate portfolio returns (equal-weighted in this example)
weights = np.array([0.25, 0.25, 0.25, 0.25])  # Equal weights
portfolio_simulated_returns = simulated_returns.dot(weights)


# Calculate actual portfolio returns
actual_portfolio_returns = portfolio_returns.dot(weights)

# Plot comparison
plt.figure(figsize=(12, 6))
plt.plot(actual_portfolio_returns.cumsum(), label='Actual Portfolio Returns')
plt.plot(portfolio_simulated_returns.cumsum(), label='CAPM-Simulated Returns')
plt.title('Actual vs CAPM-Simulated Portfolio Returns')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.legend()
plt.grid(True)
plt.show()



def calculate_metrics(returns, risk_free=0):
    """Calculate performance metrics"""
    excess_returns = returns - risk_free
    sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
    annualized_return = (1 + returns.mean())**252 - 1
    return sharpe_ratio, annualized_return

# For actual portfolio
sharpe_actual, ann_return_actual = calculate_metrics(actual_portfolio_returns)

# For simulated portfolio
sharpe_sim, ann_return_sim = calculate_metrics(portfolio_simulated_returns)

print(f"Actual Portfolio - Sharpe: {sharpe_actual:.2f}, Annual Return: {ann_return_actual:.2%}")
print(f"Simulated Portfolio - Sharpe: {sharpe_sim:.2f}, Annual Return: {ann_return_sim:.2%}")


def monte_carlo_capm_simulation(betas, alphas, market_returns, risk_free, n_simulations=1000, days=252):
    """Monte Carlo simulation using CAPM"""
    simulated_portfolios = []
    
    for _ in range(n_simulations):
        # Randomly sample market returns (with replacement)
        sampled_market = np.random.choice(market_returns.squeeze(), size=days, replace=True)
        sampled_risk_free = np.random.choice(risk_free.squeeze(), size=days, replace=True)
        
        # Simulate each asset
        asset_returns = []
        for beta, alpha in zip(betas, alphas):
            asset_return = sampled_risk_free + alpha + beta * (sampled_market - sampled_risk_free)
            asset_returns.append(asset_return)
        
        # Calculate portfolio return (equal-weighted)
        portfolio_return = np.mean(asset_returns, axis=0)
        simulated_portfolios.append(np.prod(1 + portfolio_return) - 1)  # Total return
    
    return np.array(simulated_portfolios)

# Get betas and alphas
betas = capm_df['Beta'].values
alphas = capm_df['Alpha'].values

# Run simulation
mc_results = monte_carlo_capm_simulation(betas, alphas, market_returns.values, 
                                        risk_free_returns.reindex(market_returns.index, method='ffill').values,
                                        n_simulations=10000)

# Plot results
plt.figure(figsize=(10, 6))
plt.hist(mc_results, bins=50, edgecolor='k', alpha=0.65)
plt.title('Monte Carlo Simulation of Portfolio Returns Using CAPM')
plt.xlabel('Portfolio Return')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Calculate statistics
print(f"Mean simulated return: {np.mean(mc_results):.2%}")
print(f"5th percentile return: {np.percentile(mc_results, 5):.2%}")
print(f"95th percentile return: {np.percentile(mc_results, 95):.2%}")

