import yfinance as yf
import pandas as pd
import statsmodels.api as sm

# Inputs
stock = "AAPL"
proxies = {
    "S&P500": "^GSPC",
    "MSCI_World": "URTH",     # iShares MSCI World ETF
    "NASDAQ": "^IXIC",
    "Russell1000": "^RUI"     # if unavailable, try IWB (ETF) as a proxy
}

# Download adjusted closes
tickers = [stock] + list(proxies.values())
px = yf.download(tickers, start="2020-01-01", end="2025-01-01",
                 auto_adjust=False, progress=False)["Adj Close"]

# Returns aligned across all series
rets = px.ffill().pct_change().dropna(how="any")

def beta_cov(stock_ret: pd.Series, mkt_ret: pd.Series) -> float:
    return stock_ret.cov(mkt_ret) / mkt_ret.var()

def beta_reg(stock_ret: pd.Series, mkt_ret: pd.Series) -> float:
    X = sm.add_constant(mkt_ret)
    model = sm.OLS(stock_ret, X).fit()
    return float(model.params.iloc[1])  # slope

rows = []
for name, mkt_tkr in proxies.items():
    s = rets[stock]
    m = rets[mkt_tkr]
    rows.append({
        "Proxy": name,
        "CovBeta": beta_cov(s, m),
        "RegBeta": beta_reg(s, m)
    })

df_betas = pd.DataFrame(rows).set_index("Proxy").sort_index()
print(df_betas)
