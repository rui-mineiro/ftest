# pip install yfinance pandas numpy scipy
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

# --- Inputs ---
TICKER = "SPPW.DE"          # stock  SPPW.DE   DFEN.DE
BENCH  = "^GSPC"         # market (S&P 500). For Europe, e.g., "^STOXX50E"
START  = "2018-01-01"
END    = None            # None = today
RF_ANN = 0.02            # annual risk-free assumption (edit to your market)
CONF   = 0.95            # VaR/CVaR confidence
TRADING_DAYS = 252

# --- Download adjusted prices ---
px = yf.download([TICKER, BENCH], start=START, end=END, auto_adjust=True, progress=False)["Close"].dropna()
px.columns = ["S", "M"]  # stock, market

# --- Returns ---
ret = px.pct_change().dropna()              # simple daily returns
rf_daily = (1 + RF_ANN)**(1/TRADING_DAYS) - 1
excess_s = ret["S"] - rf_daily
excess_m = ret["M"] - rf_daily

# --- Volatility (std) and variance ---
vol_daily = ret["S"].std()
vol_ann   = vol_daily * np.sqrt(TRADING_DAYS)
var_daily = vol_daily**2
var_ann   = vol_ann**2

# --- Beta (covariance method) ---
beta = np.cov(excess_s, excess_m, ddof=1)[0,1] / np.var(excess_m, ddof=1)

# --- Sharpe (annualized) ---
mean_excess_daily = excess_s.mean()
sharpe = (mean_excess_daily * TRADING_DAYS) / vol_ann if vol_ann != 0 else np.nan

# --- Drawdown and Max Drawdown ---
cum = (1 + ret["S"]).cumprod()
peak = cum.cummax()
dd = cum/peak - 1.0
max_dd = dd.min()

# --- Historical VaR / CVaR (daily) ---
alpha = 1 - CONF
VaR_hist_daily = -np.quantile(excess_s, alpha)             # positive number = loss threshold
CVaR_hist_daily = -excess_s[excess_s <= np.quantile(excess_s, alpha)].mean()

# --- Parametric Normal VaR / CVaR (daily) ---
mu_d, sd_d = excess_s.mean(), excess_s.std()
z = norm.ppf(alpha)
VaR_param_daily = -(mu_d + sd_d * z)
CVaR_param_daily = -(mu_d - sd_d * norm.pdf(z)/alpha)

# --- Annualize VaR/CVaR with sqrt(time) (approx; ignores drift and fat tails) ---
VaR_hist_ann   = VaR_hist_daily * np.sqrt(TRADING_DAYS)
CVaR_hist_ann  = CVaR_hist_daily * np.sqrt(TRADING_DAYS)
VaR_param_ann  = VaR_param_daily * np.sqrt(TRADING_DAYS)
CVaR_param_ann = CVaR_param_daily * np.sqrt(TRADING_DAYS)

# --- Output ---
out = pd.Series({
    "Obs (days)": len(ret),
    "Mean daily return": ret["S"].mean(),
    "Daily vol": vol_daily,
    "Annual vol": vol_ann,
    "Daily variance": var_daily,
    "Annual variance": var_ann,
    "Beta vs market": beta,
    "Sharpe (ann)": sharpe,
    "Max drawdown": max_dd,
    f"Hist VaR {int(CONF*100)}% (daily)": VaR_hist_daily,
    f"Hist CVaR {int(CONF*100)}% (daily)": CVaR_hist_daily,
    f"Param VaR {int(CONF*100)}% (daily)": VaR_param_daily,
    f"Param CVaR {int(CONF*100)}% (daily)": CVaR_param_daily,
    f"Hist VaR {int(CONF*100)}% (ann≈√t)": VaR_hist_ann,
    f"Hist CVaR {int(CONF*100)}% (ann≈√t)": CVaR_hist_ann,
    f"Param VaR {int(CONF*100)}% (ann≈√t)": VaR_param_ann,
    f"Param CVaR {int(CONF*100)}% (ann≈√t)": CVaR_param_ann,
})
pd.set_option("display.float_format", lambda x: f"{x:,.6f}")
print(out)



