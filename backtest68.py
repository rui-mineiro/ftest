import yfinance as yf
import vectorbt as vbt
import pandas as pd
import numpy as np

# Download data
symbol = "SPY"
data = yf.download(symbol, start="2015-01-01", end="2025-01-01", auto_adjust=True)
price = pd.DataFrame(data["Close"])
price.columns = [symbol]

# First trading day of each month
#  https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
buy_dates = price.resample('MS').first().dropna().index
entries = pd.DataFrame(price.index.isin(buy_dates), index=price.index, columns=[symbol])
size = pd.DataFrame(1.0, index=price.index, columns=[symbol])

# Backtest
pf = vbt.Portfolio.from_signals(
    close=price,
    entries=entries,
    exits=None,
    size=size,
    freq='1D',
    accumulate=True,
    init_cash=0
)

# Plot and display
# pf.plots_defaults['subplots']
# print(pf.plots_defaults['subplots'])
#
# print(list(pf.plot.__self__.subplots.keys()))
#

pf.plot(column=symbol,subplots=['orders', 'trades', 'trade_pnl', 'asset_flow', 'cash_flow', 'assets', 'cash', 'asset_value', 'value', 'cum_returns', 'drawdowns', 'underwater', 'gross_exposure', 'net_exposure']).show()

