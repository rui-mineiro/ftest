import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ruptures as rpt
import os


CACHE_FILE = "stock_history.csv"
ticker_symbol = "SPY"

# Step 1: Load from cache or fetch from yfinance
if os.path.exists(CACHE_FILE):
    print(f"Loading cached data from {CACHE_FILE}")
    df = pd.read_csv(CACHE_FILE, index_col=0, parse_dates=True)
else:
    print("Fetching data from yfinance...")
    ticker = yf.Ticker(ticker_symbol)
    df = ticker.history(period="max")
    df.to_csv(CACHE_FILE)
    print(f"Data saved to {CACHE_FILE}")


# Step 2: Compute daily returns
df['Return'] = df['Close'].pct_change()
# df['Return'] = df['Close'].values
# df['Return'] = np.log(df['Close'].values) 

# Step 3: Rolling z-score
window = 30  # 30-day window
df['RollingMean'] = df['Return'].rolling(window=8).mean()
df['RollingStd'] = df['Return'].rolling(window=120).std()
df['ZScore'] = (df['Return'] - df['RollingMean']) / df['RollingStd']

# Step 4: Mark significant changes
threshold = 3  # z-score threshold
df['Anomaly'] = df['ZScore'].abs() > threshold

# Step 5: Segment the data
df['SegmentID'] = df['Anomaly'].cumsum()

# Step 6: Plot
plt.figure(figsize=(14, 6))
plt.plot(df.index, df['Close'], label='Close Price')
plt.scatter(df[df['Anomaly']].index, df[df['Anomaly']]['Close'], color='red', label='Significant Change', zorder=5)
plt.title("Tesla Close Price with Z-Score Based Regime Changes")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()