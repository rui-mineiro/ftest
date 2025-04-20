import yfinance as yf
import ruptures as rpt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
import os

os.environ["QT_QPA_PLATFORM"] = "xcb"

# Step 1: Download S&P 500 data
data = yf.download("^GSPC", start="2015-01-01", end="2023-01-01",auto_adjust=False)
data = data['Adj Close']
returns = data.pct_change().dropna().values  # Use daily returns

# Step 2: Changepoint detection (using 'rbf' cost function)
# model = rpt.Window(width=15, model="l2").fit(returns)
model = rpt.KernelCPD(kernel="linear", min_size=5).fit(returns)
breaks = model.predict(n_bkps=4)  # Penalty controls number of segments

# Step 3: Plot with breakpoints
rpt.display(returns,breaks)
plt.title("Detected Regimes in S&P 500 Returns")
plt.show()

# Step 4: Estimate PDF for each regime
segments = np.split(returns, breaks[:-1])
for i, seg in enumerate(segments):
    plt.figure()
    sns.histplot(seg, kde=True, stat="density", bins=30, label=f"Regime {i+1}", color='skyblue')
    
    # Fit normal distribution
    mu, std = norm.fit(seg)
    x = np.linspace(seg.min(), seg.max(), 100)
    plt.plot(x, norm.pdf(x, mu, std), 'r--', label=f'N({mu:.4f}, {std:.4f})')
    plt.title(f"PDF Estimate for Regime {i+1}")
    plt.legend()
    plt.xlabel("Daily Return")
    plt.ylabel("Density")
    plt.grid(True)
    plt.show()
