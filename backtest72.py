import vectorbt as vbt
import numpy as np
import pandas as pd
from itertools import product

# Download price data
price = vbt.YFData.download('SPPW.DE').get('Close')  # shape (n,)

# Define parameter ranges
fast_windows = np.arange(3, 20, 1)     # 5, 10, 15
slow_windows = np.arange(10, 60, 1)   # 20, 30, 40, 50
rsi_windows  = np.arange(5, 30, 1)    # 10, 15, 20, 25

# Create all combinations
combinations = list(product(fast_windows, slow_windows, rsi_windows))

# Storage for all entry/exit signals
entries_list = []
exits_list = []

for fast, slow, rsi_win in combinations:
    # Skip invalid combinations
    if fast >= slow:
        entries_list.append(np.full(price.shape, False))
        exits_list.append(np.full(price.shape, False))
        continue

    fast_ma = vbt.MA.run(price, window=fast).ma
    slow_ma = vbt.MA.run(price, window=slow).ma
    rsi_val = vbt.RSI.run(price, window=rsi_win).rsi

    entries = (fast_ma > slow_ma) & (rsi_val > 50)
    exits = (slow_ma > fast_ma) & (rsi_val < 50)

    entries_list.append(entries)
    exits_list.append(exits)

# Stack into DataFrames
entries_df = pd.DataFrame(entries_list).T  # shape (n, combos)
exits_df = pd.DataFrame(exits_list).T

# Backtest
pf = vbt.Portfolio.from_signals(price, entries_df, exits_df, init_cash=10_000)

# Analyze
total_profits = pf.total_profit()
best_idx = total_profits.idxmax()
best_params = combinations[best_idx]
best_profit = total_profits[best_idx]

print("Best Parameters:")
print(f"  fast_ma: {best_params[0]}")
print(f"  slow_ma: {best_params[1]}")
print(f"  rsi_window: {best_params[2]}")
print(f"Total Profit: ${best_profit:.2f}")
