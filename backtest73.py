import vectorbt as vbt
import numpy as np
import pandas as pd
from itertools import product
import numba
import os

os.environ["NUMBA_NUM_THREADS"] = "8"
print("Numba threads:", numba.get_num_threads())
numba.set_num_threads(8)

price = vbt.YFData.download('SPPW.DE').get('Close')

# fast_windows = np.array([5, 10, 15])
# slow_windows = np.array([20, 30, 40, 50])
# rsi_windows = np.array([10, 15, 20, 25])
fast_windows = np.arange(3, 5, 1)
slow_windows = np.arange(4, 10, 1)
rsi_windows =  np.arange(5, 10, 1) 

# Valid combinations
valid_combinations = [(f, s, r) for f in fast_windows for s in slow_windows for r in rsi_windows if f < s]
num_combos = len(valid_combinations)
print(f"Number of valid combinations: {num_combos}")


# Storage for all entry/exit signals

entries_list = []
exits_list = []
labels = []

for fast, slow, rsi_win in valid_combinations:
    fast_ma = vbt.MA.run(price, window=fast).ma
    slow_ma = vbt.MA.run(price, window=slow).ma
    rsi_val = vbt.RSI.run(price, window=rsi_win).rsi

    entries = (fast_ma > slow_ma) & (rsi_val > 50)
    exits = (slow_ma > fast_ma) & (rsi_val < 50)

    entries_list.append(entries)
    exits_list.append(exits)
    labels.append(f"fast={fast}_slow={slow}_rsi={rsi_win}")

# Create DataFrames with columns = strategy labels
entries_df = pd.DataFrame(entries_list).T
exits_df = pd.DataFrame(exits_list).T
entries_df.columns = labels
exits_df.columns = labels

# Make sure price Series has no name or unique name
price_clean = price.copy()
price_clean.name = None  # or 'Price'

pf = vbt.Portfolio.from_signals(price_clean, entries_df, exits_df, init_cash=10000)

total_profits = pf.total_profit()

best_label = total_profits.idxmax()
best_profit = total_profits[best_label]

print(f"Best strategy: {best_label}")
print(f"Total Profit: ${best_profit:.2f}")

# If you want to parse the label back to params:
fast_str, slow_str, rsi_str = best_label.split('_')
best_fast = int(fast_str.split('=')[1])
best_slow = int(slow_str.split('=')[1])
best_rsi = int(rsi_str.split('=')[1])

print(f"Parsed params -> fast_ma: {best_fast}, slow_ma: {best_slow}, rsi_window: {best_rsi}")
