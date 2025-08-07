import vectorbt as vbt
import numpy as np
import pandas as pd
import itertools

# Download price data
price = vbt.YFData.download('SPPW.DE').get('Close')

# Parameter grid
# fast_windows = [5, 10, 15]
# slow_windows = [20, 30, 50]
# rsi_windows = [10, 14, 21]

fast_windows = np.arange(3, 30, 1)
slow_windows = np.arange(4, 60, 1)
rsi_windows =  np.arange(20, 30, 1) 

# Generate all valid parameter combinations where fast < slow
valid_combinations = [(f, s, r) for f, s, r in itertools.product(fast_windows, slow_windows, rsi_windows) if f < s]
num_combos = len(valid_combinations)
print(f"Number of valid combinations: {num_combos}")

# Extract individual parameter lists
fast_list, slow_list, rsi_list = zip(*valid_combinations)

# Compute all indicators at once, each combination gets its own column
fast_ma = vbt.MA.run(price, window=fast_list).ma
slow_ma = vbt.MA.run(price, window=slow_list).ma
rsi_val = vbt.RSI.run(price, window=rsi_list).rsi

# Ensure all have the same column labels
column_labels = [f"fast={f}_slow={s}_rsi={r}" for f, s, r in valid_combinations]
fast_ma.columns = column_labels
slow_ma.columns = column_labels
rsi_val.columns = column_labels

# Compute entry and exit signals
entries = (fast_ma > slow_ma) & (rsi_val > 50)
exits = (slow_ma > fast_ma) & (rsi_val < 50)

# Broadcast price to match shape
price_broadcasted = price.vbt.tile(len(valid_combinations))
price_broadcasted.columns = column_labels

# Run backtest
pf = vbt.Portfolio.from_signals(
    price_broadcasted,
    entries,
    exits,
    init_cash=10000
)

# Analyze performance
total_profits = pf.total_profit()
best_label = total_profits.idxmax()
best_profit = total_profits.max()

# Output
print("Best Parameters:")
print(f"  {best_label}")
print(f"Total Profit: ${best_profit:.2f}")
