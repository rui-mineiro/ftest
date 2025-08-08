import vectorbt as vbt
import numpy as np
import pandas as pd
import itertools
import plotly.graph_objects as go
from datetime import datetime, timedelta


symbol = 'DFEN.DE' # 'DAVV.DE' # 'DFEN.DE' # 'SPPW.DE'


yesterday = 1

# Download price data
enddate=datetime.today()-timedelta(days=yesterday)
startdate=datetime.today()-timedelta(days=yesterday)-timedelta(days=365*2)
price = vbt.YFData.download(symbol,start=startdate,end=enddate).get('Close')

# Parameter grid

b_fast_windows = np.arange(1, 30, 1)
s_fast_windows = np.arange(1, 30, 1)
b_slow_windows = np.arange(1, 15, 1)
s_slow_windows = np.arange(1, 15, 1)


# Generate all valid parameter combinations where fast < slow
valid_combinations = [(b_f, b_s, s_f, s_s) for b_f, b_s, s_f, s_s in itertools.product(b_fast_windows, b_slow_windows, s_fast_windows, s_slow_windows) if ( s_f < s_s and b_f < b_s) ]
num_combos = len(valid_combinations)
print(f"Number of valid combinations: {num_combos}")

# Extract individual parameter lists
b_fast_list, b_slow_list, s_fast_list, s_slow_list  = zip(*valid_combinations)

# Compute all indicators at once, each combination gets its own column
b_fast_ma = vbt.MA.run(price, window=b_fast_list).ma
b_slow_ma = vbt.MA.run(price, window=b_slow_list).ma
s_fast_ma = vbt.MA.run(price, window=s_fast_list).ma
s_slow_ma = vbt.MA.run(price, window=s_slow_list).ma


# Ensure all have the same column labels
column_labels = [f"b_fast={b_f}_slow={b_s}_s_fast={s_f}_slow={s_s}" for b_f, b_s, s_f , s_s in valid_combinations]
b_fast_ma.columns = column_labels
b_slow_ma.columns = column_labels
s_fast_ma.columns = column_labels
s_slow_ma.columns = column_labels


# Compute entry and exit signals
raw_entries = (b_fast_ma > b_slow_ma)
raw_exits =   (s_fast_ma < s_slow_ma)

entries, exits = raw_entries.vbt.signals.clean(raw_exits)


# Broadcast price to match shape
price_broadcasted = price.vbt.tile(len(valid_combinations))
price_broadcasted.columns = column_labels

# Run backtest
pf = vbt.Portfolio.from_signals(
    price_broadcasted,
    entries,
    exits,
    init_cash=10000,
    fees=0.001
)

# Analyze performance
total_profits = pf.total_profit()
best_label = total_profits.idxmax()
best_profit = total_profits.max()

# Output
print("Best Parameters:")
print(f"  {best_label}")
print(f"Total Profit: ${best_profit:.2f}")

# 
# fig = go.Figure()
# value = pf[best_label].value()
# 
# fig.add_trace(go.Scatter(x=value.index,
#                          y=value.values,
#                          mode='lines',
#                          name='Asset Value',
#                          line=dict(color='blue')
#                          ))
# 
# 
# fig.add_trace(go.Scatter(x=value[entries[best_label]].index,
#                          y=value[entries[best_label]].values,
#                          mode='markers',
#                          name='Compras',
#                          marker=dict(color='green', size=8, symbol='triangle-up')
# ))
# 
# fig.add_trace(go.Scatter(x=value[exits[best_label]].index,
#                          y=value[exits[best_label]].values,
#                          mode='markers',
#                          name='Vendas',
#                          marker=dict(color='red', size=8, symbol='triangle-down')
# ))
# 
# fig.update_layout(title="Valor do portfólio",
#                   xaxis_title="Data",
#                   yaxis_title="€")
# fig.show()
# 
# 
# 

pf.plot(column=best_label,subplots=['orders', 'trades', 'trade_pnl', 'asset_flow', 'cash_flow', 'assets', 'cash', 'asset_value', 'value', 'cum_returns', 'drawdowns', 'underwater', 'gross_exposure', 'net_exposure']).show()
