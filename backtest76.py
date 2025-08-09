import vectorbt as vbt
import numpy as np
import pandas as pd
import itertools
import plotly.graph_objects as go
from datetime import datetime, timedelta
import re


symbol = 'SPPW.DE' # 'DAVV.DE' # 'DFEN.DE' # 'SPPW.DE'


def get_price(symbol,startdate,enddate):
    price = vbt.YFData.download(symbol,start=startdate,end=enddate).get('Close')
    return price



def get_mas(price):
    
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
    column_labels = [f"bfast={b_f}_bslow={b_s}_sfast={s_f}_sslow={s_s}" for b_f, b_s, s_f , s_s in valid_combinations]
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

    print(f"Total Profit: ${best_profit:.2f}")

    _ , b_f , _ , b_s , _ , s_f , _ , s_s   = re.split('[_=]', best_label)
    

    return price,best_label,entries[best_label],exits[best_label]


yesterday=4
startdate=datetime(2024, 1, 1)
enddate=datetime(2025,6,5)

price                          = get_price(symbol,startdate,enddate)
price,best_label,entries,exits = get_mas(price)

today=yesterday+1

print("Best Parameters:")
print(f"bfast={b_f}_bslow={b_s}_sfast={s_f}_sslow={s_s}")

