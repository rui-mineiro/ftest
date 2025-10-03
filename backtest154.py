import yfinance as yf
import random
import numpy as np
import datetime
import itertools
import pulp
from numba import njit
from env154 import *
from env_plot_00 import *
from env_plot_01 import *
from env_plot_03 import *

data                   = get_data(tickerIdx,start_date,end_date)
indicator              = get_indicator(data,indicators)
SIG                    = get_SIG(indicator)
records,valueRef,value = backtest(data,indicator,SIG)
df                     = pd.DataFrame(records)

print()
print(f"Reference: {valueRef:.2f}€")
print(f"Optimized: {value:.2f}€")

plot_fig00(df)

# indicator=get_indicator(data,indicators)
indicator.columns=indicator.columns.swaplevel(0,1)

for ticker in tickerIdx:
    plot_fig03(df,indicator[ticker],indicatorSIG,ticker)

