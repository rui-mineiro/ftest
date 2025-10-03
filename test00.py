
import pandas as pd
import numpy as np

tickerIdx = ['DAVV.DE', 'VETH.DE']
TradeAction = ['Buy', 'Sell']
date_index = pd.date_range('2024-10-01', periods=5, freq='B')


cols = pd.MultiIndex.from_product([TradeAction, tickerIdx],names=['Trade', 'Ticker'])
moved = pd.DataFrame(np.nan,index=date_index,columns=cols)

print(moved)


# tickerIdx
# ['DAVV.DE', 'VETH.DE']
# 
# I need to create multiindex dataframe called moved with level 0 called BuySell and level 1 tickerIdx


