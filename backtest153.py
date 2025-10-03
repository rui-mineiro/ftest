import yfinance as yf
import random
import numpy as np
import datetime
import itertools
import pulp
from numba import njit
from env153 import *
from env_plot_00 import *
from env_plot_01 import *
from env_plot_03 import *

data        = get_data(tickerIdx,start_date,end_date)
indicator   = get_indicator(data,indicators)
               
SIG            = get_SIG(indicator)   # Score Limit Hold , Sell and Buy
TickerPrice    = rnd_TickerPrice(indicator)

pTicker        = TickerPrice.iloc[0]
unitsTicker    = pd.Series(1, index=tickerIdx, dtype=int)
unitsTickerRef = get_unitsTickerBuy2(tickerIdx,pTicker,cash )
cash           = cash - unitsTicker.mul(pTicker).sum()

for _ , index in enumerate(data.index, start=1):

    date            = index
    pTicker         = TickerPrice.loc[index]
    S               = SIG.loc[index]
    
    moved = ''

    if t > 0 :
        t-=1
        SBuy=(S[S==1])
        if not ( SBuy.empty ):
            if not SBuy.empty:
                unitsTickerBuy = unitsTicker[unitsTicker.index.intersection(SBuy.index)]
                unitsTickerBuy = get_unitsTickerBuy2(unitsTickerBuy.index,pTicker[unitsTickerBuy.index],cash)
                unitsTickerBuy = unitsTickerBuy[unitsTickerBuy>0]
                if not unitsTickerBuy.empty:
                    tickersH=unitsTickerBuy.index
                    unitsTickerH    = unitsTickerBuy
                    for ticker in tickersH:     
                        moved        = moved+"+"+str(unitsTickerH[ticker])+"#"+ticker
                    cash = cash - unitsTickerH.mul(pTicker[tickersH])[tickersH].sum()
                    unitsTicker[tickersH] = unitsTicker[tickersH] + unitsTickerH[tickersH]
        SSell=(S[S==-1])
        if not SSell.empty:
            unitsTickerSell = unitsTicker[unitsTicker[unitsTicker>0].index.intersection(SSell.index)].astype(int)
            unitsTickerSell = unitsTickerSell[unitsTickerSell>0]
            if not unitsTickerSell.empty:
                unitsTickerL    = unitsTickerSell
                tickersL=unitsTickerL.index
                for ticker in tickersL:     
                    moved        = moved+"-"+str(unitsTickerL[ticker])+"#"+ticker
                cash = cash  + unitsTickerL.mul(pTicker[tickersL])[tickersL].sum()
                unitsTicker[tickersL] = unitsTicker[tickersL] - unitsTickerL[tickersL]
    else:
        t=random.randint(2,10)

    value = unitsTicker.mul(pTicker).sum() + cash

    if not moved:
        movedStr=None
    else:
        movedStr=moved

    records.append({
        "date":  date,       
        "price": pTicker.copy(),    
        "units": unitsTicker.copy(),
        "value": value.copy(),      
        "S"    : S.copy(),
        "moved": movedStr
    })


df = pd.DataFrame(records)

valueRef=unitsTickerRef.mul(pTicker).sum()
print()
print(f"Reference: {valueRef:.2f}€")
print(f"Optimized: {value:.2f}€")

plot_fig00(df)
# indicator_raw_swap=indicator_raw.copy()
# indicator_raw_swap.columns=indicator_raw_swap.columns.swaplevel(0,1)
# # 
# idx=[ 5 , 6,7,8]
# for ticker in tickerIdx:
#     plot_fig03(indicator_raw_swap[ticker], idx,ticker)

