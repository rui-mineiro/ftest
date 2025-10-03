import yfinance as yf
import random
import numpy as np
import datetime
import itertools
import pulp
from env152 import *
from env_plot_00 import *
from env_plot_01 import *
from env_plot_03 import *

data           = get_data(tickerIdx,start_date,end_date)
indicator_raw  = get_indicator(data,indicators)
indicator      = indicator_raw.copy()
               
SIG            = rnd_SIG(indicator)   # Score Limit Hold , Sell and Buy
TickerPrice    = rnd_TickerPrice(indicator)

# indicator.columns  = indicator.columns.droplevel(0)
# indicator_Prev     = indicator.shift(1)


pTicker        = TickerPrice.iloc[0]
unitsTicker    = pd.Series(1, index=tickerIdx, dtype=int)
unitsTickerRef = get_unitsTickerBuy2(tickerIdx,pTicker,cash )
cash           = cash - unitsTicker.mul(pTicker).sum()


for _ , index in enumerate(data.index, start=1):

    date            = index
    pTicker         = TickerPrice.loc[index]
    S               = SIG.loc[index]
    

#    S , S_Prev      = get_currentScore(indicator,indicator_Prev,index)
#    S_H , S_S , S_B = SL_H.loc[index] , SL_S.loc[index] , SL_B.loc[index]
#    S_H_Prev , S_S_Prev , S_B_Prev  =   SL_H_Prev.loc[index] , SL_S_Prev.loc[index] , SL_B_Prev.loc[index]

    
    moved = ''

    if t > 0 :
        t-=1
        SBuy=(S[S==1])
        if not ( SBuy.empty ):
            # If there are tickes with score > S_B , buy all units of those tickers
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
        # If there are tickes with score < S_S , sell all units of those tickers
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
        # t=random.randint(2,N)
        t=5



    # update portfolio value
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
# 
# idx=[ 5 , 6,7,8]
# for ticker in tickerIdx:
#     plot_fig03(indicator_raw_swap[ticker], idx,ticker)

