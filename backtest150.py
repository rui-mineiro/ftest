import yfinance as yf
import random
import numpy as np
import datetime
import itertools
import pulp
from env150 import *
from env_plot_00 import *
from env_plot_01 import *
from env_plot_02 import *
from env_plot_03 import *

data               = get_data(tickerIdx,start_date,end_date)
indicator_raw      = get_indicator(data,indicators)
SL_H , SL_S , SL_B , SL_H_Prev , SL_S_Prev , SL_B_Prev = get_ScoreLimits(data)   # Score Limit Hold , Sell and Buy
indicator          = indicator_raw[indicatorScore]
indicator.columns  = indicator.columns.droplevel(0)
indicator_Prev     = indicator.shift(1)


priceTicker    = data["Close"].iloc[0]
unitsTicker    = pd.Series(0, index=tickerIdx, dtype=int)
unitsTickerRef = get_unitsTickerBuy2(tickerIdx,priceTicker,cash )
cash  = cash - unitsTicker.mul(priceTicker).sum()


for _ , index in enumerate(data.index, start=1):

    pTicker         = data["Adj Close"].loc[index]
    date            = index
    S , S_Prev      = get_currentScore(indicator,indicator_Prev,index)
    S_H , S_S , S_B = SL_H.loc[index] , SL_S.loc[index] , SL_B.loc[index]
    S_H_Prev , S_S_Prev , S_B_Prev  =   SL_H_Prev.loc[index] , SL_S_Prev.loc[index] , SL_B_Prev.loc[index]

    
    moved = ''

    if t > 0 :
        t-=1
        # if Score of all existing tickers is <S_H then sell all and hold
        SHold  = S[(S_S_Prev > S_Prev ) & (S_S < S )]
        nSHold = S[~(S_S_Prev > S_Prev ) & (S_S < S )]
        if nSHold.empty and unitsTicker.sum()>0:
            unitsTickerL = unitsTicker[unitsTicker[unitsTicker>0].index.intersection(SHold.index)].astype(int)
            tickersL=unitsTickerL.index
            for ticker in tickersL:     
                moved        = moved+"-"+str(unitsTickerL[ticker])+"#"+ticker
            cash = cash + unitsTickerL.mul(pTicker[tickersL]).sum()
            unitsTicker[tickersL] = unitsTicker[tickersL] - unitsTickerL
        else:
            # If there are tickes with score > S_B , buy all units of those tickers
            SBuy   = S[(S_B_Prev < S_Prev ) & ( S_B > S )]
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
            SSell  = S[(S_S_Prev > S_Prev ) & (S_S < S )]
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
        t=2



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

# plot_fig00(df)
indicator_raw_swap=indicator_raw.copy()
indicator_raw_swap.columns=indicator_raw.columns.swaplevel(0,1)

#for ticker in tickerIdx:
#    plot_fig01(indicator_raw_swap[ticker],ticker)
#
#for ticker in tickerIdx:
#    plot_fig02(indicator_raw_swap[ticker],ticker)



# for ticker in tickerIdx:
#     plot_fig03(indicator_raw_swap[ticker], indicators ,ticker)

idx=[ 5,  9, 13]


indicator_raw      = get_indicator(data,[ "TR002" ])
indicator_raw_swap=indicator_raw.copy()
indicator_raw_swap.columns=indicator_raw.columns.swaplevel(0,1)
for ticker in tickerIdx:
    plot_fig03(indicator_raw_swap[ticker], idx ,ticker)


indicator_raw      = get_indicator(data,[ "TR003" ])
indicator_raw_swap=indicator_raw.copy()
indicator_raw_swap.columns=indicator_raw.columns.swaplevel(0,1)
for ticker in tickerIdx:
    plot_fig03(indicator_raw_swap[ticker], idx ,ticker)

indicator_raw      = get_indicator(data,[ "TR005" ])
indicator_raw_swap=indicator_raw.copy()
indicator_raw_swap.columns=indicator_raw.columns.swaplevel(0,1)
for ticker in tickerIdx:
    plot_fig03(indicator_raw_swap[ticker], idx ,ticker)

indicator_raw      = get_indicator(data,[ "TR010" ])
indicator_raw_swap=indicator_raw.copy()
indicator_raw_swap.columns=indicator_raw.columns.swaplevel(0,1)
for ticker in tickerIdx:
    plot_fig03(indicator_raw_swap[ticker], idx ,ticker)


indicator_raw      = get_indicator(data,[ "TR015" ])
indicator_raw_swap=indicator_raw.copy()
indicator_raw_swap.columns=indicator_raw.columns.swaplevel(0,1)
for ticker in tickerIdx:
    plot_fig03(indicator_raw_swap[ticker], idx ,ticker)




