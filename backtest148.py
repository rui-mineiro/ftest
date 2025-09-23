import yfinance as yf
# import pandas as pd
import random
import numpy as np
import datetime
import itertools
import pulp
from env148_01 import *
from env_plot_00 import *
from env_plot_01 import *



data          = get_data(tickerIdx,start_date,end_date)
indicator_raw = get_indicator(data,indicators)
indicator     = indicator_raw[indicators[0]].copy()

# indicator_raw[("TR","TR")]
# indicator_raw.loc[:, [("TR","TR" ,"AAPL"), ("MID","MID", "AAPL")]]
# indicator_raw.loc[:,["TRSTD05","TRMA05"]]

priceTicker    = data["Adj Close"].iloc[0]
unitsTicker    = pd.Series(0, index=tickerIdx, dtype=int)
unitsTickerRef = get_unitsTickerBuy2(tickerIdx,priceTicker,cash )
cash  = cash - unitsTicker.mul(priceTicker).sum()


for t, index in enumerate(data.index, start=1):

    pTicker = data["Adj Close"].loc[index]
    date    = index
    S       = get_currentScore(indicator,index)
    
    moved = ''

    if t % N == 0 :
        # if Score of all existing tickers is <S_H then sell all and hold
        SHold  = S[S <= S_H]
        nSHold = S[S >  S_H]
        if nSHold.empty and unitsTicker.sum()>0:
            unitsTickerL = unitsTicker[unitsTicker[unitsTicker>0].index.intersection(SHold.index)].astype(int)
            tickersL=unitsTickerL.index
            for ticker in tickersL:     
                moved        = moved+"-"+str(unitsTickerL[ticker])+"#"+ticker
            cash = cash + unitsTickerL.mul(pTicker[tickersL]).sum()
            unitsTicker[tickersL] = unitsTicker[tickersL] - unitsTickerL
        else:
            # If there are tickes with score > S_B , buy all units of those tickers
            SBuy   = S[S >  S_B]
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
            SSell  = S[S <= S_S]
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
plot_fig01(indicator_raw)

