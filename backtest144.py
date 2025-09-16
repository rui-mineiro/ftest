import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random
import numpy as np
import datetime
import itertools
import pulp
from env144 import *
import plotly.io as pio

pio.renderers.default = "browser"
data          = get_data(tickerIdx,start_date,end_date)
indicator_raw = get_indicator(data,indicators)
indicator     = indicator_raw["Adj Close", "PCT01"].copy()


priceTicker    = data["Adj Close"].iloc[0]
unitsTicker    = pd.Series(0, index=tickerIdx, dtype=int)
unitsTickerRef = get_unitsTickerBuy2(tickerIdx,priceTicker,cash )
cash  = cash - unitsTicker.mul(priceTicker).sum()


for t, index in enumerate(data.index, start=1):

    pTicker = data["Adj Close"].loc[index]
    date    = index
    S       = get_currentScore(indicator,index)
    
    moved = ''

    if False:
        C -= 1
    else:
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


# --- Step 0: Create subplot (1 row, 1 col here, but extendable) ---
fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                    vertical_spacing=0.05 ,
                    row_heights=[0.4 , 0.6],
                    specs=[[{"secondary_y": True}],[{"secondary_y": True}]])


# --- Step 1: Expand df["price"] into wide format ---
price_wide = df["price"].apply(pd.Series)
price_wide["date"] = df["date"].values   # attach the real dates
price_df = price_wide.melt(id_vars="date", var_name="ticker", value_name="price")
for ticker in price_df["ticker"].unique():
    plotData = price_df[price_df["ticker"] == ticker]
    fig.add_trace(
        go.Scatter(
            x=plotData["date"],
            y=plotData["price"],
            mode="lines",
            name=ticker
        ),
        row=1, col=1
    )



S_wide = df["S"].apply(pd.Series)
S_wide["date"] = df["date"].values   # attach the real dates
S_df = S_wide.melt(id_vars="date", var_name="ticker", value_name="Score")
for ticker in S_df["ticker"].unique():
    plotData = S_df[S_df["ticker"] == ticker]
    fig.add_trace(
        go.Scatter(
            x=plotData["date"],
            y=plotData["Score"],
            mode="lines",
            name="S"+ticker
        ),
        row=1, col=1
    )


moved_wide = df["moved"].apply(pd.Series)
if len(moved_wide.columns) == 1:
    moved_wide["date"] = df["date"].values
    moved_wide=moved_wide.dropna()
    moved_wide.columns=['moved','date']
    
    fig.add_trace(go.Scatter(
        x=moved_wide["date"], y=[df.loc[df["date"]==d, "value"].values[0] for d in moved_wide["date"]],
        mode="markers+text",
        marker=dict(size=9, symbol="triangle-up", color="black"),
        text=moved_wide["moved"], textposition="top center",
        name="Switches"
        ),secondary_y=True,
        row=1, col=1
    )



fig.add_trace(
    go.Scatter(
        x=df["date"],
        y=df["value"],
        mode="lines",
        name="Value",
        line=dict(color="green", dash="dot")
    ),
    row=1, col=1 ,
    secondary_y=True
)


units_wide = df["units"].apply(pd.Series)
price_wide = df["price"].apply(pd.Series)
unitsValue_wide=price_wide.mul(units_wide)

unitsValue_wide["date"] = df["date"].values   # attach the real dates
units_df = unitsValue_wide.melt(id_vars="date", var_name="ticker", value_name="units")
for ticker in units_df["ticker"].unique():
    plotData = units_df[units_df["ticker"] == ticker]
    fig.add_trace(
        go.Scatter(
            x=plotData["date"],
            y=plotData["units"],
            mode="lines",
            name=ticker
        ),
        row=2, col=1
    )


# --- Step 4: Layout ---
fig.update_layout(
    title="Prices by Ticker",
    xaxis_title="Date",
    yaxis_title="Price",
    template="plotly_white",
    yaxis_type="linear"   # logarithmic scale
)

fig.show()


