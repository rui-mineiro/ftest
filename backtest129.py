import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random
import numpy as np
import datetime
import itertools
import pulp


# --- PARAMETERS ---

tickerIdx = ["AAPL" , "MSFT" , "DAVV.DE" , "NVDA" , "INTC"]
W_N = 10
tickerNum = len(tickerIdx)
tickerPct = [ 1/tickerNum for _ in range(tickerNum) ]
S_H       = [ -5/100 for _ in range(tickerNum) ]
S_B       = [  2/100 for _ in range(tickerNum) ]
S_S       = [ -2/100 for _ in range(tickerNum) ]
C_K_w     = [  x/W_N for x in range(1,W_N+1) ]
C_K       = pd.DataFrame([C_K_w] * tickerNum , index=tickerIdx).T


start_date = "2023-11-01"
end_date   = "2025-01-31"

cash=1000

# S_H, S_B , S_S , C_K = -1/100, 2/100, -2/100, 5 # threshold and cooldown

# C_K=5


tickers    = pd.DataFrame( {'ticker' : tickerIdx })
tickersPct = pd.DataFrame( [ tickerPct ], columns=tickerIdx)


# --- DOWNLOAD DATA ---
data = yf.download(list(tickers["ticker"]), start=start_date, end=end_date,auto_adjust=False)["Adj Close"]
data = data.dropna()

# Normalize
for ticker in tickers["ticker"]:
    data[ticker]=data[ticker] / data[ticker].iloc[0]

rets = data.pct_change().dropna()
data = data.iloc[1:]



# --- SIMULATION ---
records = []
unitsTicker    = pd.Series()   # Tickers Units
unitsTickerH   = pd.Series()  # Tickers High >   S_K
unitsTickerL   = pd.Series()  # Tickers Low  <  -S_K

value = 0

priceTicker = data.iloc[0]


unitsTicker = cash // (priceTicker.div(tickerPct)*2)



# Initial cash for transactions
cash  = cash - unitsTicker.mul(priceTicker).sum()



def get_unitsTickerBuy(tickerIdx, pTicker, cash):
    """
    Integer Linear Programming approach.
    Solves large cases (dozens/hundreds of tickers).
    """
    tickers = pd.Index(tickerIdx)

    # Define problem
    prob = pulp.LpProblem("Portfolio", pulp.LpMaximize)

    # Decision vars: number of units for each ticker
    units = {t: pulp.LpVariable(f"units_{t}", lowBound=0, cat="Integer") for t in tickers}

    # Objective: maximize portfolio value
    prob += pulp.lpSum([units[t] * pTicker[t] for t in tickers])

    # Constraint: total cost <= cash
    prob += pulp.lpSum([units[t] * pTicker[t] for t in tickers]) <= cash

    # Solve
    prob.solve(pulp.COIN_CMD(msg=0))
#    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    # Extract solution
    alloc = pd.Series({t: int(units[t].value()) for t in tickers})
    return alloc



for t, index in enumerate(data.index, start=1):
    pTicker = data.loc[index]
    rTicker = rets.loc[index]
    date    = index

    # update score
    # S += rTicker
    if t-4<W_N:
        S=pd.Series(0, index=tickerIdx)
    else:
        rets_win = rets.loc[:index].iloc[-W_N:]
        S=pd.DataFrame( (C_K.values * rets_win.values) , columns=tickerIdx).sum(axis=0)
    


    moved = pd.Series(None, index=tickerIdx,dtype=str)

    if False:
        C -= 1
    else:
        nSHold = S[S >  S_H]
        SHold  = S[S <= S_H]
        if nSHold.empty and unitsTicker.sum()>0:
            # Sell all positive number of units of all the tickers
            unitsTickerL = unitsTicker[unitsTicker[unitsTicker>0].index.intersection(SHold.index)].astype(int)
            tickersL=unitsTickerL.index
            for ticker in tickersL:     
                moved[ticker]        = "-"+str(unitsTickerL[ticker])+"#"+ticker
            cash = cash + unitsTickerL.mul(pTicker[tickersL]).sum()
            unitsTicker[tickersL] = unitsTicker[tickersL] - unitsTickerL
        else:
            SBuy   = S[S >  S_B]
            nSBuy  = S[S <= S_B]

            if not SBuy.empty:                  # If there are some tickers above S_B 
                                                # Sell random units of ticker with negative value bellow S_B with lowest score  # Not random ?!!!
                nSBuy  = S[S <= S_B]
                nSBuy  = nSBuy[nSBuy < 0]
                unitsTickerSell=unitsTicker[unitsTicker[unitsTicker>0].index.intersection(nSBuy.index)].astype(int)
                if not unitsTickerSell.empty:
                    tmpS         = nSBuy[unitsTickerSell.index]
                    tickersL     = [tmpS.idxmin()]
                    unitsTickerL = unitsTicker[tickersL].astype(int)
                    for ticker in tickersL:
                        moved[ticker]    = "-"+str(unitsTickerL[ticker])+"#"+ticker
                    cash = cash + unitsTickerL.mul(pTicker[tickersL])[tickersL].sum()
                    unitsTicker[tickersL] = unitsTicker[tickersL] - unitsTickerL[tickersL]

                unitsTickerBuy=unitsTicker[unitsTicker.index.intersection(SBuy.index)].astype(int)    
                                                        # Buy all units of all the tickers above S_B
                unitsTickerBuy = get_unitsTickerBuy(unitsTickerBuy.index,pTicker[unitsTickerBuy.index],cash)
                unitsTickerBuy = unitsTickerBuy[unitsTickerBuy>0]
                if not unitsTickerBuy.empty:
                    tickersH=unitsTickerBuy.index
                    unitsTickerH    = unitsTickerBuy
                    for ticker in tickersH:     
                        moved[ticker]        = "+"+str(unitsTickerH[ticker])+"#"+ticker
                    cash = cash - unitsTickerH.mul(pTicker[tickersH])[tickersH].sum()
                    unitsTicker[tickersH] = unitsTicker[tickersH] + unitsTickerH[tickersH]

            SSell  = S[S <= S_S]
            if not SSell.empty:                # If there are some tickers bellow S_S
                                               # Buy random units of the ticker with positive value above S_S with higest score
                nSSell = S[S >  S_S]
                nSSell = nSSell[nSSell > 0]
                if not nSSell.empty:                     
                    tickersH=[nSSell.idxmax()]
                    unitsTickerBuy = get_unitsTickerBuy(tickersH,pTicker[tickersH],cash)
                    unitsTickerBuy = unitsTickerBuy[unitsTickerBuy>0]
                    if not unitsTickerBuy.empty:
                        unitsTickerH   = unitsTickerBuy.apply(lambda x: np.random.randint(0, int(x)+1))
                        for ticker in tickersH:
                            moved[ticker]    = "+"+str(unitsTickerH[ticker])+"#"+ticker
                        cash = cash - unitsTickerH.mul(pTicker[tickersH])[tickersH].sum()
                        unitsTicker[tickersH] = unitsTicker[tickersH] + unitsTickerH[tickersH]

                                                  # Sell random units of all the ticker bellow S_S #
                unitsTickerSell=unitsTicker[unitsTicker[unitsTicker>0].index.intersection(SSell.index)].astype(int)
                unitsTickerSell=unitsTickerSell[unitsTickerSell>0]
                if not unitsTickerSell.empty:
                    unitsTickerL    = unitsTickerSell
                    tickersL=unitsTickerL.index
                    for ticker in tickersL:     
                        moved[ticker]        = "-"+str(unitsTickerL[ticker])+"#"+ticker
                    cash = cash  + unitsTickerL.mul(pTicker[tickersL])[tickersL].sum()
                    unitsTicker[tickersL] = unitsTicker[tickersL] - unitsTickerL[tickersL]

#        S=pd.Series(0, index=tickerIdx)
#        C = C_K

    # update portfolio value
    value = unitsTicker.mul(pTicker).sum() + cash

    movedStr=''.join(moved.dropna().astype(str).tolist().copy())
    if not movedStr:
        movedStr=None


    records.append({
        "date":  date,       
        "price": pTicker.copy(),    
        "units": unitsTicker.copy(),
        "value": value.copy(),      
        "S"    : S.copy(),
        "moved": movedStr
    })


df = pd.DataFrame(records)

# --- Step 0: Create subplot (1 row, 1 col here, but extendable) ---
fig = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]])


# --- Step 1: Expand df["price"] into wide format ---
price_wide = df["price"].apply(pd.Series)
price_wide["date"] = df["date"].values   # attach the real dates
price_df = price_wide.melt(id_vars="date", var_name="ticker", value_name="price")
for ticker in price_df["ticker"].unique():
    data = price_df[price_df["ticker"] == ticker]
    fig.add_trace(
        go.Scatter(
            x=data["date"],
            y=data["price"],
            mode="lines",
            name=ticker
        ),
        row=1, col=1
    )



S_wide = df["S"].apply(pd.Series)
S_wide["date"] = df["date"].values   # attach the real dates
S_df = S_wide.melt(id_vars="date", var_name="ticker", value_name="Score")
for ticker in S_df["ticker"].unique():
    data = S_df[S_df["ticker"] == ticker]
    fig.add_trace(
        go.Scatter(
            x=data["date"],
            y=data["Score"],
            mode="lines",
            name="S"+ticker
        ),
        row=1, col=1
    )


moved_wide = df["moved"].apply(pd.Series)
moved_wide["date"] = df["date"].values
moved_wide=moved_wide.dropna()
moved_wide.columns=['moved','date']
# # moved_df = moved_wide.melt(id_vars="date", var_name="ticker", value_name="moved")
# moved_df = moved_wide.melt(id_vars="date", value_name="moved")
# moved_df = moved_df.dropna()
# for ticker in moved_df["ticker"].unique():
#    data = moved_df[moved_df["ticker"] == ticker]
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


# --- Step 4: Layout ---
fig.update_layout(
    title="Prices by Ticker",
    xaxis_title="Date",
    yaxis_title="Price",
    template="plotly_white"
)

fig.show()


