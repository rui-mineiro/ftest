
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

tickerIdx = [ "DAVV.DE" , "NVDA" ] # ["NVDA" , "INTC"] # ["AAPL" , "MSFT" , "DAVV.DE" , "NVDA" , "INTC"]
start_date = "2023-11-01"
end_date   = "2025-01-31"
cash=1000

tickerNum = len(tickerIdx)
tickerPct = [ 1/tickerNum for _ in range(tickerNum) ]
tickers    = pd.DataFrame( {'ticker' : tickerIdx })
tickersPct = pd.DataFrame( [ tickerPct ], columns=tickerIdx)


S_H       = [ -10/100 for _ in range(tickerNum) ]                        # Score Hold   Real*tickerNum < 0 [-0.05, -0.05]
S_S       = [  -5/100 for _ in range(tickerNum) ]                        # Score Sell   Real*tickerNum < 0 [-0.01, -0.01]
S_B       = [   5/100 for _ in range(tickerNum) ]                        # Score Buy    Real*tickerNum > 0 [ 0.01,  0.01]



# W_N  = 5
# W    = [  x/W_N for x in range(1,W_N+1) ]
# C_K  = pd.DataFrame([W] * tickerNum , index=tickerIdx).T

# W         = [ 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 ]  # Window Real > 0 and sum = 1
# W_T       = [15/100 , 25/100]
# C_K = pd.DataFrame([[x * w for x in W] for w in W_T] , index=tickerIdx).T

W_L  = 20                                                                     # Window Lenght
W_V  = [  1 for x in range(1,W_L+1) ]                                         # Window Vector
W_W  = [50/100 , 1]                                                           # Window Weights
W    = pd.DataFrame([[x * w for x in W_V] for w in W_W] , index=tickerIdx).T    # Window

W_L       = len(W)

# --- SIMULATION ---
records = []
unitsTicker    = pd.Series()  # Tickers Units
unitsTickerH   = pd.Series()  # Tickers High >   S_K
unitsTickerL   = pd.Series()  # Tickers Low  <  -S_K


def get_currentScore(indicator,W,t,index,tickerIdx):

    # update score
    # S += rTicker
    W_L = len(W)

    if t < W_L+4:
        S=pd.Series(0, index=tickerIdx)
    else:
        indicator_win = indicator.loc[:index].iloc[-W_L:]
        # Score Values
        S=pd.DataFrame( (W.values * indicator_win.values) , columns=tickerIdx).sum(axis=0)
    
    return S



def get_unitsTickerBuy2(tickerIdx, pTicker, cash):
    """
    ILP: maximize sum(units[t]*price[t]) <= cash
    Subject to: for all i,j selected, value_i <= 2 * value_j
    (allows zero positions; ratio enforced only between selected tickers)
    """
    tickers = pd.Index(tickerIdx)
    prices  = pd.Series(pTicker, index=tickers, dtype=float)

    # Upper bound per ticker: you can't buy more than cash // price
    max_units = (np.floor(cash / prices).astype(int)).clip(lower=0)

    # If all max bounds are zero -> infeasible
    if (max_units == 0).all():
        return pd.Series(0, index=tickers, dtype=int)

    prob = pulp.LpProblem("Portfolio_2x_ratio", pulp.LpMaximize)

    # Decision vars
    units = {
        t: pulp.LpVariable(f"units_{t}", lowBound=0, upBound=int(max_units[t]), cat="Integer")
        for t in tickers
    }
    # Binary selection: 1 if ticker is held (>=1 unit), else 0
    y = {t: pulp.LpVariable(f"y_{t}", lowBound=0, upBound=1, cat="Binary") for t in tickers}

    # Objective: maximize portfolio value
    prob += pulp.lpSum(units[t] * prices[t] for t in tickers)

    # Budget
    prob += pulp.lpSum(units[t] * prices[t] for t in tickers) <= cash

    # Link units and selection: if y=0 -> units=0 ; if y=1 -> units>=1 (when feasible)
    for t in tickers:
        prob += units[t] <= max_units[t] * y[t]
        if max_units[t] >= 1:
            prob += units[t] >= y[t]

    # Ratio constraint: for any two selected tickers i,j:
    # value_i - 2*value_j <= M*(2 - y_i - y_j)
    # Tight Big-M: total value cannot exceed cash
    M = float(cash)
    for ti in tickers:
        for tj in tickers:
            prob += units[ti]*prices[ti] - 2*units[tj]*prices[tj] <= M * (2 - y[ti] - y[tj])

    # Solve (your env shows COIN_CMD available)
    status = prob.solve(pulp.COIN_CMD(msg=0))
    if pulp.LpStatus[prob.status] != "Optimal":
        return False

    alloc = pd.Series({t: int(round(units[t].value() or 0)) for t in tickers})
    # Safety: enforce budget (numerical tolerance)
    if (alloc * prices).sum() > cash + 1e-6:
        return False
    return alloc



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


def get_data(tickerIdx,start_date,end_date):
    # --- DOWNLOAD DATA ---
    data = yf.download(list(tickers["ticker"]), start=start_date, end=end_date,auto_adjust=False)["Adj Close"]
    data = data.dropna()
    
    # Normalize
    for ticker in tickers["ticker"]:
        data[ticker]=data[ticker] / data[ticker].iloc[0]

    indicator = data.pct_change().dropna()
    data      = data.iloc[1:]

    return data

def get_indicator(data):
    indicator = data.pct_change().dropna().reindex(data.index, fill_value=0)

    return indicator

