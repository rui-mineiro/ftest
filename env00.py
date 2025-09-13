
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

tickerIdx = ["AAPL" , "MSFT" , "DAVV.DE" , "NVDA" , "INTC"] # [ "DAVV.DE" , "NVDA" ] # ["NVDA" , "INTC"] # ["AAPL" , "MSFT" , "DAVV.DE" , "NVDA" , "INTC"]
start_date = "2023-11-01"
end_date   = "2025-01-31"
cash=10000

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
W_TW  = [1 , 1 , 1 , 1 , 1 ]                               # Window tickerIdx Weights
W    = pd.DataFrame([[x * w for x in W_V] for w in W_TW] , index=tickerIdx).T    # Window

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



def get_unitsTickerBuy2(tickerIdx, pTicker, cash , lambda_disp=0.05):
    """
    Choose integer units per ticker under cash.
    Objective: maximize total invested - lambda_disp * (max(allocation) - min(allocation)),
    where allocation[t] = units[t] * price[t].
    Smaller lambda_disp -> prioritize value; larger -> equalize allocations more.
    """
    tickers = pd.Index(tickerIdx)
    pTicker = pd.Series(pTicker).reindex(tickers)

    # Model
    prob = pulp.LpProblem("BalancedPortfolio", pulp.LpMaximize)

    # Vars
    units = {t: pulp.LpVariable(f"units_{t}", lowBound=0, cat="Integer") for t in tickers}
    alloc = {t: pulp.LpVariable(f"alloc_{t}", lowBound=0) for t in tickers}  # € invested in t
    V = pulp.LpVariable("total_invested", lowBound=0)
    a_max = pulp.LpVariable("alloc_max", lowBound=0)
    a_min = pulp.LpVariable("alloc_min", lowBound=0)

    # Link allocation and units; total and bounds for max/min
    for t in tickers:
        prob += alloc[t] == pTicker[t] * units[t]
        prob += a_max >= alloc[t]
        prob += a_min <= alloc[t]

    prob += V == pulp.lpSum(alloc[t] for t in tickers)

    # Budget
    prob += V <= cash

    # Objective: invest as much as possible while keeping allocations close
    prob += V - lambda_disp * (a_max - a_min)

    # Solve
    prob.solve(pulp.COIN_CMD(msg=0))

    # Extract
    if pulp.LpStatus[prob.status] != "Optimal":
        return pd.Series(0, index=tickers, dtype=int)

    alloc_units = pd.Series({t: int(max(0, round(units[t].value()))) for t in tickers}, dtype=int)

    # If nothing affordable, return zeros
    if (alloc_units == 0).all():
        return pd.Series(0, index=tickers, dtype=int)

    return alloc_units



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
    # for ticker in tickers["ticker"]:
    #     data[ticker]=data[ticker] / data[ticker].iloc[0]

    indicator = data.pct_change().dropna()
    data      = data.iloc[1:]

    return data

def get_indicator(data):
    indicator = data.pct_change().dropna().reindex(data.index, fill_value=0)

    return indicator

