
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random
import numpy as np
import datetime
import itertools
import pulp
import re




# --- PARAMETERS ---

tickerIdx = ["AAPL" , "MSFT" , "DAVV.DE" , "NVDA" , "INTC"] # [ "DAVV.DE" , "NVDA" ] # ["NVDA" , "INTC"] # ["AAPL" , "MSFT" , "DAVV.DE" , "NVDA" , "INTC"]
indicators = ["MA05", "MA10", "MSTD05", "MSTD10", "EMA05", "EMA10" , "PCT01" , "PCT05" , "PCT10"]
start_date = "2025-01-01"
end_date   = "2025-09-16"
cash=10000

N = 6

tickerNum = len(tickerIdx)
tickers    = pd.DataFrame( {'ticker' : tickerIdx })


S_H       = [  -1/100 for _ in range(tickerNum) ]                        # Score Hold   Real*tickerNum < 0 [-0.05, -0.05]
S_S       = [  -1/100 for _ in range(tickerNum) ]                        # Score Sell   Real*tickerNum < 0 [-0.01, -0.01]
S_B       = [   1/100 for _ in range(tickerNum) ]                        # Score Buy    Real*tickerNum > 0 [ 0.01,  0.01]

records = []
unitsTicker    = pd.Series()  # Tickers Units
unitsTickerH   = pd.Series()  # Tickers High >   S_K
unitsTickerL   = pd.Series()  # Tickers Low  <  -S_K


def get_currentScore(indicator,index):

    S=indicator.loc[index]
    
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
    data = yf.download(list(tickers["ticker"]), start=start_date, end=end_date,auto_adjust=False)
    # data = yf.download(list(tickers["ticker"]), start=start_date, end=end_date,auto_adjust=False)["Adj Close"]
    data = data.dropna()
    
    # Normalize
    # for ticker in tickers["ticker"]:
    #     data[ticker]=data[ticker] / data[ticker].iloc[0]

    data      = data.iloc[1:]

    return data



def get_indicator(data: pd.DataFrame, indicators: list[str], fields=None) -> pd.DataFrame:


    # Medida do tempo em que não é ultrapassado determinado limite
    # Maximo e minimo nos ultimos X dias

    # data.columns must be a MultiIndex: level0=Price field, level1=Ticker
    if fields is None:
        fields = data.columns.get_level_values(0).unique()

    cols = {}
    for price_field in fields:
        # subframe for one price field -> columns are tickers
        df_field = data[price_field]

        for ticker in df_field.columns:
            s = df_field[ticker]
            for ind in indicators:
                m = re.search(r"\d+$", ind)
                w = int(m.group()) if m else None

                if ind.startswith("MA"):
                    vals = s.rolling(w).mean().fillna(0)
                elif ind.startswith("MSTD"):
                    vals = s.rolling(w).std().fillna(0)
                elif ind.startswith("EMA"):
                    vals = s.ewm(span=w, adjust=False).mean().fillna(0)
                elif ind.startswith("PCT"):
                    vals = s.pct_change(periods=w).fillna(0)
                else:
                    raise ValueError(f"Unknown indicator {ind}")

                cols[(price_field, ind, ticker)] = vals

    out = pd.DataFrame(cols, index=data.index)
    out.columns = pd.MultiIndex.from_tuples(out.columns, names=["Price", "Indicator", "Ticker"])
    out = out.sort_index(axis=1, level=["Price","Indicator","Ticker"])

    return out



