import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import datetime
import itertools
import pulp
import re
import re
import pandas as pd
import numpy as np



# --- PARAMETERS ---

tickerIdx = [ "AAPL"  , "MSFT"  ]   #  "DAVV.DE" , "NVDA" , "INTC"] # [ "DAVV.DE" , "NVDA" ] # ["NVDA" , "INTC"] # ["AAPL" , "MSFT" , "DAVV.DE" , "NVDA" , "INTC"]
# indicators = ["MA05", "MA10", "MSTD05", "MSTD10", "EMA05", "EMA10" , "PCT01" , "PCT05" , "PCT10" , "TRMA05", "TRSTD10" , "MID05" , "MID10" ]
# indicators = [ "MA05", "MA10", "TR" , "TRMA05", "TRSTD05" , "MID" , "MIDMA05" , "MIDSTD05" ]  # True Range and Median Price with previous close
indicators = [ "TR002" , "MID002"]  # True Range and Median Price with previous close
indicatorScore = [ "MID002" ]
start_date = "2025-01-05"
end_date   = "2025-09-20"
cash=10000

N = 4
t = 1

tickerNum = len(tickerIdx)
tickers    = pd.DataFrame( {'ticker' : tickerIdx })


# S_H       = [  -1/100 for _ in range(tickerNum) ]                        # Score Hold   Real*tickerNum < 0 [-0.05, -0.05]
# S_S       = [  -1/100 for _ in range(tickerNum) ]                        # Score Sell   Real*tickerNum < 0 [-0.01, -0.01]
# S_B       = [   1/100 for _ in range(tickerNum) ]                        # Score Buy    Real*tickerNum > 0 [ 0.01,  0.01]

records = []
unitsTicker    = pd.Series()  # Tickers Units
unitsTickerH   = pd.Series()  # Tickers High >   S_K
unitsTickerL   = pd.Series()  # Tickers Low  <  -S_K

def get_ScoreLimits(data):

    S_H=data["High"]
    S_S=data["High"]
    S_B=data["Low" ]

    SL_H_Prev , SL_S_Prev , SL_B_Prev = S_H.shift(1) , S_S.shift(1) , S_B.shift(1)


    return S_H , S_S , S_B , SL_H_Prev , SL_S_Prev , SL_B_Prev

def get_currentScore(indicator,indicator_Prev,index):

    S      = indicator.loc[index]
    S_Prev = indicator_Prev.loc[index]
    
    return S , S_Prev



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






def get_indicator(data: pd.DataFrame, indicators: list[str], price_field="Adj Close") -> pd.DataFrame:
    """
    data.columns: MultiIndex with level0=Price field ('Open','High','Low','Close', ...)
                  level1=Ticker
    indicators: list like ['TR','TRMA05','TRSTD10','MID','MIDMA10','MIDSTD05', ...]
    """

    cols       = {}
    
    # ---- Preload OHLC if needed for TR/MID ----
    need_tr = any(ind.startswith("TR0") for ind in indicators)
    need_mid = any(ind.startswith("MID0") for ind in indicators)
    H = L = C = O = None

    if need_tr or need_mid:
        try:
            H     = data["High"]
            L     = data["Low"]
            C     = data["Close"]
            O     = data["Open"]
        except KeyError as e:
            raise ValueError("TR/MID need 'Open','High','Low','Close' in data columns") from e

    # ---- TRUE RANGE + rolling stats ----
    if need_tr or need_mid:
        common_tickers = H.columns.intersection(L.columns).intersection(C.columns).intersection(O.columns)
        for t in common_tickers:
            for ind in indicators:
                # TR0xx                
                if ind.startswith("TR0"):
                    m = re.search(r"\d+$", ind)
                    if not m:
                        raise ValueError(f"{ind} requires a numeric window, e.g., TR0010")
                    w = int(m.group())
                    Cprev = C[t].shift(w)
                    Low   = L[t].rolling(w).min()
                    High  = H[t].rolling(w).max()
                    tr    = pd.concat([(High-Low),
                                    (High-Cprev).abs(),
                                    (Low-Cprev).abs()], axis=1).max(axis=1)
                    cols[(ind, t)] = tr
                # MID0xx                                    
                elif ind.startswith("MID0"):
                    m = re.search(r"\d+$", ind)
                    if not m:
                        raise ValueError(f"{ind} requires a numeric window, e.g., MID0010")
                    w = int(m.group())
                    Cprev = C[t].shift(w)
                    Close = C[t]
                    Open  = O[t].shift(w-1)
                    mid = (Open+2*Close) / 3.0
                    cols[( ind, t)] = mid
            cols[("High", t)] = H[t]
            cols[("Low" , t)] = L[t]
            for ind in indicators:
                if ind.startswith("TR0"):
                    TR=cols[(ind, t)]
                if ind.startswith("MID0"):
                    MID=cols[(ind, t)]
            cols[("ML", t)]=MID-TR/2
            cols[("MH", t)]=MID+TR/2
    
    out = pd.DataFrame(cols, index=data.index)
    out.columns = pd.MultiIndex.from_tuples(out.columns, names=["Indicator", "Ticker"])
    out = out.sort_index(axis=1, level=["Indicator", "Ticker"])
    
    

    return out





