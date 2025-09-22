
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
import re
import pandas as pd
import numpy as np



# --- PARAMETERS ---

tickerIdx = ["AAPL" ] # , "MSFT" , "DAVV.DE" , "NVDA" , "INTC"] # [ "DAVV.DE" , "NVDA" ] # ["NVDA" , "INTC"] # ["AAPL" , "MSFT" , "DAVV.DE" , "NVDA" , "INTC"]
# indicators = ["MA05", "MA10", "MSTD05", "MSTD10", "EMA05", "EMA10" , "PCT01" , "PCT05" , "PCT10" , "TRMA05", "TRSTD10" , "MID05" , "MID10" ]
indicators = [ "MA05", "MA10", "TR" , "TRMA05", "TRSTD05" , "MID" , "MIDMA05" , "MIDSTD05" ]  # True Range and Median Price with previous close
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

    cols = {}

    # ---- Preload OHLC if needed for TR/MID ----
    need_tr = any(ind.startswith("TR") for ind in indicators)
    need_mid = any(ind.startswith("MID") for ind in indicators)
    H = L = C = O = None

    if need_tr or need_mid:
        try:
            H = data["High"]
            L = data["Low"]
            C = data["Close"]
            O = data["Open"]
        except KeyError as e:
            raise ValueError("TR/MID need 'Open','High','Low','Close' in data columns") from e

    # ---- TRUE RANGE + rolling stats ----
    if need_tr:
        common_tickers = H.columns.intersection(L.columns).intersection(C.columns)
        for t in common_tickers:
            prevC = C[t].shift(1)
            tr = pd.concat([(H[t]-L[t]),
                            (H[t]-prevC).abs(),
                            (L[t]-prevC).abs()], axis=1).max(axis=1)

            # raw TR
            if "TR" in indicators:
                cols[("TR" , t)] = tr.fillna(0)

            # TRMAxx / TRSTDxx
            for ind in indicators:
                if ind.startswith("TRMA"):
                    m = re.search(r"\d+$", ind)
                    if not m:
                        raise ValueError(f"{ind} requires a numeric window, e.g., TRMA14")
                    w = int(m.group())
                    cols[(ind, t)] = tr.rolling(w).mean().fillna(0)
                elif ind.startswith("TRSTD"):
                    m = re.search(r"\d+$", ind)
                    if not m:
                        raise ValueError(f"{ind} requires a numeric window, e.g., TRSTD14")
                    w = int(m.group())
                    cols[( ind, t)] = tr.rolling(w).std(ddof=0).fillna(0)

    # ---- MID price + rolling stats ----
    if need_mid:
        common_tickers = H.columns.intersection(L.columns).intersection(C.columns).intersection(O.columns)
        for t in common_tickers:
            mid = (C[t].shift(1) + O[t] + H[t] + L[t]) / 4.0

            if "MID" in indicators:
                cols[("MID", t)] = mid.fillna(0)

            for ind in indicators:
                if ind.startswith("MIDMA"):
                    m = re.search(r"\d+$", ind)
                    if not m:
                        raise ValueError(f"{ind} requires a numeric window, e.g., MIDMA14")
                    w = int(m.group())
                    cols[(ind, t)] = mid.rolling(w).mean().fillna(0)
                elif ind.startswith("MIDSTD"):
                    m = re.search(r"\d+$", ind)
                    if not m:
                        raise ValueError(f"{ind} requires a numeric window, e.g., MIDSTD14")
                    w = int(m.group())
                    cols[(ind, t)] = mid.rolling(w).std(ddof=0).fillna(0)

    # ---- Existing per-field indicators (MA, MSTD, EMA, PCT) ----
    df_field = data[price_field]
    
    for ticker in df_field.columns:
        s = df_field[ticker]
        for ind in indicators:
            m = re.search(r"\d+$", ind)
            w = int(m.group()) if m else None
    
            if ind.startswith("MA"):
                if not w: raise ValueError("MA requires a window, e.g., MA20")
                vals = s.rolling(w).mean().fillna(0)
            elif ind.startswith("MSTD") and not ind.startswith(("TRSTD", "MIDSTD")):
                if not w: raise ValueError("MSTD requires a window, e.g., MSTD20")
                vals = s.rolling(w).std(ddof=0).fillna(0)
            elif ind.startswith("EMA"):
                if not w: raise ValueError("EMA requires a span, e.g., EMA20")
                vals = s.ewm(span=w, adjust=False).mean().fillna(0)
            elif ind.startswith("PCT"):
                if not w: raise ValueError("PCT requires periods, e.g., PCT1")
                vals = s.pct_change(periods=w).fillna(0)
            else:
                continue  # handled above or not applicable
    
            cols[(ind, ticker)] = vals

    out = pd.DataFrame(cols, index=data.index)
    out.columns = pd.MultiIndex.from_tuples(out.columns, names=["Indicator", "Ticker"])
    out = out.sort_index(axis=1, level=["Indicator", "Ticker"])
    return out





