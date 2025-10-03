import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import datetime
import itertools
import pulp
import re
import pandas as pd
import numpy as np
from datetime import date
from dateutil.relativedelta import relativedelta


# --- PARAMETERS ---

tickerIdx   = [ "VETH.DE" ] #  "DAVV.DE" ] # "MSFT" ]   #  "AAPL" , "MSFT"  "DAVV.DE" , "NVDA" , "INTC"] # [ "DAVV.DE" , "NVDA" ] # ["NVDA" , "INTC"] # ["AAPL" , "MSFT" , "DAVV.DE" , "NVDA" , "INTC"]
indicators  = [ "TR005" ]  # True Range Period


end_date = date.today()
start_date = end_date - relativedelta(months=12)

end_date_str = end_date.strftime("%Y-%m-%d")
start_date_str = start_date.strftime("%Y-%m-%d")

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


def get_ScoreLimits(df):
    for t in df.columns.get_level_values("Ticker").unique():
        base = df[("#RUTR005", t)]
        df[("S_H", t)] = (base > 0.6).astype(int)
        df[("S_L", t)] = (base < 0.4).astype(int)
    df = df.sort_index(axis=1)

    S_H , S_S , S_B = df[("S_H", tickerIdx[0])] , df[("S_H", tickerIdx[0])] , df[("S_H", tickerIdx[0])] 

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


    need_mid = any(ind.startswith("TR0") for ind in indicators)
    if need_mid:
        H = data["High"]
        L = data["Low"]
        C = data["Close"]
        O = data["Open"]
        common_tickers = H.columns.intersection(L.columns).intersection(C.columns).intersection(O.columns)
        for t in common_tickers:
            for ind in indicators:
                # MIDxxx and TRxxx
                if ind.startswith("TR0"):
                    m = re.search(r"\d+$", ind)
                    if not m:
                        raise ValueError(f"{ind} requires a numeric window, e.g., MID0010")
                    w = int(m.group())
                    if w == 0:
                        Min   = L[t]
                        Max  = H[t]
                    else:
                        Min   = L[t].rolling(w).min()
                        Max  = H[t].rolling(w).max()
                    Cprev = C[t].shift(1)
                    Open  = O[t]
                    Close = C[t]
                    Mid   = (Open+Close)/2
                    DMid  = Mid.rolling(w).mean().diff()
                    TR    = Max - Min
                    cols[("Low" , t)] = L[t]
                    cols[("High", t)] = H[t]
                    cols[("Min", t)]  = Min
                    cols[("Max", t)]  = Max
                    cols[("MID"                   , t)]  = Mid
                    cols[("#DMID0"+str(w).zfill(2), t)]  = DMid      # 5
                    cols[("TR0"+str(w).zfill(2)   , t)]  = TR        # 6
##                    cols[("RTR0"+str(w).zfill(2)  , t)]  = TR/Mid    # 
##                    cols[("UTR0"+str(w).zfill(2)  , t)]  = Max-Mid   # 
                    cols[("#RUTR0"+str(w).zfill(2), t)]  = (Max-Mid)/TR            # 7
##                    cols[("#DRUTR0"+str(w).zfill(2), t)] = ((Max-Mid)/TR).diff()   # 
##                    cols[("LTR0"+str(w).zfill(2)  , t)]  = Mid-Min                 # 
##                    cols[("RLTR0"+str(w).zfill(2) , t)]  = (Mid-Min)/TR            # 
                    cols[("#MMR0"+str(w).zfill(2) , t)]  = (Max-Mid)/(Mid-Min)     # 8
##                    cols[("DMMR0"+str(w).zfill(2) , t)]  = ((Max-Mid)/(Mid-Min)).diff()  # 14




    out = pd.DataFrame(cols, index=data.index)
    out.columns = pd.MultiIndex.from_tuples(out.columns, names=["Indicator", "Ticker"])
    # out = out.sort_index(axis=1, level=["Indicator", "Ticker"])
    
    

    return out


