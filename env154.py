import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import datetime
import itertools
import pulp
import re
import os
import pandas as pd
import numpy as np
from datetime import date,datetime, timedelta
from dateutil.relativedelta import relativedelta


# --- PARAMETERS ---

tickerIdx     = [ "VETH.DE" , "DAVV.DE"  ] #  "DAVV.DE" ] # "MSFT" ]   #  "AAPL" , "MSFT"  "DAVV.DE" , "NVDA" , "INTC"] # [ "DAVV.DE" , "NVDA" ] # ["NVDA" , "INTC"] # ["AAPL" , "MSFT" , "DAVV.DE" , "NVDA" , "INTC"]
indicators    = [ "TR001" , "TR003" , "TR005" , "TR010" , "TR015" ]  # True Range Period
indicatorSIG  = [ "TR005" ]  # True Range Period  "#RLTR005"


end_date = date.today()
start_date = end_date - relativedelta(months=12)

end_date_str = end_date.strftime("%Y-%m-%d")
start_date_str = start_date.strftime("%Y-%m-%d")

tickerNum = len(tickerIdx)
tickers    = pd.DataFrame( {'ticker' : tickerIdx })


def get_SIG(df):
    
    tickers = df.columns.get_level_values("Ticker").unique()

    S = pd.DataFrame(0, index=df.index, columns=df[indicatorSIG].columns)

    S[df[indicatorSIG] > 4] = -1
    S[df[indicatorSIG] < 2] =  1
    S.columns = S.columns.droplevel(0)

    return S


def get_TickerPrice(df):
    
    # df: MultiIndex columns = ['Indicator','Ticker']
    low  = df.xs('Low',  level='Indicator', axis=1)
    high = df.xs('High', level='Indicator', axis=1)
    
    rng = np.random.default_rng()              # or np.random.default_rng(42) for reproducible results
    u = rng.random(low.shape)                  # uniform [0,1) per cell
    
    pTicker = low + (high - low) * u              # random price within [Low, High]

    return pTicker



def get_currentScore(indicator,indicator_Prev,index):

    S      = indicator.loc[index]
    S_Prev = indicator_Prev.loc[index]
    
    return S , S_Prev



def get_unitsTickerBuy(tickerIdx, pTicker, cash , lambda_disp=0.05):
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


def get_data(tickerIdx, start_date, end_date, cache_dir="data"):
    os.makedirs(cache_dir, exist_ok=True)

    ticker = list(tickers["ticker"])
    file_path = os.path.join(cache_dir, f"data.csv")

    if os.path.exists(file_path):
        data            = pd.read_csv(file_path,  header=[0,1], index_col=0, parse_dates=True)
        tickers_in_data = data.columns.get_level_values("Ticker").unique().tolist()
        file_mtime      = datetime.fromtimestamp(os.path.getmtime(file_path))
        is_old_file     = (datetime.now() - file_mtime > timedelta(days=1))
        not_same_ticker  = set(tickers_in_data) != set(ticker)
        if is_old_file or not_same_ticker:
            data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)
            data = data.dropna().iloc[1:]
            data.to_csv(file_path)
    else:
        data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)
        data = data.dropna().iloc[1:]
        data.to_csv(file_path)

    return data




def get_indicator(data: pd.DataFrame, indicators: list[str], price_field="Adj Close") -> pd.DataFrame:

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
                    cols[("Min"+str(w).zfill(2), t)]     = Min
                    cols[("Max"+str(w).zfill(2), t)]     = Max
                    cols[("MID"                   , t)]  = Mid
                    cols[("#DMID0"+str(w).zfill(2), t)]  = DMid      # 5
                    cols[("TR0"+str(w).zfill(2)   , t)]  = TR        # 6
##                    cols[("RTR0"+str(w).zfill(2)  , t)]  = TR/Mid    # 
##                    cols[("UTR0"+str(w).zfill(2)  , t)]  = Max-Mid   # 
##                    cols[("#RUTR0"+str(w).zfill(2), t)]  = (Max-Mid)/TR            # 7
##                    cols[("#DRUTR0"+str(w).zfill(2), t)] = ((Max-Mid)/TR).diff()   # 
##                    cols[("LTR0"+str(w).zfill(2)  , t)]  = Mid-Min                 # 
                    cols[("#RLTR0"+str(w).zfill(2) , t)] = (Mid-Min)/TR            # 
                    cols[("#MMR0"+str(w).zfill(2) , t)]  = (Max-Mid)/(Mid-Min)     # 8
##                    cols[("DMMR0"+str(w).zfill(2) , t)]  = ((Max-Mid)/(Mid-Min)).diff()  # 14


    out = pd.DataFrame(cols, index=data.index)
    out.columns = pd.MultiIndex.from_tuples(out.columns, names=["Indicator", "Ticker"])
    

    return out




def backtest(data,indicator,SIG,cash=10000,N=2):

    records = []

    tickerIdx = indicator.columns.get_level_values("Ticker").unique().tolist()

    TickerPrice    = get_TickerPrice(indicator)
    
    pTicker        = TickerPrice.iloc[0]
    unitsTicker    = pd.Series(0, index=tickerIdx, dtype=int)
    unitsTickerRef = get_unitsTickerBuy(tickerIdx,pTicker,cash)
    cash           = cash - unitsTicker.mul(pTicker).sum()
    t = 0
    
    for _ , index in enumerate(indicator.index, start=1):
    
        date            = index
        pTicker         = TickerPrice.loc[index]
        S               = SIG.loc[index]
        
        moved = ''
    
        if t == 0 :
            SBuy=(S[S==1])
            if not SBuy.empty:
                unitsTickerBuy = unitsTicker[unitsTicker.index.intersection(SBuy.index)]
                unitsTickerBuy = get_unitsTickerBuy(unitsTickerBuy.index,pTicker[unitsTickerBuy.index],cash)
                unitsTickerBuy = unitsTickerBuy[unitsTickerBuy>0]
                if not unitsTickerBuy.empty:
                    tickersH=unitsTickerBuy.index
                    unitsTickerH    = unitsTickerBuy
                    for ticker in tickersH:     
                        moved        = moved+"+"+str(unitsTickerH[ticker])+"#"+ticker
                    cash = cash - unitsTickerH.mul(pTicker[tickersH])[tickersH].sum()
                    unitsTicker[tickersH] = unitsTicker[tickersH] + unitsTickerH[tickersH]
                    t=N
            SSell=(S[S==-1])
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
                    t=N
        else:
            t-=1
    
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

    valueRef=unitsTickerRef.mul(pTicker).sum()

    return records,valueRef,value
    

