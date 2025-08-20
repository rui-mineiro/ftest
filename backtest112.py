import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random
import numpy as np

# --- PARAMETERS ---

tickerIdx     = ["AAPL", "MSFT" ] # , "DAVV.DE"  ]
tickerPct     = [ 0.5 , 0.5 ] # , 0.4 ]

start_date = "2024-01-01"
end_date   = "2025-05-31"

cash=1000

S_K , C_K = 1/100 , 5 # threshold and cooldown


tickers    = pd.DataFrame( {'ticker' : tickerIdx })
tickersPct = pd.DataFrame( [ tickerPct ], columns=tickerIdx)

S=pd.Series(0, index=tickerIdx)
C = 0

# --- DOWNLOAD DATA ---
data = yf.download(list(tickers["ticker"]), start=start_date, end=end_date,auto_adjust=False)["Adj Close"]
data = data.dropna()

# Normalize
for ticker in tickers["ticker"]:
    data[ticker]=data[ticker] / data[ticker].iloc[0]

rets = data.pct_change().dropna()
data=data.iloc[1:]


# --- SIMULATION ---
records = []
unitsTicker  = pd.DataFrame()   # Tickers Units
unitsTickerH = pd.DataFrame()  # Tickers High >   S_K
unitsTickerL = pd.DataFrame()  # Tickers Low  <  -S_K
value = 0

priceTicker = data.iloc[0]


for ticker in tickers["ticker"]:
    unitsTicker[ticker] = cash // ( priceTicker[ticker] * 2 / tickersPct[ticker] )



# Initial cash for transactions
cash  = cash - unitsTicker.mul(priceTicker).sum(axis=1).iloc[0]

for t, index in enumerate(data.index, start=1):
    pTicker = data.loc[index]
    rTicker = rets.loc[index]
    date    = index

    # update score
    S += rTicker

    moved = None
    if C > 0:
        C -= 1
    else:
        SHBuy         = S[S >    S_K]  # Compra estas
        SLSell        = S[S <=   S_K]  # Se não houver SHBuy vende apenas
        unitsTickerRnd  = unitsTicker.apply(lambda x: np.random.randint(0, int(x.iloc[0]) + 1))
        if not SHBuy.empty:
            unitsTickerL[SLSell.index] = unitsTicker[SLSell.index]-unitsTickerRnd[SLSell.index]
            cash = cash + unitsTickerL[SLSell.index].mul(pTicker[SLSell.index]).sum(axis=1).iloc[0]
            unitsTicker[SLSell.index] = unitsTicker[SLSell.index] - unitsTickerL[SLSell.index]

            for ticker in SHBuy.index:
                unitsTickerH[ticker] = cash // ( priceTicker[ticker] / tickersPct[ticker] )
            cash = cash - unitsTickerH[SHBuy.index].mul(pTicker[SHBuy.index]).sum(axis=1).iloc[0]
            unitsTicker[SHBuy.index] = unitsTicker[SHBuy.index] + unitsTickerH[SHBuy.index]
#            moved = f"{unitsB0}{tickerB}→{tickerA}"
        C = C_K
        S=pd.Series(0, index=tickerIdx)

    # update portfolio value
    value = unitsTicker.mul(pTicker).sum(axis=1).iloc[0] + cash


    records.append({
        "date":  date,       
        "price": pTicker.copy(),    
        "units": unitsTicker.copy(),
        "value": value.copy(),      
        "S"    : S.copy()          
    })
    a=1

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


