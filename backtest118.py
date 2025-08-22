import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random
import numpy as np

# --- PARAMETERS ---

tickerIdx     = ["AAPL", "MSFT" , "DAVV.DE"  ]
tickerPct     = [ 0.3 , 0.3 , 0.4 ]

start_date = "2024-01-01"
end_date   = "2025-05-31"

cash=1000

S_H, S_B , S_S , C_K = -1/100, 1/100, -1/100, 5 # threshold and cooldown


tickers    = pd.DataFrame( {'ticker' : tickerIdx })
tickersPct = pd.DataFrame( [ tickerPct ], columns=tickerIdx)

S = pd.Series(0 , index=tickerIdx)
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
unitsTicker    = pd.Series()   # Tickers Units
unitsTickerH   = pd.Series()  # Tickers High >   S_K
unitsTickerL   = pd.Series()  # Tickers Low  <  -S_K
unitsTickerRnd = pd.Series()
value = 0

priceTicker = data.iloc[0]


unitsTicker = cash // (priceTicker.div(tickerPct)*2)



# Initial cash for transactions
cash  = cash - unitsTicker.mul(priceTicker).sum()

for t, index in enumerate(data.index, start=1):
    pTicker = data.loc[index]
    rTicker = rets.loc[index]
    date    = index

    # update score
    S += rTicker

    moved = pd.Series(None, index=tickerIdx,dtype=str)

    if C > 0:
        C -= 1
    else:
        nSHold  = S[S > S_H]
        if nSHold.empty:         
            # Sell all units of all the tickers
            SHold  = S[S <= S_H]
            tickersL=SHold.index
            unitsTickerL = unitsTicker
            for ticker in tickersL:     
                moved[ticker]        = "-"+str(unitsTickerL[ticker])+"#"+ticker
            cash = cash - unitsTickerL.mul(pTicker[tickersL])[tickersL].sum()
            unitsTicker[tickersL] = unitsTicker[tickersL] + unitsTickerL[tickersL]



                        
        else:
            SBuy   = S[S >  S_B]
            if not SBuy.empty:               # If there are some tickers above S_B 
                nSBuy  = S[S <= S_B]

                if not nSBuy.empty:          # Sell random units of the ticker bellow S_B with lowest score
                    unitsTickerRnd  = unitsTicker.apply(lambda x: np.random.randint(0, int(x) + 1))
                    tickersL=[nSBuy.idxmin()]
                    unitsTickerL = unitsTickerRnd[tickersL]
                    cash = cash + unitsTickerL.mul(pTicker[tickersL])[tickersL].sum()
                    unitsTicker[tickersL] = unitsTicker[tickersL] - unitsTickerL[tickersL]
                    for ticker in tickersL:
                        moved[ticker]    = "-"+str(unitsTickerL[ticker])+"#"+ticker
                
                # Buy random units of all the tickers above S_B
                unitsTickerRnd  = unitsTicker.apply(lambda x: np.random.randint(0, int(x) + 1))
                tickersH=SBuy.index
                unitsTickerH    = unitsTickerRnd[tickersH]
                for ticker in tickersH:     
                    moved[ticker]        = "+"+str(unitsTickerH[ticker])+"#"+ticker
                cash = cash - unitsTickerH.mul(pTicker[tickersH])[tickersH].sum()
                unitsTicker[tickersH] = unitsTicker[tickersH] + unitsTickerH[tickersH]

            SSell  = S[S <= S_S]
            if not SSell.empty:              # If there are some tickers bellow S_S


                # Sell random units of all the ticker bellow S_S # Ultima alteração
                unitsTickerRnd  = unitsTicker.apply(lambda x: np.random.randint(0, int(x) + 1))
                unitsTickerL    = pd.Series(0, index=tickerIdx)
                tickersL=SSell.index
                unitsTickerL = unitsTickerRnd[tickersL]
                for ticker in tickersL:     
                    moved[ticker]        = "-"+str(unitsTickerL[ticker])+"#"+ticker
                cash = cash - unitsTickerL.mul(pTicker[tickersL])[tickersL].sum()
                unitsTicker[tickersL] = unitsTicker[tickersL] + unitsTickerL[tickersL]

                nSSell = S[S >  S_S]         
                if not nSSell.empty:          # Buy random units of the ticker above S_S with highest score
                    unitsTickerRnd  = unitsTicker.apply(lambda x: np.random.randint(0, int(x) + 1))
                    unitsTickerH    = pd.Series(0, index=tickerIdx)
                    tickersH=[nSSell.idxmax()]
                    unitsTickerH = unitsTickerRnd[tickersH]
                    cash = cash + unitsTickerH.mul(pTicker[tickersH])[tickersH].sum()
                    unitsTicker[tickersH] = unitsTicker[tickersH] - unitsTickerH[tickersH]
                    for ticker in tickersH:
                        moved[ticker]    = "+"+str(unitsTickerH[ticker])+"#"+ticker

            C = C_K
            S=pd.Series(0, index=tickerIdx)

    # update portfolio value
    value = unitsTicker.mul(pTicker).sum() + cash


    records.append({
        "date":  date,       
        "price": pTicker.copy(),    
        "units": unitsTicker.copy(),
        "value": value.copy(),      
        "S"    : S.copy(),
        "moved": moved.copy()
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


moved_wide = df["moved"].apply(pd.Series)
moved_wide["date"] = df["date"].values
# moved_wide=moved_wide.dropna()
moved_df = moved_wide.melt(id_vars="date", var_name="ticker", value_name="moved")
moved_df = moved_df.dropna()
for ticker in moved_df["ticker"].unique():
    data = moved_df[moved_df["ticker"] == ticker]
    fig.add_trace(go.Scatter(
        x=moved_df["date"], y=[df.loc[df["date"]==d, "value"].values[0] for d in moved_df["date"]],
        mode="markers+text",
        marker=dict(size=9, symbol="triangle-up", color="black"),
        text=moved_df["moved"], textposition="top center",
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


