import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random

# --- PARAMETERS ---

ticker     = ["AAPL", "MSFT" , "DAVV.DE"  ]
tickerPct = [ 0.3 , 0.3 , 0.4 ]

tickers = pd.DataFrame( {'ticker' : ticker })

tickersPct = pd.DataFrame( [ tickerPct ], columns=ticker)

tickerB = "MSFT" # "DAVV.DE" # "MSFT"
start_date = "2024-01-01"
end_date   = "2025-05-31"



cash=1000


S_K , C_K = 0.6 , 5 # threshold and cooldown
S = 0
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
unitsTicker = pd.DataFrame()
value = 0

priceTicker = data.iloc[0]


for ticker in tickers["ticker"]:
    unitsTicker[ticker] = cash // ( priceTicker[ticker] * 2 / tickersPct[ticker] )



# Initial cash for transactions
cash  = cash - unitsTicker.mul(priceTicker).sum(axis=1)

for t, index in enumerate(data.index, start=1):
    pTicker = data.loc[index]
    rTicker = rets.loc[index]
    


# for t, (date, row) in enumerate(data.iterrows(), start=1):
#     pA, pB, retA, retB = row[tickerA], row[tickerB], row["retA"], row["retB"]

    
    # update score
    S += (retA - retB)*(pA*unitsA - pB*unitsB)

    moved = None
    if C > 0:
        C -= 1
    else:
        unitsB0=random.randint(0, int(unitsB)-1)
        unitsA0=random.randint(0, int(unitsA)-1)
        if S >= S_K and unitsB>1 and value >= pA and unitsB0>0:
            unitsB  -= unitsB0
            cash    += pB*unitsB0
            unitsA0  = cash // pA
            unitsA  += unitsA0
            cash    -= unitsA0*pA
            moved = f"{unitsB0}{tickerB}→{tickerA}"
            C = C_K
        elif S <= -S_K and unitsA>1 and value >= pB  and unitsA0>0:
            unitsA  -= unitsA0
            cash    += pA*unitsA0
            unitsB0  = cash // pB
            unitsB  += unitsB0
            cash    -= unitsB0*pB
            moved = f"{unitsA0}{tickerA}→{tickerB}"
            C = C_K
        S = 0




    # update portfolio value
    value = unitsA * pA + unitsB * pB + cash

    records.append({
        "date": date,
        "priceA": pA, "priceB": pB,
        "unitsA": unitsA, "unitsB": unitsB,
        "value": value, "moved": moved,
        "S" : S
    })

df = pd.DataFrame(records)

# --- PLOTLY ---
fig = make_subplots(specs=[[{"secondary_y": True}]])

# cumulative buy & hold performance (normalized to 100)

fig.add_trace(go.Scatter(
    x=df["date"], y=df["priceA"],
    mode="lines", name=f"{tickerA} (Buy & Hold)",
    line=dict(color="blue")
), secondary_y=False)

fig.add_trace(go.Scatter(
    x=df["date"], y=df["priceB"],
    mode="lines", name=f"{tickerB} (Buy & Hold)",
    line=dict(color="red")
), secondary_y=False)

# dynamic portfolio value
fig.add_trace(go.Scatter(
    x=df["date"], y=df["value"],
    mode="lines", name="Dynamic Portfolio",
    line=dict(color="green", dash="dot")
), secondary_y=True)# dynamic portfolio value

fig.add_trace(go.Scatter(
    x=df["date"], y=df["S"],
    mode="lines", name="Score",
    line=dict(color="green", dash="dot")
), secondary_y=True)

# mark switch points (on portfolio axis)
moves = df.dropna(subset=["moved"])
fig.add_trace(go.Scatter(
    x=moves["date"], y=[df.loc[df["date"]==d, "value"].values[0] for d in moves["date"]],
    mode="markers+text",
    marker=dict(size=9, symbol="triangle-up", color="black"),
    text=moves["moved"], textposition="top center",
    name="Switches"
), secondary_y=True)

# layout
fig.update_layout(
    title=f"Trading Simulation using Daily Returns: {tickerA} vs {tickerB}",
    xaxis=dict(title="Date"),
    legend=dict(x=0.02, y=0.98)
)

fig.update_yaxes(title_text="Buy & Hold (normalized, start=100)", secondary_y=False)
fig.update_yaxes(title_text="Dynamic Portfolio Value (USD)", secondary_y=True)

fig.show()