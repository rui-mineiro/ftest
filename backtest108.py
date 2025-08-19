import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random

# --- PARAMETERS ---
tickerA = "AAPL"
tickerB = "DAVV.DE" # "MSFT"
start_date = "2024-01-01"
end_date   = "2025-05-31"



cash=1000
unitsApct, unitsBpct = 0.5 , 0.5

S_K , C_K = 0 , 3 # threshold and cooldown
S = 0
C = 0

# --- DOWNLOAD DATA ---
data = yf.download([tickerA, tickerB], start=start_date, end=end_date,auto_adjust=False)["Adj Close"]
data = data.dropna()

# Normalize to start at 100 (like in queue example)
data[tickerA] = 100 * data[tickerA] / data[tickerA].iloc[0]
data[tickerB] = 100 * data[tickerB] / data[tickerB].iloc[0]



# Daily returns (as growth multipliers)
rets = data.pct_change().dropna()
rets[tickerA] = rets[tickerA]
rets[tickerB] = rets[tickerB]
rets["xAB"] = rets[tickerA] > rets[tickerB]

data["retA"]=rets[tickerA]
data["retB"]=rets[tickerB]



data["xAB"]=rets["xAB"]
data=data.iloc[1:]

# --- SIMULATION ---
records = []
value = 0
pA , pB = data[tickerA].iloc[0] , data[tickerB].iloc[0]



# Initial cash for transactions
unitsA= cash // ( pA * 2  / unitsApct )
unitsB= cash // ( pB * 2  / unitsBpct )
cash  = cash - unitsA*pA - unitsB*pB

for t, (date, row) in enumerate(data.iterrows(), start=1):
    pA, pB, retA, retB, xAB = row[tickerA], row[tickerB], row["retA"], row["retB"], row["xAB"]

    
    # update score
    S += retA - retB




    moved = None
    if C > 0:
        C -= 1
    else:
        if S >= S_K and unitsB>1 and value >= pA:
            unitsB0=random.randint(0, int(unitsB)-1)
            unitsB  -= unitsB0
            cash    += pB*unitsB0
            unitsA0  = cash // pA
            unitsA  += unitsA0
            cash    -= unitsA0*pA
            moved = f"{tickerB}→{tickerA}"
            C = C_K
            S = 0
        elif S <= -S_K and unitsA>1 and value >= pB:
            unitsA0=random.randint(0, int(unitsA)-1)
            unitsA  -= unitsA0
            cash    += pA*unitsA0
            unitsB0  = cash // pB
            unitsB  += unitsB0
            cash    -= unitsB0*pB
            moved = f"{tickerA}→{tickerB}"
            C = C_K
            S = 0



    # update portfolio value
    value = unitsA * pA + unitsB * pB + cash

    records.append({
        "date": date,
        "priceA": pA, "priceB": pB,
        "unitsA": unitsA, "unitsB": unitsB,
        "value": value, "moved": moved
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