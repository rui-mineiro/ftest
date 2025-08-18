import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- PARAMETERS ---
tickerA = "AAPL"
tickerB = "MSFT"
start_date = "2022-01-01"
end_date   = "2023-12-31"



unitsA, unitsB = 4, 4

S_K, C_K = 4, 4   # threshold and cooldown
S = 0
C = 0

# --- DOWNLOAD DATA ---
data = yf.download([tickerA, tickerB], start=start_date, end=end_date,auto_adjust=False)["Adj Close"]
data = data.dropna()



# Daily returns (as growth multipliers)
rets = data.pct_change().dropna()
rets[tickerA] = rets[tickerA]+1
rets[tickerB] = rets[tickerB]+1
rets["xAB"] = rets[tickerA] > rets[tickerB]

data["xAB"]=rets["xAB"]

# --- SIMULATION ---
records = []
pA , pB = data[tickerA].iloc[0] , data[tickerB].iloc[0]

# Initial cash for transactions
cash = unitsA*pA + unitsB*pB



for t, (date, row) in enumerate(data.iterrows(), start=1):
    pA, pB , xAB = row[tickerA], row[tickerB] , row["xAB"]

    # update units
    if xAB and unitsA > 0:
        S = min(S + 1, S_K)
    elif not xAB and unitsB > 0:
        S = max(S - 1, -S_K)

    moved = None
    if C > 0:
        C -= 1
    else:
        if S >= S_K and unitsA > 0 and cash >= (pA - pB):
            unitsA -= 1
            unitsB += 1
            cash = cash+pA-pB
            moved = f"{tickerA}→{tickerB}"
            S = 0
            C = C_K
        elif S <= -S_K and unitsB > 0 and cash >= (pB - pA):
            unitsB -= 1
            unitsA += 1
            cash = cash+pB-pA
            moved = f"{tickerB}→{tickerA}"
            S = 0
            C = C_K

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

# dynamic portfolio value (real, starting at 800)
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