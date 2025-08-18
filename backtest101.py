import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

# --- PARAMETERS ---
tickerA = "AAPL"
tickerB = "DAVV.DE" # "DFEN.DE" # "MSFT"
start_date = "2024-01-01"
end_date   = "2025-05-31"

initial_investment = 10000
# n_units = 8
unitsA, unitsB = 3, 3

S_K , C_K = 2, 2   # Score Threshold and Cooldown Threshold
S = 0
C = 0

# --- DOWNLOAD DATA ---
data = yf.download([tickerA, tickerB], start=start_date, end=end_date)["Close"]
data = data.dropna()

# Normalize to start at 100 (like in queue example)
priceA = 100 * data[tickerA] / data[tickerA].iloc[0]
priceB = 100 * data[tickerB] / data[tickerB].iloc[0]

# --- SIMULATION ---
records = []
prevA, prevB = priceA.iloc[0], priceB.iloc[0]

for t, (pa, pb) in enumerate(zip(priceA, priceB), start=1):
    advA = pa > prevA
    advB = pb > prevB

    # Update score
    if advB:
        S = min(S + 1, S_K)
    if advA:
        S = max(S - 1, -S_K)

    moved = None
    if C > 0:
        C -= 1
    else:
        if S >= S_K and unitsA > 0:
            unitsA -= 1
            unitsB += 1
            moved = f"{tickerA}→{tickerB}"
            S = 0
            C = C_K
        elif S <= -S_K and unitsB > 0:
            unitsB -= 1
            unitsA += 1
            moved = f"{tickerB}→{tickerA}"
            S = 0
            C = C_K

    # Portfolio value
    value = unitsA * initial_investment * (pa / 100) + \
            unitsB * initial_investment * (pb / 100)

    records.append({
        "date": priceA.index[t-1],
        "priceA": pa, "priceB": pb,
        "unitsA": unitsA, "unitsB": unitsB,
        "value": value, "moved": moved
    })

    prevA, prevB = pa, pb

df = pd.DataFrame(records)

# --- PLOTLY CHART ---
fig = go.Figure()

# Stock A & B
fig.add_trace(go.Scatter(x=df["date"], y=df["priceA"],
                         mode="lines", name=tickerA, line=dict(color="blue")))
fig.add_trace(go.Scatter(x=df["date"], y=df["priceB"],
                         mode="lines", name=tickerB, line=dict(color="red")))

# Portfolio value (secondary y-axis)
fig.add_trace(go.Scatter(x=df["date"], y=df["value"],
                         mode="lines", name="Portfolio Value",
                         line=dict(color="green", dash="dot"), yaxis="y2"))

# Mark moves
moves = df.dropna(subset=["moved"])
fig.add_trace(go.Scatter(
    x=moves["date"], y=moves["priceA"],
    mode="markers+text",
    marker=dict(size=9, symbol="triangle-up", color="black"),
    text=moves["moved"], textposition="top center",
    name="Switches"
))

# Layout with dual y-axis
fig.update_layout(
    title=f"Trading Simulation: {tickerA} vs {tickerB}",
    xaxis=dict(title="Date"),
    yaxis=dict(title="Stock Price (normalized)"),
    yaxis2=dict(title="Portfolio Value", overlaying="y", side="right"),
    legend=dict(x=0.02, y=0.98)
)

fig.show()
