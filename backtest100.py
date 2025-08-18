import random
import pandas as pd
import plotly.graph_objects as go

# Parameters
pA, pB = 0.5, 0.5    # probability of each stock advancing
steps = 1000
initial_investment = 100
n_units = 8
K, H = 3, 3            # threshold & cooldown

# Prices
priceA = 100
priceB = 100

# Allocations (start 4+4)
unitsA = 4
unitsB = 4

# Score & cooldown
S = 0
cooldown = 0

# History
records = []

for t in range(1, steps + 1):
    # Random advances
    advA = random.random()*2 < pA
    advB = random.random()   < pB
    if advA: priceA += 1
    if advB: priceB += 1

    # Update score: positive = B faster, negative = A faster
    if advB: 
        S = min(S + 1, K)
    if advA: 
        S = max(S - 1, -K)

    if cooldown > 0:
        cooldown -= 1

    moved = None
    # Decision: move 1 unit when threshold reached
    if cooldown == 0:
        if S >= K and unitsA > 0:  # shift A -> B
            unitsA -= 1
            unitsB += 1
            moved = f"t{t}: A→B"
            S = 0
            cooldown = H
        elif S <= -K and unitsB > 0:  # shift B -> A
            unitsB -= 1
            unitsA += 1
            moved = f"t{t}: B→A"
            S = 0
            cooldown = H

    # Portfolio value
    value = unitsA * initial_investment * (priceA / 100) + \
            unitsB * initial_investment * (priceB / 100)

    records.append({
        "t": t, "priceA": priceA, "priceB": priceB,
        "unitsA": unitsA, "unitsB": unitsB,
        "value": value, "moved": moved
    })

# Convert to DataFrame
df = pd.DataFrame(records)

# --- Plotly Charts ---
fig = go.Figure()

# Stock A and B
fig.add_trace(go.Scatter(x=df["t"], y=df["priceA"],
                         mode="lines", name="Stock A", line=dict(color="blue")))
fig.add_trace(go.Scatter(x=df["t"], y=df["priceB"],
                         mode="lines", name="Stock B", line=dict(color="red")))

# Portfolio value (secondary axis)
fig.add_trace(go.Scatter(x=df["t"], y=df["value"],
                         mode="lines+markers", name="Portfolio Value",
                         line=dict(color="green", dash="dot"), yaxis="y2"))

# Mark moves
moves = df.dropna(subset=["moved"])
fig.add_trace(go.Scatter(
    x=moves["t"], y=moves["priceA"],
    mode="markers+text",
    marker=dict(size=10, symbol="triangle-up", color="black"),
    text=moves["moved"], textposition="top center",
    name="Moves"
))

# Layout with two y-axes
fig.update_layout(
    title="Trading Simulation: Switching Between Two Stocks",
    xaxis=dict(title="Time"),
    yaxis=dict(title="Stock Price"),
    yaxis2=dict(title="Portfolio Value", overlaying="y", side="right"),
    legend=dict(x=0.02, y=0.98)
)

fig.show()
