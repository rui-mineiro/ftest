from plotly.subplots import make_subplots
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

pio.renderers.default = "browser"


def plot_fig00(df):

    # --- Step 0: Create subplot (1 row, 1 col here, but extendable) ---
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.05 ,
                        row_heights=[0.4 , 0.6],
                        specs=[[{"secondary_y": True}],[{"secondary_y": True}]])
    
    
    # --- Step 1: Expand df["price"] into wide format ---
    price_wide = df["price"].apply(pd.Series)
    price_wide["date"] = df["date"].values   # attach the real dates
    price_df = price_wide.melt(id_vars="date", var_name="ticker", value_name="price")
    for ticker in price_df["ticker"].unique():
        plotData = price_df[price_df["ticker"] == ticker]
        fig.add_trace(
            go.Scatter(
                x=plotData["date"],
                y=plotData["price"],
                mode="lines",
                name=ticker
            ),
            row=1, col=1
        )
    
    
    
    S_wide = df["S"].apply(pd.Series)
    S_wide["date"] = df["date"].values   # attach the real dates
    S_df = S_wide.melt(id_vars="date", var_name="ticker", value_name="Score")
    for ticker in S_df["ticker"].unique():
        plotData = S_df[S_df["ticker"] == ticker]
        fig.add_trace(
            go.Scatter(
                x=plotData["date"],
                y=plotData["Score"],
                mode="lines",
                name="S"+ticker
            ),
            row=1, col=1
        )
    
    
    moved_wide = df["moved"].apply(pd.Series)
    if len(moved_wide.columns) == 1:
        moved_wide["date"] = df["date"].values
        moved_wide=moved_wide.dropna()
        moved_wide.columns=['moved','date']
        
        fig.add_trace(go.Scatter(
            x=moved_wide["date"], y=[df.loc[df["date"]==d, "value"].values[0] for d in moved_wide["date"]],
            mode="markers+text",
            marker=dict(size=9, symbol="triangle-up", color="black"),
            text=moved_wide["moved"], textposition="top center",
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
    
    
    units_wide = df["units"].apply(pd.Series)
    price_wide = df["price"].apply(pd.Series)
    unitsValue_wide=price_wide.mul(units_wide)
    
    unitsValue_wide["date"] = df["date"].values   # attach the real dates
    units_df = unitsValue_wide.melt(id_vars="date", var_name="ticker", value_name="units")
    for ticker in units_df["ticker"].unique():
        plotData = units_df[units_df["ticker"] == ticker]
        fig.add_trace(
            go.Scatter(
                x=plotData["date"],
                y=plotData["units"],
                mode="lines",
                name=ticker
            ),
            row=2, col=1
        )
    
    
    # --- Step 4: Layout ---
    fig.update_layout(
        title="Prices by Ticker",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_white",
        yaxis_type="linear"   # logarithmic scale
    )
    
    fig.show()

