from plotly.subplots import make_subplots
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

pio.renderers.default = "browser"


def plot_fig04(df,indicator):

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.05 ,
                        row_heights=[0.4 , 0.6],
                        specs=[[{"secondary_y": True}],[{"secondary_y": True}]])
    
###  Start Prices    
    price_df   = indicator["TickerPrice"]
    for ticker in price_df.columns:
        fig.add_trace(
            go.Scatter(
                x=price_df.index,
                y=price_df[ticker],
                mode="lines",
                name=ticker
            ),
            row=1, col=1
        )
    
###  End Prices    
    

### Start Switches + Value 

    trades = indicator[indicator["TradeUnits"].sum(axis=1) > 0]["TotalPrice"]
    fig.add_trace(go.Scatter(
        x=trades.index , y=trades["ALL"],
        mode="markers+text",
        marker=dict(size=9, symbol="triangle-up", color="black"),
#        text=moved_wide["moved"], textposition="top center",
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

### END Switches + Value     
    

### Start Units x Prices  -  Buy / Sell
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
                name="#Units "+ticker+" x Price"
            ),
            row=2, col=1
        )
### End Units x Prices  -  Buy / Sell
    
#  Layout ---
    fig.update_layout(
        title="Prices by Ticker",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_white",
        yaxis_type="linear"   # logarithmic scale
    )
    
    fig.show()

