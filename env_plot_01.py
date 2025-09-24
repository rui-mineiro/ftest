from plotly.subplots import make_subplots
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

pio.renderers.default = "browser"


def plot_fig01(data,ticker):

    df=data.copy()
    # assuming df is your DataFrame
    df["plus"]  = df[df.columns[2]] + df[df.columns[3]]/2
    df["mid"]   = df[df.columns[2]]
    df["minus"] = df[df.columns[2]] - df[df.columns[3]]/2
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df.index, y=df["plus"],
        mode="lines", name=df.columns[2]+"+"+df.columns[3]
    ))

    fig.add_trace(go.Scatter(
        x=df.index, y=df["High"],
        mode="lines", name="High"
    ))
    
    fig.add_trace(go.Scatter(
        x=df.index, y=df["mid"],
        mode="lines", name=df.columns[2]
    ))
    
    fig.add_trace(go.Scatter(
        x=df.index, y=df["Low"],
        mode="lines", name=df.columns[1]
    ))
    

    fig.add_trace(go.Scatter(
        x=df.index, y=df["minus"],
        mode="lines", name=df.columns[2]+"-"+df.columns[3]
    ))
    

    fig.update_layout(
        title=ticker,
        xaxis_title="Date",
        yaxis_title="Value",
        template="plotly_white"
    )
    
    fig.show()