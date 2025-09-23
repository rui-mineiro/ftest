from plotly.subplots import make_subplots
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

pio.renderers.default = "browser"


def plot_fig01(df):


    # assuming df is your DataFrame
    df["plus"]  = df[df.columns[2][0]] + df[df.columns[3][0]]
    df["mid"]   = df[df.columns[2][0]]
    df["minus"] = df[df.columns[2][0]] - df[df.columns[3][0]]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df.index, y=df["plus"],
        mode="lines", name=df.columns[2][0]+"+"+df.columns[3][0]
    ))

    fig.add_trace(go.Scatter(
        x=df.index, y=df["High"]["AAPL"],
        mode="lines", name="High"
    ))
    
    fig.add_trace(go.Scatter(
        x=df.index, y=df["mid"],
        mode="lines", name=df.columns[2][0]
    ))
    
    fig.add_trace(go.Scatter(
        x=df.index, y=df["Low"]["AAPL"],
        mode="lines", name=df.columns[1][0]
    ))
    

    fig.add_trace(go.Scatter(
        x=df.index, y=df["minus"],
        mode="lines", name=df.columns[2][0]+"-"+df.columns[3][0]
    ))
    

    fig.update_layout(
        title="MID03 ± TR03",
        xaxis_title="Date",
        yaxis_title="Value",
        template="plotly_white"
    )
    
    fig.show()