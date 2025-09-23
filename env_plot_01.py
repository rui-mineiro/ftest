from plotly.subplots import make_subplots
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

pio.renderers.default = "browser"


def plot_fig01(df):


    # assuming df is your DataFrame
    df["plus"] = df["MID03"] + df["TR03"]
    df["minus"] = df["MID03"] - df["TR03"]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df.index, y=df["plus"],
        mode="lines", name="MID03 + TR03"
    ))
    
    fig.add_trace(go.Scatter(
        x=df.index, y=df["minus"],
        mode="lines", name="MID03 - TR03"
    ))
    
    fig.update_layout(
        title="MID03 ± TR03",
        xaxis_title="Date",
        yaxis_title="Value",
        template="plotly_white"
    )
    
    fig.show()