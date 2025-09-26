from plotly.subplots import make_subplots
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

pio.renderers.default = "browser"


def plot_fig01(df,ticker):

#    df=data.copy()
#    # assuming df is your DataFrame
#    df["plus"]  = df[df.columns[2]] + df[df.columns[3]]/2
#    df["mid"]   = df[df.columns[2]]
#    df["minus"] = df[df.columns[2]] - df[df.columns[3]]/2
    
#    fig = go.Figure()
    fig = make_subplots(rows=5, cols=1, shared_xaxes=True,
                    vertical_spacing=0.05 ,
                    row_heights=[0.4 ,0.1 ,0.1 ,0.2 ,0.1 ]
                    )
        

    fig.add_trace(go.Scatter(
        x=df.index, y=df["Max"],
        mode="lines", name="Max"
    ),row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df.index, y=df["High"],
        mode="lines", name="High"
    ),row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df.index, y=df[df.columns[5]],
        mode="lines", name=df.columns[5]
    ),row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=df.index, y=df["Low"],
        mode="lines", name="Low"
    ),row=1, col=1)
    


    fig.add_trace(go.Scatter(
        x=df.index, y=df["Min"],
        mode="lines", name="Min"
    ),row=1, col=1)


    fig.add_trace(go.Scatter(
        x=df.index, y=df[df.columns[6]],
        mode="lines", name=df.columns[6]
    ),row=2, col=1)

    fig.add_trace(go.Scatter(
        x=df.index, y=df[df.columns[7]],
        mode="lines", name=df.columns[7]
    ),row=3, col=1)

    fig.add_trace(go.Scatter(
        x=df.index, y=df[df.columns[8]],
        mode="lines", name=df.columns[8]
    ),row=4, col=1)


    fig.add_trace(go.Scatter(
        x=df.index, y=df[df.columns[9]],
        mode="lines", name=df.columns[9]
    ),row=5, col=1)


    fig.update_layout(
        title=ticker,
        template="plotly_white",
        xaxis =dict(title="Upper axis label"),
        yaxis =dict(title="Prices"), 
        xaxis2=dict(title="Date"),
        yaxis2=dict(title=df.columns[6]), 
        xaxis3=dict(title="Date"),
        yaxis3=dict(title=df.columns[7]), 
        xaxis4=dict(title="Date"),
        yaxis4=dict(title=df.columns[8]), 
        xaxis5=dict(title="Date"),
        yaxis5=dict(title=df.columns[9]) 
    )
    

    fig.show()