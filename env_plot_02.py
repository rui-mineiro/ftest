from plotly.subplots import make_subplots
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

pio.renderers.default = "browser"


def plot_fig02(df,ticker):

#    df=data.copy()
#    # assuming df is your DataFrame
#    df["plus"]  = df[df.columns[2]] + df[df.columns[3]]/2
#    df["mid"]   = df[df.columns[2]]
#    df["minus"] = df[df.columns[2]] - df[df.columns[3]]/2
    
#    fig = go.Figure()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=False,
                    vertical_spacing=0 ,
                    row_heights=[0.3 ,0.7  ] ,
                    specs=[[{"secondary_y": False}],[{"secondary_y": True}]])
        

    fig.add_trace(go.Scatter(
        x=df.index, y=df[df.columns[4]],
        mode="markers", name=df.columns[4]
    ),row=1, col=1)


    fig.add_trace(go.Scatter(
        x=df.index, y=df[df.columns[5]],
        mode="markers", name=df.columns[5]
    ),row=2, col=1)


    fig.add_trace(go.Scatter(
        x=df.index, y=df[df.columns[6]],
        mode="markers", name=df.columns[6]
    ),row=2, col=1, secondary_y=True )

    # fig.update_yaxes(type="log", row=2, col=1, secondary_y=True)
    fig.update_yaxes(
    zeroline=True,
    zerolinecolor="black",
    zerolinewidth=2,   # adjust thickness if needed
    row=2, col=1, secondary_y=True
    )

    fig.update_yaxes(
    zeroline=True,
    zerolinecolor="blue",
    zerolinewidth=1,   # adjust thickness if needed
    row=2, col=1, secondary_y=False
    )

    fig.update_layout(
        title=ticker,
        template="plotly_white",
        xaxis =dict(title="Date"),
        yaxis =dict(title=df.columns[4]), 
        xaxis2=dict(title="Date"),
        yaxis2=dict(title=df.columns[5]), 
#        xaxis3=dict(title="Date"),
#        yaxis3=dict(title=df.columns[6]), 
#        xaxis4=dict(title="Date"),
#        yaxis4=dict(title=df.columns[7]), 
#        xaxis5=dict(title="Date"),
#        yaxis5=dict(title=df.columns[8]), 
#        xaxis6=dict(title="Date"),
#        yaxis6=dict(title=df.columns[9])
#        xaxis6=dict(title="Date"),
#        yaxis6=dict(title=df.columns[8]), 
#        xaxis7=dict(title="Date"),
#        yaxis7=dict(title=df.columns[9]),
    )
    

    fig.show()