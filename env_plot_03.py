from plotly.subplots import make_subplots
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

pio.renderers.default = "browser"


def plot_fig03(df, idx , ticker):

    # Generic variable for remaining rows
    rowVar = [df.columns[i] for i in idx]
    # rowVar = idx


    fig = make_subplots(
        rows=len(rowVar)+1, cols=1, shared_xaxes=True
    )

    # Row 1 traces
    for col in ["Max", "High", "Low", "Min", df.columns[4]]:
        fig.add_trace(
            go.Scatter(x=df.index, y=df[col], mode="lines", name=str(col)),
            row=1, col=1
        )


    # Add traces for rows 2..n
    for i, col in enumerate(rowVar, start=2):
        fig.add_trace(
            go.Scatter(x=df.index, y=df[col], mode="lines", name=str(col)),
            row=i, col=1
        )

    # Build axis layout
    layout_kwargs = {
        "title": ticker,
        "template": "plotly_white",
        "xaxis": dict(title="Date"),
        "yaxis": dict(title="Prices"),
    }
    for i, col in enumerate(rowVar, start=2):
        layout_kwargs[f"xaxis{i}"] = dict(title="Date")
        layout_kwargs[f"yaxis{i}"] = dict(title=col)

    fig.update_layout(**layout_kwargs)
    fig.show()