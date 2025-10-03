from plotly.subplots import make_subplots
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

pio.renderers.default = "browser"


def plot_fig03(df,indicator, indicatorSIG , ticker):

    # Generic variable for remaining rows
#    rowVar = [indicator.columns[i] for i in idx]
    suffix = indicatorSIG[0][-2:]
    maskRow1 = indicator.columns.get_level_values("Indicator").isin(["Max"+suffix,"High", "Low","Min"+suffix,"MID"])
    maskRow2 = indicator.columns.get_level_values("Indicator").isin(["#DMID0"+suffix,"TR0"+suffix,"#RLTR0"+suffix,"#MMR0"+suffix])
    rowVar = indicator.loc[:, maskRow1]
    # rowVar = idx



    fig = make_subplots(
        rows=len(rowVar.columns)+1, cols=1, shared_xaxes=True
    )

    # Row 1 traces
    for col in rowVar.columns:
        fig.add_trace(
            go.Scatter(x=rowVar.index, y=rowVar[col], mode="lines", name=str(col)),
            row=1, col=1
        )


    # Add traces for rows 2..n
    rowVar = indicator.loc[:, maskRow2]
    for i, col in enumerate(rowVar.columns, start=2):
        fig.add_trace(
            go.Scatter(x=rowVar.index, y=rowVar[col], mode="lines", name=str(col)),
            row=i, col=1
        )
        if col == indicatorSIG[0] and df['moved'].notna().any():
            moved_wide = df["moved"].apply(pd.Series)
            moved_wide["date"] = df["date"].values
            moved_wide=moved_wide.dropna()
            moved_wide.columns=["moved","date"]
            indicatorSIGidx = pd.DatetimeIndex(moved_wide["date"])
            for d in indicatorSIGidx:
                fig.add_vline(
                    x=d,
                    line_width=2,
                    line_dash="dash",
                    line_color="red",
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