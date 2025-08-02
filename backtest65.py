import vectorbt as vbt
import plotly.graph_objects as go

price = vbt.YFData.download('DAVV.DE', period='1y').get('Close') # Added a period for better data range

pf = vbt.Portfolio.from_holding(price, init_cash=100)
# print(pf.total_profit()) # No need to print this twice

fast_ma = vbt.MA.run(price, 3)
slow_ma = vbt.MA.run(price, 10)
entries = fast_ma.ma_crossed_above(slow_ma)
exits = fast_ma.ma_crossed_below(slow_ma)

pf = vbt.Portfolio.from_signals(price, entries, exits, init_cash=100)
# print(pf.total_profit())

# Plotting the entry and exit points
fig = pf.plot()

# You can add the moving averages to the plot for context
fig.add_trace(go.Scatter(x=price.index, y=fast_ma.ma, name='Fast MA', line=dict(color='orange')))
fig.add_trace(go.Scatter(x=price.index, y=slow_ma.ma, name='Slow MA', line=dict(color='purple')))

fig.show()

