import yfinance as yf
import vectorbt as vbt
import pandas as pd
import numpy as np
import plotly.graph_objects as go

symbol = "SPY"
start_date = "2015-01-01"
end_date = "2025-01-01"
monthly_budget = 100  # € por mês

# Baixar dados
data = yf.download(symbol, start=start_date, end=end_date, auto_adjust=True)
price = pd.DataFrame(data["Close"])
price.columns = [symbol]

# Gerar datas alvo: início de cada mês
# First trading day of each month
#  https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
target_dates = price.resample('MS').first().index

# Mapear para datas válidas no índice de preços
valid_buy_dates = price.index[price.index.get_indexer(target_dates, method='bfill')]
valid_buy_dates = valid_buy_dates.dropna()

# Criar sinal de entrada
entries = pd.DataFrame(False, index=price.index, columns=[symbol])
entries.loc[valid_buy_dates] = True

# Calcular tamanhos fracionários: 100€ / preço no dia
size = pd.DataFrame(0.0, index=price.index, columns=[symbol])
size.loc[valid_buy_dates, symbol] = monthly_budget / price.loc[valid_buy_dates, symbol]

# Backtest
pf = vbt.Portfolio.from_signals(
    close=price,
    entries=entries,
    exits=None,
    size=size,
    freq='1D',
    accumulate=True,
    init_cash=0
)



total_invested = size.loc[valid_buy_dates, symbol].mul(price.loc[valid_buy_dates, symbol]).sum()
final_value = pf[symbol].asset_value().iloc[-1]
cumulative_return = (final_value - total_invested) / total_invested

print(f"Total investido: €{total_invested:.2f}")
print(f"Valor final: €{final_value:.2f}")
print(f"Retorno acumulado: {cumulative_return:.2%}")

fig = go.Figure()
asset_value = pf[symbol].asset_value()


fig.add_trace(go.Scatter(x=asset_value.index,
                         y=asset_value.values,
                         mode='lines',
                         name='Asset Value',
                         line=dict(color='blue')
                         ))

fig.add_trace(go.Scatter(x=price[symbol].index,
                         y=price[symbol].values*10,
                         mode='lines',
                         name='Price',
                         line=dict(color='green')
                         ))


fig.add_trace(go.Scatter(x=asset_value[entries[symbol]].index,
                         y=asset_value[entries[symbol]].values,
                         mode='markers',
                         name='Compras',
                         marker=dict(color='green', size=8, symbol='triangle-up')
))

fig.update_layout(title="Valor do portfólio",
                  xaxis_title="Data",
                  yaxis_title="€")
fig.show()
