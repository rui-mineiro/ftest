import vectorbt as vbt
import numpy as np
import itertools
from datetime import datetime, timedelta
import re
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def download_price_with_next(symbol, startdate, enddate):
    """
    Faz download do histórico e inclui o 'Open' do próximo dia de negociação.
    Retorna: price (Close histórico), price_next (Close + próximo dia com preço de abertura).
    """
    # Pega um pouco além do enddate para garantir que pegamos o próximo dia de mercado
    price_data = vbt.YFData.download(symbol, start=startdate, end=enddate)
    close = price_data.get("Close")
    open_ = price_data.get("Open")

    # Histórico até enddate
    # price_hist = close.loc[close.index <= enddate]
    price_hist = close.loc[close.index]

    # Próximo dia de mercado
    # next_day_idx = close.index[close.index > enddate][0]
    # next_open_price = open_.loc[next_day_idx]

    next_day_idx=close.index[-1]+timedelta(days=1)

    user_input = input(f"Enter today price (default is yesterday , {close.index[-1].strftime('%Y-%m-%d')} ,  Close price of $ {price_hist.iloc[-1]:.2f}: ").strip()  

    if user_input == "":
        next_open_price = price_hist.iloc[-1]
    else:
        next_open_price = float(user_input)
    
    print(f"Using value: $ {next_open_price:.2f}")
    

    # Adiciona próximo dia ao histórico usando preço de abertura
    price_with_next = price_hist.copy()
    price_with_next.loc[next_day_idx] = next_open_price

    return price_hist, price_with_next

def find_best_params(price,maLimits):
    """
    Faz grid search para encontrar os melhores parâmetros (b_f, b_s, s_f, s_s).
    Retorna best_label (string) e pf_all (portfólio de todos os testes).
    """

    b_fastLL=maLimits[0]
    b_fastLH=maLimits[1]
    b_slowLL=maLimits[2]
    b_slowLH=maLimits[3]    
    s_fastLL=maLimits[4]
    s_fastLH=maLimits[5]
    s_slowLL=maLimits[6]    
    s_slowLH=maLimits[7]    


    b_fast_windows = np.arange(b_fastLL, b_fastLH)
    b_slow_windows = np.arange(b_slowLL, b_slowLH)
    s_fast_windows = np.arange(s_fastLL, s_fastLH)
    s_slow_windows = np.arange(s_slowLL, s_slowLH)

    valid_combinations = [
        (b_f, b_s, s_f, s_s)
        for b_f, b_s, s_f, s_s in itertools.product(b_fast_windows, b_slow_windows, s_fast_windows, s_slow_windows)
        if (s_f < s_s and b_f < b_s)
    ]
    print(f"  {len(valid_combinations)} combinações válidas")

    b_fast_list, b_slow_list, s_fast_list, s_slow_list = zip(*valid_combinations)

    # Médias móveis
    b_fast_ma = vbt.MA.run(price, window=b_fast_list).ma
    b_slow_ma = vbt.MA.run(price, window=b_slow_list).ma
    s_fast_ma = vbt.MA.run(price, window=s_fast_list).ma
    s_slow_ma = vbt.MA.run(price, window=s_slow_list).ma

    labels = [f"bfast={b_f}_bslow={b_s}_sfast={s_f}_sslow={s_s}"
              for b_f, b_s, s_f, s_s in valid_combinations]
    for ma_df in [b_fast_ma, b_slow_ma, s_fast_ma, s_slow_ma]:
        ma_df.columns = labels

    # Sinais
    raw_entries = (b_fast_ma > b_slow_ma)
    raw_exits = (s_fast_ma < s_slow_ma)
    entries, exits = raw_entries.vbt.signals.clean(raw_exits)

    # Backtest
    price_broadcasted = price.vbt.tile(len(valid_combinations))
    price_broadcasted.columns = labels

    pf_all = vbt.Portfolio.from_signals(
        price_broadcasted, entries, exits,
        init_cash=10000, fees=0.001
    )

    total_profits = pf_all.total_profit()
    best_label = total_profits.idxmax()
    best_profit = total_profits.max()
    
    print(f"Dados entre as datas {entries.index[0].strftime('%Y-%m-%d')} e {entries.index[-1].strftime('%Y-%m-%d')}")
    print(f"Janela optima: {best_label} | Lucro: €{best_profit:.2f}")

    return best_label

def get_next_signals(price_next, mas):
    """
    Calcula sinais de compra/venda para um conjunto de parâmetros.
    Retorna: entries, exits, b_fast_ma, b_slow_ma.
    """
    b_fast, b_slow, s_fast, s_slow = mas
    b_fast_ma = vbt.MA.run(price_next, window=b_fast).ma
    b_slow_ma = vbt.MA.run(price_next, window=b_slow).ma
    s_fast_ma = vbt.MA.run(price_next, window=s_fast).ma
    s_slow_ma = vbt.MA.run(price_next, window=s_slow).ma

    raw_entries = (b_fast_ma > b_slow_ma)
    raw_exits   = (s_fast_ma < s_slow_ma)
    entries_next, exits_next = raw_entries.vbt.signals.clean(raw_exits)

    return entries_next, exits_next, b_fast_ma, b_slow_ma, s_fast_ma, s_slow_ma

def plot_strategy(symbol, price, entries, exits, b_fast_ma, b_slow_ma, b_f, b_s, s_fast_ma, s_slow_ma, s_f, s_s, pf_best, signal_text):
                  
    """
    Plota dois gráficos: Preço + sinais e Lucro acumulado.
    """
    fig = make_subplots(rows=3, cols=2, shared_xaxes=True,
                        vertical_spacing=0.05 ,
                        row_heights=[0.4 , 0.3, 0.3 ],
                        column_widths=[0.4, 0.6],
                        specs=[[{"type": "scatter"},       None                         ],
                               [{"type": "scatter"}, {"type":  "scatter", "rowspan": 2} ],
                               [{"type": "scatter"},       None                         ]],
                        subplot_titles=("Lucro Acumulado", "Compra", "Compra/Venda" , "Venda" ))


    # Painel 2 - Price
    fig.add_trace(go.Scatter(x=price.index, y=price.values,
                             mode='lines', name='Preço', line=dict(color='black')), row=2, col=2)
    fig.add_trace(go.Scatter(x=[price.index[-1]], y=[price.iloc[-1]],
                         mode='markers+text', text=[signal_text], textposition='top left',
                         marker=dict(color='purple', size=14, symbol='star'),
                         name='Próximo dia'), row=2, col=2)
    fig.add_trace(go.Scatter(x=entries.index[entries], y=price[entries],
                             mode='markers', name='Compra', marker=dict(color='green', size=10, symbol='triangle-up')),   row=2, col=2)                         
    fig.add_trace(go.Scatter(x=exits.index[exits], y=price[exits],
                             mode='markers', name='Venda' , marker=dict(color='red'  , size=10, symbol='triangle-down')), row=2, col=2)                             


    # Painel 1 - Buy
    fig.add_trace(go.Scatter(x=price.index, y=price.values,
                         mode='lines', name='Preço', line=dict(color='black')), row=2, col=1)
    fig.add_trace(go.Scatter(x=price.index, y=b_fast_ma.values,
                             mode='lines', name=f'MA Rápida ({b_f})', line=dict(color='rgb(60, 179, 113)')), row=2, col=1)
    fig.add_trace(go.Scatter(x=price.index, y=b_slow_ma.values,
                             mode='lines', name=f'MA Lenta ({b_s})', line=dict(color='rgb(0, 100, 0)')), row=2, col=1)
    fig.add_trace(go.Scatter(x=entries.index[entries], y=price[entries],
                             mode='markers', name='Compra', marker=dict(color='green', size=10, symbol='triangle-up')), row=2, col=1)


    # Painel 3 - Sell
    fig.add_trace(go.Scatter(x=price.index, y=price.values,
                             mode='lines', name='Preço', line=dict(color='black')), row=3, col=1)
    fig.add_trace(go.Scatter(x=price.index, y=s_fast_ma.values,
                             mode='lines', name=f'MA Rápida ({s_f})', line=dict(color='rgb(255, 99, 71)')), row=3, col=1)
    fig.add_trace(go.Scatter(x=price.index, y=s_slow_ma.values,
                             mode='lines', name=f'MA Lenta ({s_s})', line=dict(color='rgb(178, 34, 34)')), row=3, col=1)
    fig.add_trace(go.Scatter(x=exits.index[exits], y=price[exits],
                             mode='markers', name='Venda', marker=dict(color='red', size=10, symbol='triangle-down')), row=3, col=1)


    # Painel 5
    fig.add_trace(go.Scatter(x=pf_best.value().index, y=pf_best.value().values,
                             mode='lines', name='Lucro acumulado', line=dict(color='green')), row=1, col=1)

    fig.update_layout(title=f"Estratégia para {symbol} — {signal_text}",
                      height=800,
                      legend=dict(x=0.45, y=1.15, bgcolor='rgba(255,255,255,0)'))

    fig.show()


# ============================================================
# LOOP PARA VÁRIOS SÍMBOLOS
# ============================================================

symbol = 'DFEN.DE'  # ['SPPW.DE', 'DAVV.DE']  # podes meter quantos quiseres


b_fastLL=1
b_fastLH=15

b_slowLL=10
b_slowLH=40

s_fastLL=1
s_fastLH=15

s_slowLL=10
s_slowLH=40

maLimits= [ b_fastLL , b_fastLH , b_slowLL , b_slowLH , s_fastLL , s_fastLH , s_slowLL , s_slowLH ]

# startdate = datetime(2023, 4, 10)
backdays=0
periodAnalysis=30*6

startdate = datetime.today()-timedelta(days=periodAnalysis)-timedelta(days=backdays)
enddate   = datetime.today()-timedelta(days=backdays)
startdate = pd.Timestamp(startdate, tz="UTC")
enddate   = pd.Timestamp(enddate, tz="UTC")

print(f"\n=== {symbol} ===")
# 1. Download dos preços
price_hist, price_next = download_price_with_next(symbol, startdate, enddate)
# 2. Encontrar melhores parâmetros
best_label = find_best_params(price_hist,maLimits)
_, b_f, _, b_s, _, s_f, _, s_s = re.split('[_=]', best_label)
mas = list(map(int, [b_f, b_s, s_f, s_s]))
# 3. Sinais com parâmetros ótimos
entries_next, exits_next, b_fast_ma, b_slow_ma, s_fast_ma, s_slow_ma = get_next_signals(price_next, mas)
pf_best = vbt.Portfolio.from_signals(price_next, entries_next, exits_next, init_cash=10000, fees=0.001)
# 5. Decisão para o próximo dia
last_entry = entries_next.iloc[-1]
last_exit  = exits_next.iloc[-1]
if last_entry and not last_exit:
    signal_text = "📈 Sinal de COMPRA para o próximo dia"
elif last_exit and not last_entry:
    signal_text = "📉 Sinal de VENDA para o próximo dia"
else:
    signal_text = "⏸️ Sem ação — manter posição"
print(f"  {signal_text}")
# 6. Gráfico
plot_strategy(symbol, price_next, entries_next, exits_next, b_fast_ma, b_slow_ma, b_f, b_s, s_fast_ma, s_slow_ma, s_f, s_s, pf_best, signal_text)
