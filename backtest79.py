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
    price_data = vbt.YFData.download(symbol, start=startdate, end=enddate + timedelta(days=7))
    close = price_data.get("Close")
    open_ = price_data.get("Open")

    # Histórico até enddate
    price_hist = close.loc[close.index <= enddate]

    # Próximo dia de mercado
    next_day_idx = close.index[close.index > enddate][0]
    next_open_price = open_.loc[next_day_idx]

    # Adiciona próximo dia ao histórico usando preço de abertura
    price_with_next = price_hist.copy()
    price_with_next.loc[next_day_idx] = next_open_price

    return price_hist, price_with_next


def find_best_params(price):
    """
    Faz grid search para encontrar os melhores parâmetros (b_f, b_s, s_f, s_s).
    Retorna best_label (string) e pf_all (portfólio de todos os testes).
    """
    b_fast_windows = np.arange(1, 30)
    s_fast_windows = np.arange(1, 30)
    b_slow_windows = np.arange(1, 15)
    s_slow_windows = np.arange(1, 15)

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
    print(f"  Melhor: {best_label} | Lucro: €{best_profit:.2f}")

    return best_label, pf_all


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
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.15, row_heights=[0.7, 0.3],
                        subplot_titles=("Preço e Sinais", "Lucro Acumulado"))

    fig.add_trace(go.Scatter(x=price.index, y=price.values,
                             mode='lines', name='Preço', line=dict(color='black')), row=1, col=1)

    # Painel 1 - Buy
    fig.add_trace(go.Scatter(x=price.index, y=b_fast_ma.values,
                             mode='lines', name=f'MA Rápida ({b_f})', line=dict(color='rgb(60, 179, 113)')), row=1, col=1)
    fig.add_trace(go.Scatter(x=price.index, y=b_slow_ma.values,
                             mode='lines', name=f'MA Lenta ({b_s})', line=dict(color='rgb(0, 100, 0)')), row=1, col=1)
    fig.add_trace(go.Scatter(x=entries.index[entries], y=price[entries],
                             mode='markers', name='Compra', marker=dict(color='green', size=10, symbol='triangle-up')), row=1, col=1)

    # Painel 1 - Sell
    fig.add_trace(go.Scatter(x=price.index, y=s_fast_ma.values,
                             mode='lines', name=f'MA Rápida ({s_f})', line=dict(color='rgb(255, 99, 71)')), row=1, col=1)
    fig.add_trace(go.Scatter(x=price.index, y=s_slow_ma.values,
                             mode='lines', name=f'MA Lenta ({s_s})', line=dict(color='rgb(178, 34, 34)')), row=1, col=1)
    fig.add_trace(go.Scatter(x=exits.index[exits], y=price[exits],
                             mode='markers', name='Venda', marker=dict(color='red', size=10, symbol='triangle-down')), row=1, col=1)
                             

    fig.add_trace(go.Scatter(x=[price.index[-1]], y=[price.iloc[-1]],
                             mode='markers+text', text=[signal_text], textposition='top center',
                             marker=dict(color='purple', size=14, symbol='star'),
                             name='Próximo dia'), row=1, col=1)

    # Painel 2
    fig.add_trace(go.Scatter(x=pf_best.value().index, y=pf_best.value().values,
                             mode='lines', name='Lucro acumulado', line=dict(color='green')), row=2, col=1)

    fig.update_layout(title=f"Estratégia para {symbol} — {signal_text}",
                      xaxis2_title="Data",
                      yaxis_title="Preço",
                      yaxis2_title="Valor da Carteira (€)",
                      height=800,
                      legend=dict(x=0, y=1.15, bgcolor='rgba(255,255,255,0)'))

    fig.show()


# ============================================================
# LOOP PARA VÁRIOS SÍMBOLOS
# ============================================================

symbols = ['SPPW.DE']  # ['SPPW.DE', 'DAVV.DE']  # podes meter quantos quiseres
startdate = datetime(2024, 1, 1)
enddate   = datetime(2025, 6, 5)
startdate = pd.Timestamp(startdate, tz="UTC")
enddate   = pd.Timestamp(enddate, tz="UTC")

for symbol in symbols:
    print(f"\n=== {symbol} ===")

    # 1. Download dos preços
    price_hist, price_next = download_price_with_next(symbol, startdate, enddate)

    # 2. Encontrar melhores parâmetros
    best_label, _ = find_best_params(price_hist)
    _, b_f, _, b_s, _, s_f, _, s_s = re.split('[_=]', best_label)
    mas = list(map(int, [b_f, b_s, s_f, s_s]))

    # 3. Sinais com parâmetros ótimos
    entries_next, exits_next, b_fast_ma, b_slow_ma, s_fast_ma, s_slow_ma = get_next_signals(price_next, mas)

    pf_best = vbt.Portfolio.from_signals(price_next, entries_next, exits_next, init_cash=10000, fees=0.001)

    # 5. Decisão para o próximo dia
    last_entry = entries_next.iloc[-1]
    last_exit = exits_next.iloc[-1]
    if last_entry and not last_exit:
        signal_text = "📈 Sinal de COMPRA para o próximo dia"
    elif last_exit and not last_entry:
        signal_text = "📉 Sinal de VENDA para o próximo dia"
    else:
        signal_text = "⏸️ Sem ação — manter posição"
    print(f"  {signal_text}")

    # 6. Gráfico
    plot_strategy(symbol, price_next, entries_next, exits_next, b_fast_ma, b_slow_ma, b_f, b_s, s_fast_ma, s_slow_ma, s_f, s_s, pf_best, signal_text)
