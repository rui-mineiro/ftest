import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Baixar dados históricos do SPY (S&P 500 ETF)
spy = yf.download("SPY", start="2015-01-01", end="2025-01-01",  auto_adjust=False)
prices = spy["Adj Close"]

# Simulação de investidor passivo
initial_investment = 10000
passive_shares = initial_investment / prices.iloc[0].values[0]
passive_value = passive_shares * prices

# Simular N investidores ativos
N = 1000
active_values = []

for _ in range(N):
    cash = initial_investment
    shares = 0
    for day in prices.index:
        # 50% chance de comprar ou vender cada dia (totalmente aleatório)
        if np.random.rand() < 0.01:  # só decide negociar em 1% dos dias
            if np.random.rand() < 0.5 and cash > 0:
                # compra
                shares += cash // prices.loc[day].values[0]
                cash = 0
            elif shares > 0:
                # vende
                cash += shares * prices.loc[day].values[0]
                shares = 0
    # Valor final da carteira
    total = cash + shares * prices.iloc[-1]
    active_values.append(total)

# Resultados
active_values = np.array(active_values)
passive_final = passive_value.iloc[-1]

# Gráficos
plt.figure(figsize=(12,6))
plt.hist(active_values, bins=50, alpha=0.7, label="Investidores Ativos")
plt.axvline(passive_final.item(), color="red", linestyle="--", label=f"Investidor Passivo (${passive_final.item():.2f})")
plt.title("Distribuição do Retorno Final - Investidores Ativos vs Passivo")
plt.xlabel("Valor final da carteira")
plt.ylabel("Número de investidores")
plt.legend()
plt.grid(True)
plt.show()
