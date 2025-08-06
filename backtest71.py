
import vectorbt as vbt

# Download daily closing prices for 'SPPW.DE' from Yahoo Finance
eth_price = vbt.YFData.download('SPPW.DE').get('Close')

# Compute technical indicators on the price series
fast_ma = vbt.MA.run(eth_price, window=10)    # 10-day moving average (fast MA)
slow_ma = vbt.MA.run(eth_price, window=50)    # 50-day moving average (slow MA)
rsi_ind = vbt.RSI.run(eth_price, window=14)   # 14-day RSI indicator

# Access the indicator values (as pandas Series)
fast_ma_values = fast_ma.ma  # or fast_ma.values
slow_ma_values = slow_ma.ma
rsi_values = rsi_ind.rsi

# Define entry and exit conditions
entries = (fast_ma.ma_crossed_above(slow_ma)) & (rsi_ind.rsi_above(50))
exits   = (slow_ma.ma_crossed_above(fast_ma)) & (rsi_ind.rsi_below(50))

# Run the backtest with our signals on ETH-USD, starting with $10,000
pf = vbt.Portfolio.from_signals(eth_price, entries, exits, init_cash=10000)

# Evaluate performance
total_profit = pf.total_profit()
print(f"Total Profit: ${total_profit:.2f}")

