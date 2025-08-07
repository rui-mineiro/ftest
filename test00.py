import pandas as pd
import numpy as np
import vectorbt as vbt

# Create a sample price series
price = pd.Series([
    100, 102, 105, 103, 101,
    98, 96, 99, 102, 104,
    106, 105, 103, 100, 98,
    97, 99, 101
])

# Calculate a simple moving average
moving_average = price.rolling(window=3).mean()

# Generate raw entry and exit signals using the .vbt accessor
entries = price.vbt.crossed_above(moving_average)
exits = price.vbt.crossed_below(moving_average)

# Display the messy signals
print("------ Raw Signals ------")
raw_signals_df = pd.DataFrame({'Price': price, 'MA': moving_average, 'Entries': entries, 'Exits': exits})
print(raw_signals_df.replace(False, '-'))


# --- CORRECTED LINE ---
# Clean the signals using the .vbt.signals accessor
cleaned_entries, cleaned_exits = entries.vbt.signals.clean(exits)


# Display the cleaned signals
print("\n------ Cleaned Signals ------")
cleaned_signals_df = pd.DataFrame({'Price': price, 'MA': moving_average, 'Entries': cleaned_entries, 'Exits': cleaned_exits})
print(cleaned_signals_df.replace(False, '-'))