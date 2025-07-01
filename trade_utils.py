import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing
import random
from datetime import datetime, timedelta




def etf_ticker_simulation(percent_drop , long_mean , short_mean , allowance_rate ):
    """
    Simulates the ETF trading strategy based on the given parameters.
    Accessed globally shared data for efficiency in multiprocessing.
    """
    # Access the globally set data for this worker process
    local_data = global_data_for_workers.copy() # Use a copy to avoid modification issues across processes

    investment = 0
    shares = 0
    initial_cash   = 1000          # Initial cash
    cash_available = initial_cash  # Initial cash

    # Initialize columns for simulation results
    local_data['portfolio_value'] = 0.0
    local_data['portfolio_pct'] = 0.0
    local_data['invested_value'] = 0.0
    local_data['shares'] = 0

    buy_dates = []
    buy_performance = []
    buy_values = []

    for i in range(1, len(local_data)):
        today = local_data.index[i]
        price_today = local_data[etf_ticker].iloc[i]

        # Add monthly cash infusion after one month from last purchase
        if len(buy_dates) > 0:
            is_more_than_one_month = abs(today - buy_dates[-1]) > timedelta(days=30)
            if is_more_than_one_month:
                cash_available += initial_cash*allowance_rate


        # Calculate long and short moving averages
        # Ensure sufficient data exists for the moving averages
        if i >= long_mean:
            price_long_mean = np.mean(local_data[etf_ticker].iloc[i - long_mean + 1:i + 1])
        else:
            price_long_mean = np.mean(local_data[etf_ticker].iloc[:i + 1]) # Use all available data

        if i >= short_mean:
            price_short_mean = np.mean(local_data[etf_ticker].iloc[i - short_mean + 1:i + 1])
        else:
            price_short_mean = np.mean(local_data[etf_ticker].iloc[:i + 1]) # Use all available data

        bought = False

        # Buy condition: if long mean minus short mean drops below percent_drop and cash is available
        if (price_long_mean - price_short_mean < percent_drop) and (cash_available >= 0):
            qty = 100 // price_today # Buy shares worth approximately 100 units of currency
            if qty > 0:
                cost = qty * price_today
                shares += qty
                cash_available -= cost
                investment += cost
                bought = True

        # Update daily portfolio performance
        today_value = shares * price_today
        today_pct = (today_value - investment) / investment * 100 if investment > 0 else 0

        local_data.loc[today,'portfolio_value'] = today_value
        local_data.loc[today,'portfolio_pct'] = today_pct
        local_data.loc[today,'invested_value'] = investment
        local_data.loc[today,'shares'] = shares

        # Record buy events
        if bought:
            buy_dates.append(today)
            buy_performance.append(today_pct)
            buy_values.append(investment)

    return buy_dates, buy_performance, buy_values, local_data


def trade_simulation(params):
    """
    Fitness function for the genetic algorithm.
    It takes a tuple of parameters and returns the negative of the final performance.
    Lower values indicate better performance (since we are minimizing).
    """
    percent_drop, long_mean, short_mean , allowance_rate = params

    # Constraint: long_mean must be strictly greater than short_mean
    # Also, ensure short_mean is at least 1 (to have a valid mean)
    if not (1 <= short_mean < long_mean):
        # Penalize invalid combinations heavily to guide the GA away from them
        return 1e10 # A very large number representing a bad fitness

    # Run the simulation
    buy_dates, buy_performance, buy_values, xdata = etf_ticker_simulation(percent_drop , long_mean , short_mean , allowance_rate )

    final_value = xdata['portfolio_value'].iloc[-1]
    investment  = xdata['invested_value'].iloc[-1]

    # Calculate final performance
    performance = (final_value - investment) / investment if investment > 0 else 0

    # Return negative performance for minimization (maximizing return)
    return -round(performance * 100, 2)

