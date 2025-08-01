import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing
import random
from bt_64_utils import *
from datetime import datetime, timedelta
import os

# --- Global data for multiprocessing ---
# This variable will be set in each worker process when the pool is initialized
global_data_for_workers = None
etf_ticker = 'DAVV.DE' # 'DFEN.DE' # 'SPPW.DE'
csv_filename = f"{etf_ticker}_data.csv"


# --- Main execution block ---
if __name__ == "__main__":
    print(f"Downloading ETF data for {etf_ticker}...")
    # Check if the file exists and is not older than 1 day
    if os.path.exists(csv_filename):
        file_mtime = datetime.fromtimestamp(os.path.getmtime(csv_filename))
        if datetime.now() - file_mtime < timedelta(days=3):
            original_data = pd.read_csv(csv_filename, index_col=0, parse_dates=True)
            original_data.name = etf_ticker
            print("Data read_csv complete. Starting genetic algorithm optimization...")
        else:
            download_required = True
    else:
        download_required = True

    if 'download_required' in locals() and download_required:
        original_data = yf.download(etf_ticker, start='2023-11-26', end='2025-06-27', auto_adjust=False)
        original_data = original_data['Adj Close'].dropna()
        original_data.name = etf_ticker
        original_data.to_csv(csv_filename)
        print("Data download complete. Starting genetic algorithm optimization...")




    # Define genetic algorithm parameters and bounds
    POPULATION_SIZE = 50*8
    GENERATIONS     = 10
    MUTATION_RATE   = 0.01
    ELITISM_COUNT   = 1 # Keep the top 2 individuals

    buy_percent_drop_bounds  = [-30, 30]
    sell_percent_drop_bounds = [-30, 30]
    buy_long_mean_bounds     = [1 ,  60]
    buy_short_mean_bounds    = [1 ,  60]
    sell_long_mean_bounds    = [1 ,  60]
    sell_short_mean_bounds   = [1 ,  60]



    # Run the genetic algorithm
    best_params, best_fitness = genetic_algorithm_optimization(
        pop_size=POPULATION_SIZE,
        generations=GENERATIONS,
        buy_percent_drop_bounds  =buy_percent_drop_bounds,
        buy_long_mean_bounds     =buy_long_mean_bounds,
        buy_short_mean_bounds    =buy_short_mean_bounds,
        sell_percent_drop_bounds =sell_percent_drop_bounds,
        sell_long_mean_bounds    =sell_long_mean_bounds,
        sell_short_mean_bounds   =sell_short_mean_bounds,
        data_for_workers=original_data, # This is now correctly placed
        mutation_rate=MUTATION_RATE,
        elitism_count=ELITISM_COUNT
    )



    # Visualize the best strategy found
    print("\nGenerating strategy simulation plots...")
    strategy_simulate(original_data, best_params[0], best_params[1], best_params[2], best_params[3], best_params[4], best_params[5])
    print("Plots generated successfully.")

