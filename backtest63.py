import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing
import random
from ga_utils import *
# from trade_utils import *
from datetime import datetime, timedelta

# --- Global data for multiprocessing ---
# This variable will be set in each worker process when the pool is initialized
global_data_for_workers = None
etf_ticker = 'SPPW.DE'

# --- Main execution block ---
if __name__ == "__main__":
    print(f"Downloading ETF data for {etf_ticker}...")
    # Download data once  
    original_data = yf.download(etf_ticker , start='2020-07-26' , end='2025-06-27' ,  auto_adjust=False)
    original_data = original_data['Adj Close'].dropna()
    original_data.name = etf_ticker # Name the series for easier access

    print("Data download complete. Starting genetic algorithm optimization...")

    # Define genetic algorithm parameters and bounds
    POPULATION_SIZE = 2*8
    GENERATIONS     = 5
    MUTATION_RATE   = 0.9
    ELITISM_COUNT   = 0 # Keep the top 2 individuals

    percent_drop_bounds = [-3, 3]
    long_mean_bounds = [1 , 120]
    short_mean_bounds = [1 , 60]


    # Run the genetic algorithm
    best_params, best_fitness = genetic_algorithm_optimization(
        pop_size=POPULATION_SIZE,
        generations=GENERATIONS,
        percent_drop_bounds=percent_drop_bounds,
        long_mean_bounds=long_mean_bounds,
        short_mean_bounds=short_mean_bounds,
        data_for_workers=original_data, # This is now correctly placed
        mutation_rate=MUTATION_RATE,
        elitism_count=ELITISM_COUNT
    )



    # Visualize the best strategy found
    print("\nGenerating strategy simulation plots...")
    strategy_simulate(original_data, best_params[0], best_params[1], best_params[2])
    print("Plots generated successfully.")

