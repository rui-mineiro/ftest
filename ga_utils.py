import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing
import random
from datetime import datetime, timedelta



# --- Global data for multiprocessing ---
# This variable will be set in each worker process when the pool is initialized
global_data_for_workers           = None
global_data_for_workers_reference = None
etf_ticker = 'SPPW.DE'

def init_worker(worker_data):
    """
    Initializer function for multiprocessing pool.
    Sets the global_data_for_workers variable in each worker process.
    """
    global global_data_for_workers
    global_data_for_workers = worker_data

    global global_data_for_workers_reference
    _ , _ , _ , global_data_for_workers_reference = etf_ticker_simulation( -999 , 5 , 2 , 1 )

# --- Genetic Algorithm Components ---

def initialize_population(pop_size, percent_drop_bounds, long_mean_bounds, short_mean_bounds ):
    """
    Creates an initial random population of individuals.
    Each individual is a tuple (percent_drop, long_mean, short_mean).
    Ensures that long_mean > short_mean.
    """
    population = []
    for _ in range(pop_size):
        while True: # Loop until valid parameters are generated
            p_drop = random.uniform(percent_drop_bounds[0], percent_drop_bounds[1])
            l_mean = random.randint(long_mean_bounds[0], long_mean_bounds[1])
            s_mean = random.randint(short_mean_bounds[0], short_mean_bounds[1])
#            a_rate = random.uniform(allowance_rate_bounds[0], allowance_rate_bounds[1])

            # Ensure long_mean > short_mean and short_mean >= 1
            if 1 <= s_mean < l_mean:
                population.append((p_drop, l_mean, s_mean))
                break # Valid individual generated, break inner loop
    return population

def evaluate_population(population, pool):
    """
    Evaluates the fitness of each individual in the population using a multiprocessing pool.
    """
    fitnesses = pool.map(trade_simulation, population)
    return fitnesses

def select_parents(population, fitnesses, num_parents_to_select=2, tournament_size=5):
    """
    Selects parents for crossover using tournament selection.
    """
    selected_parents = []
    # Combine population and fitness for easier selection
    individuals_with_fitness = list(zip(population, fitnesses))

    for _ in range(num_parents_to_select):
        # Randomly select 'tournament_size' individuals
        contestants = random.sample(individuals_with_fitness, min(tournament_size, len(individuals_with_fitness)))
        # The winner is the one with the best (lowest) fitness
        winner = min(contestants, key=lambda x: x[1])[0]
        selected_parents.append(winner)
    return selected_parents

def crossover(parent1, parent2, percent_drop_bounds, long_mean_bounds, short_mean_bounds, allowance_rate_bounds):
    """
    Performs one-point crossover between two parents to create two children.
    """
    child1 = list(parent1)
    child2 = list(parent2)

    # Choose a random crossover point (excluding first and last gene)
    crossover_point = random.randint(1, len(parent1) - 1)

    # Swap segments of the genes
    child1[crossover_point:], child2[crossover_point:] = child2[crossover_point:], child1[crossover_point:]

    # Ensure correct data types for children and enforce bounds/constraints
    child1 = enforce_bounds_and_constraints(
        (float(child1[0]), int(child1[1]), int(child1[2]), float(child1[3])),
        percent_drop_bounds, long_mean_bounds, short_mean_bounds, allowance_rate_bounds
    )
    child2 = enforce_bounds_and_constraints(
        (float(child2[0]), int(child2[1]), int(child2[2]), float(child2[3])),
        percent_drop_bounds, long_mean_bounds, short_mean_bounds, allowance_rate_bounds
    )

    return child1, child2

def mutate(individual, mutation_rate, percent_drop_bounds, long_mean_bounds, short_mean_bounds, allowance_rate_bounds):
    """
    Applies mutation to an individual's genes with a given mutation rate.
    """
    mutated_individual = list(individual)

    # Iterate through each gene and apply mutation independently
    if random.random() < mutation_rate:
        # Mutate percent_drop (index 0)
        mutated_individual[0] = random.uniform(percent_drop_bounds[0], percent_drop_bounds[1])

    if random.random() < mutation_rate:
        # Mutate long_mean (index 1)
        mutated_individual[1] = random.randint(long_mean_bounds[0], long_mean_bounds[1])

    if random.random() < mutation_rate:
        # Mutate short_mean (index 2)
        mutated_individual[2] = random.randint(short_mean_bounds[0], short_mean_bounds[1])

    if random.random() < mutation_rate:
        # Mutate percent_drop (index 0)
        mutated_individual[3] = random.uniform(allowance_rate_bounds[0], allowance_rate_bounds[1])


    # Convert back to tuple and enforce bounds/constraints
    return enforce_bounds_and_constraints(
        (float(mutated_individual[0]), int(mutated_individual[1]), int(mutated_individual[2]),float(mutated_individual[3])),
        percent_drop_bounds, long_mean_bounds, short_mean_bounds, allowance_rate_bounds
    )

def enforce_bounds_and_constraints(individual, percent_drop_bounds, long_mean_bounds, short_mean_bounds, allowance_rate_bounds):
    """
    Ensures that individual parameters stay within their defined bounds
    and satisfy the long_mean > short_mean constraint.
    """
    p_drop, l_mean, s_mean , a_rate = individual

    # Enforce numerical bounds
    p_drop = max(percent_drop_bounds[0], min(p_drop, percent_drop_bounds[1]))
    l_mean = max(long_mean_bounds[0], min(l_mean, long_mean_bounds[1]))
    s_mean = max(short_mean_bounds[0], min(s_mean, short_mean_bounds[1]))
    a_rate = max(allowance_rate_bounds[0], min(a_rate, allowance_rate_bounds[1]))


    # Enforce long_mean > short_mean constraint
    # If the constraint is violated, try to adjust s_mean or l_mean
    if not (1 <= s_mean < l_mean):
        # Option 1: Re-randomize both means until valid (robust but might loop)
        # while not (1 <= s_mean < l_mean):
        #     l_mean = random.randint(long_mean_bounds[0], long_mean_bounds[1])
        #     s_mean = random.randint(short_mean_bounds[0], short_mean_bounds[1])

        # Option 2: Adjust them to the closest valid configuration (simpler)
        # Ensure s_mean is at least 1
        s_mean = max(1, s_mean)
        # If s_mean is still >= l_mean, increment l_mean or decrement s_mean
        if s_mean >= l_mean:
            # Try to make l_mean just larger than s_mean, within bounds
            l_mean = max(l_mean, s_mean + 1)
            # If l_mean goes out of its upper bound, adjust s_mean instead
            if l_mean > long_mean_bounds[1]:
                l_mean = long_mean_bounds[1]
                s_mean = min(s_mean, l_mean - 1) # s_mean must be at least 1
                s_mean = max(1, s_mean) # ensure s_mean is not less than 1

    return (p_drop, l_mean, s_mean, a_rate)


def genetic_algorithm_optimization(
    pop_size,
    generations,
    percent_drop_bounds,
    long_mean_bounds,
    short_mean_bounds,
#    allowance_rate_bounds,
    data_for_workers, # Moved this non-default argument before default ones
    mutation_rate=0.1,
    elitism_count=1
):
    """
    Main function to run the genetic algorithm..
    Args:
        pop_size (int): The number of individuals in each population.
        generations (int): The total number of generations to run the algorithm.
        percent_drop_bounds (tuple): A tuple (min, max) for percent_drop.
        long_mean_bounds (tuple): A tuple (min, max) for long_mean.
        short_mean_bounds (tuple): A tuple (min, max) for short_mean.
        data_for_workers (pd.Series): The historical ETF data to be used by worker processes.
        mutation_rate (float): The probability of mutation for a gene.
        elitism_count (int): The number of best individuals to carry over directly to the next generation.
    """
    # Initialize multiprocessing pool, passing data to each worker
    with multiprocessing.Pool(multiprocessing.cpu_count(), initializer=init_worker, initargs=(data_for_workers,)) as pool:
        population = initialize_population(pop_size, percent_drop_bounds, long_mean_bounds, short_mean_bounds, allowance_rate_bounds)
        best_overall_individual = None
        best_overall_fitness = float('inf') # Initialize with a very high value for minimization

        for gen in range(generations):
            print(f"--- Generation {gen+1}/{generations} ---")
            fitnesses = evaluate_population(population, pool)

            # Find the best individual in the current generation
            current_best_idx = np.argmin(fitnesses)
            current_best_individual = population[current_best_idx]
            current_best_fitness = fitnesses[current_best_idx]

            # Update overall best if current best is better
            if current_best_fitness < best_overall_fitness:
                best_overall_fitness = current_best_fitness
                best_overall_individual = current_best_individual

            print(f"  Current best individual: {current_best_individual}, Fitness (negative return): {current_best_fitness}")
            print(f"  Overall best individual: {best_overall_individual}, Fitness (negative return): {best_overall_fitness}")

            # Create the next generation
            new_population = []

            # Elitism: Directly copy the best individuals to the new generation
            sorted_population_fitness = sorted(zip(population, fitnesses), key=lambda x: x[1])
            for i in range(min(elitism_count, pop_size)): # Ensure not to take more than pop_size
                new_population.append(sorted_population_fitness[i][0])

            # Fill the rest of the new population through selection, crossover, and mutation
            num_offspring_needed = pop_size - len(new_population)
            while len(new_population) < pop_size:
                # Select two parents
                parents = select_parents(population, fitnesses, num_parents_to_select=2)

                # Perform crossover
                child1, child2 = crossover(parents[0], parents[1], percent_drop_bounds, long_mean_bounds, short_mean_bounds, allowance_rate_bounds)

                # Apply mutation and add to new population
                new_population.append(mutate(child1, mutation_rate, percent_drop_bounds, long_mean_bounds, short_mean_bounds, allowance_rate_bounds))
                if len(new_population) < pop_size:
                    new_population.append(mutate(child2, mutation_rate, percent_drop_bounds, long_mean_bounds, short_mean_bounds, allowance_rate_bounds))

            population = new_population[:pop_size] # Ensure population size remains constant

        return best_overall_individual, best_overall_fitness





def etf_ticker_simulation(percent_drop , long_mean , short_mean ):
    """
    Simulates the ETF trading strategy based on the given parameters.
    Accessed globally shared data for efficiency in multiprocessing.
    """
    # Access the globally set data for this worker process
    local_data = global_data_for_workers.copy() # Use a copy to avoid modification issues across processes

    investment = 0
    shares = 0
    initial_cash   = 100           # Initial cash
    cash_available = initial_cash  # Initial cash

    # Initialize columns for simulation results
    local_data['portfolio_value'] = 0.0
    local_data['portfolio_pct'] = 0.0
    local_data['invested_value'] = 0.0
    local_data['shares'] = 0

    buy_dates = []
    buy_performance = []
    buy_values = []
    is_more_than_one_month=True


    for i in range(1, len(local_data)):
        today = local_data.index[i]
        price_today = local_data[etf_ticker].iloc[i]

        # Add monthly cash infusion after one month from last purchase
        if len(buy_dates) > 0:
            is_more_than_one_month = abs(today - buy_dates[-1]) > timedelta(days=30)
            if is_more_than_one_month:
                cash_available += initial_cash


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
        qty=cash_available // price_today
        if qty > 0:
            if is_more_than_one_month:
                cost = qty * price_today
                shares += qty
                cash_available -= cost
                investment += cost
                bought = True
            elif ((price_short_mean - price_long_mean)/price_long_mean) < percent_drop and not bought:
                cash_available += initial_cash
                qty=cash_available // price_today
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
    percent_drop, long_mean, short_mean  = params

    # Constraint: long_mean must be strictly greater than short_mean
    # Also, ensure short_mean is at least 1 (to have a valid mean)
    if not (1 <= short_mean < long_mean):
        # Penalize invalid combinations heavily to guide the GA away from them
        return 1e10 # A very large number representing a bad fitness

    # Run the simulation
    buy_dates, buy_performance, buy_values, xdata = etf_ticker_simulation(percent_drop , long_mean , short_mean  )

    final_value = np.mean(xdata['portfolio_value'].iloc[-120:-1])
    investment  = xdata['invested_value'].iloc[-1]

    # Calculate final performance
    # performance = (final_value - investment) / investment if investment > 0 else 0
    performance = final_value / investment if investment > 0 else 0


    xpto=np.sum(global_data_for_workers_reference['portfolio_value']-xdata['portfolio_value'])
#    xpto=np.sum(global_data_for_workers_reference['portfolio_pct']-xdata['portfolio_pct'])


    # Return negative performance for minimization (maximizing return)
    return round(-final_value, 2)



def strategy_simulate(data, percent_drop , long_mean , short_mean ):
    """
    Visualizes the performance of the trading strategy with the given parameters.
    """
    print("\n--- Optimization Complete ---")
    print(f"Optimal percent_drop: {percent_drop:.2f}")
    print(f"Optimal long_mean: {long_mean}")
    print(f"Optimal short_mean: {short_mean}")
#    print(f"Optimal allowance_rate: {allowance_rate:.2f}")
    
    


    init_worker(data)
    buy_dates   , buy_performance   , buy_values   , xdata = etf_ticker_simulation(percent_drop , long_mean , short_mean )
    print(f"Maximized Return : {xdata['portfolio_pct'].iloc[-1]:.2f}%")

    ydata=global_data_for_workers_reference.copy()
    print(f"Maximized Return Reference: {ydata['portfolio_pct'].iloc[-1]:.2f}%")




    fig,  (ax11, ax21)  = plt.subplots(2)

    ax11.set_xlabel('Date')
    ax11.set_ylabel('Portfolio Value', color='tab:blue')
    ax11.plot(xdata.index, xdata['portfolio_value'], label='Portfolio Value' , color='tab:blue')
    ax11.plot(xdata.index, xdata['invested_value'],  label='Invested_value'  , color='tab:blue')
    ax11.plot(ydata.index, ydata['portfolio_value'], label='Portfolio Value Reference', color='tab:green')
    ax11.plot(ydata.index, ydata['invested_value'],  label='Invested_value  Reference', color='tab:green')
    ax11.legend()
    ax11.grid(True)

    

    ax12 = ax11.twinx()  # Instantiate a second axes that shares the same x-axis
    ax12.set_ylabel('Shares', color='tab:red')  # We already handled the x-label
    ax12.plot(xdata.index, xdata[etf_ticker], label='Share Price', color='tab:red')
    ax12.tick_params(axis='y', labelcolor='tab:red')
    ax12.legend(loc='lower right')
    # ax12.grid(True)


    ax21.set_xlabel('Date')
    ax21.set_ylabel('Portfolio Performance (%)', color='tab:blue')
    ax21.plot(xdata.index, xdata['portfolio_pct'], label='Portfolio Performance (%)'          , color='tab:blue')
    ax21.plot(ydata.index, ydata['portfolio_pct'], label='Portfolio Performance Reference(%)' , color='tab:green')
    ax21.legend()
    ax21.grid(True)

    plt.tight_layout()
    plt.show()



#    fig.tight_layout()
#
#
#
#    # Plot portfolio percentage return over time
#    plt.figure(figsize=(12, 6))
#    plt.plot(xdata.index, xdata['portfolio_pct'], label=f'Final Return')
#    plt.scatter(buy_dates, buy_performance, color='red', marker='o', label='Share Purchase', zorder=5)
#    plt.xlabel('Date')
#    plt.ylabel('Performance (%)')
#    plt.title(f'Portfolio Performance ({etf_ticker}) with Optimal Strategy')
#    plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
#    plt.grid(True)
#    plt.legend()
#    plt.tight_layout()
#    plt.show()
#
#    # Plot portfolio accumulated value over time
#    plt.figure(figsize=(12, 6))
#    plt.plot(xdata.index, xdata['portfolio_value'], label=f'Portfolio Value')
#    plt.plot(xdata.index, xdata['invested_value'],  label=f'Investment Value')
#    plt.scatter(buy_dates, buy_values, color='red', marker='o', label='Share Purchase', zorder=5)
#    plt.xlabel('Date')
#    plt.ylabel('Accumulated Value')
#    plt.title(f'Accumulated Investment Value ({etf_ticker}) with Optimal Strategy')
#    plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
#    plt.grid(True)
#    plt.legend()
#    plt.tight_layout()
#    plt.show()

