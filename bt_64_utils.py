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

def init_worker(worker_data):
    """
    Initializer function for multiprocessing pool.
    Sets the global_data_for_workers variable in each worker process.
    """
    global global_data_for_workers
    global_data_for_workers = worker_data

# --- Genetic Algorithm Components ---

def initialize_population(pop_size, buy_percent_drop_bounds, buy_long_mean_bounds, buy_short_mean_bounds,
                                    sell_percent_drop_bounds, sell_long_mean_bounds, sell_short_mean_bounds ):
    """
    Creates an initial random population of individuals.
    Each individual is a tuple (percent_drop, long_mean, short_mean).
    Ensures that long_mean > short_mean.
    """
    population = []
    for _ in range(pop_size):
        while True: # Loop until valid parameters are generated
            b_p_drop = random.uniform(buy_percent_drop_bounds[0] , buy_percent_drop_bounds[1])
            b_l_mean = random.randint(buy_long_mean_bounds[0]    , buy_long_mean_bounds[1])
            b_s_mean = random.randint(buy_short_mean_bounds[0]   , buy_short_mean_bounds[1])
            s_p_drop = random.uniform(sell_percent_drop_bounds[0], sell_percent_drop_bounds[1])
            s_l_mean = random.randint(sell_long_mean_bounds[0]   , sell_long_mean_bounds[1])
            s_s_mean = random.randint(sell_short_mean_bounds[0]  , sell_short_mean_bounds[1])

            # Ensure long_mean > short_mean and short_mean >= 1
            if  b_s_mean < b_l_mean and s_s_mean < s_l_mean:
                population.append((b_p_drop, b_l_mean, b_s_mean, s_p_drop, s_l_mean, s_s_mean))
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

def crossover(parent1, parent2, buy_percent_drop_bounds, buy_long_mean_bounds, buy_short_mean_bounds,
                                sell_percent_drop_bounds, sell_long_mean_bounds, sell_short_mean_bounds):
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
        (float(child1[0]), int(child1[1]), int(child1[2]),float(child1[3]), int(child1[4]), int(child1[5])),
        buy_percent_drop_bounds, buy_long_mean_bounds, buy_short_mean_bounds,
        sell_percent_drop_bounds, sell_long_mean_bounds, sell_short_mean_bounds
    )
    child2 = enforce_bounds_and_constraints(
        (float(child1[0]), int(child1[1]), int(child1[2]),float(child1[3]), int(child1[4]), int(child1[5])),
        buy_percent_drop_bounds, buy_long_mean_bounds, buy_short_mean_bounds,
        sell_percent_drop_bounds, sell_long_mean_bounds, sell_short_mean_bounds
    )

    return child1, child2

def mutate(individual, mutation_rate, buy_percent_drop_bounds, buy_long_mean_bounds, buy_short_mean_bounds,
                                      sell_percent_drop_bounds, sell_long_mean_bounds, sell_short_mean_bounds):
    """
    Applies mutation to an individual's genes with a given mutation rate.
    """
    mutated_individual = list(individual)

    # Iterate through each gene and apply mutation independently
    if random.random() < mutation_rate:
        # Mutate percent_drop (index 0)
        mutated_individual[0] = random.uniform(buy_percent_drop_bounds[0], buy_percent_drop_bounds[1])

    if random.random() < mutation_rate:
        # Mutate long_mean (index 1)
        mutated_individual[1] = random.randint(buy_long_mean_bounds[0], buy_long_mean_bounds[1])

    if random.random() < mutation_rate:
        # Mutate short_mean (index 2)
        mutated_individual[2] = random.randint(buy_short_mean_bounds[0], buy_short_mean_bounds[1])


    # Iterate through each gene and apply mutation independently
    if random.random() < mutation_rate:
        # Mutate percent_drop (index 0)
        mutated_individual[3] = random.uniform(sell_percent_drop_bounds[0], sell_percent_drop_bounds[1])

    if random.random() < mutation_rate:
        # Mutate long_mean (index 1)
        mutated_individual[4] = random.randint(sell_long_mean_bounds[0], sell_long_mean_bounds[1])

    if random.random() < mutation_rate:
        # Mutate short_mean (index 2)
        mutated_individual[5] = random.randint(sell_short_mean_bounds[0], sell_short_mean_bounds[1])


    # Convert back to tuple and enforce bounds/constraints
    return enforce_bounds_and_constraints(
        (float(mutated_individual[0]), int(mutated_individual[1]), int(mutated_individual[2]),
         float(mutated_individual[3]), int(mutated_individual[4]), int(mutated_individual[5])),
        buy_percent_drop_bounds, buy_long_mean_bounds, buy_short_mean_bounds,
        sell_percent_drop_bounds, sell_long_mean_bounds, sell_short_mean_bounds         
    )

def enforce_bounds_and_constraints(individual, buy_percent_drop_bounds, buy_long_mean_bounds, buy_short_mean_bounds,
                                               sell_percent_drop_bounds, sell_long_mean_bounds, sell_short_mean_bounds):
    """
    Ensures that individual parameters stay within their defined bounds
    and satisfy the long_mean > short_mean constraint.
    """
    b_p_drop, b_l_mean, b_s_mean, s_p_drop, s_l_mean, s_s_mean  = individual

    # Enforce numerical bounds
    b_p_drop = max(buy_percent_drop_bounds[0],  min(b_p_drop, buy_percent_drop_bounds[1]))
    b_l_mean = max(buy_long_mean_bounds[0]   ,  min(b_l_mean, buy_long_mean_bounds[1]))
    b_s_mean = max(buy_short_mean_bounds[0]  ,  min(b_s_mean, buy_short_mean_bounds[1]))
    s_p_drop = max(sell_percent_drop_bounds[0], min(s_p_drop, sell_percent_drop_bounds[1]))
    s_l_mean = max(sell_long_mean_bounds[0]   , min(s_l_mean, sell_long_mean_bounds[1]))
    s_s_mean = max(sell_short_mean_bounds[0]  , min(s_s_mean, sell_short_mean_bounds[1]))
    


    # Enforce long_mean > short_mean constraint
    # If the constraint is violated, try to adjust s_mean or l_mean
    if not ( b_s_mean < b_l_mean) or not ( s_s_mean < s_l_mean):
        # Option 1: Re-randomize both means until valid (robust but might loop)
        while not ( b_s_mean < b_l_mean) or not ( s_s_mean < s_l_mean):
            b_p_drop = random.uniform(buy_percent_drop_bounds[0] , buy_percent_drop_bounds[1])
            b_l_mean = random.randint(buy_long_mean_bounds[0]    , buy_long_mean_bounds[1])
            b_s_mean = random.randint(buy_short_mean_bounds[0]   , buy_short_mean_bounds[1])
            s_p_drop = random.uniform(sell_percent_drop_bounds[0], sell_percent_drop_bounds[1])
            s_l_mean = random.randint(sell_long_mean_bounds[0]   , sell_long_mean_bounds[1])
            s_s_mean = random.randint(sell_short_mean_bounds[0]  , sell_short_mean_bounds[1])


    return (b_p_drop, b_l_mean, b_s_mean, s_p_drop, s_l_mean, s_s_mean)


def genetic_algorithm_optimization(
    pop_size,
    generations,
    buy_percent_drop_bounds,
    buy_long_mean_bounds,
    buy_short_mean_bounds,
    sell_percent_drop_bounds,
    sell_long_mean_bounds,
    sell_short_mean_bounds,
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
        population = initialize_population(pop_size, buy_percent_drop_bounds,  buy_long_mean_bounds,  buy_short_mean_bounds,
                                                     sell_percent_drop_bounds, sell_long_mean_bounds, sell_short_mean_bounds)
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
                child1, child2 = crossover(parents[0], parents[1], buy_percent_drop_bounds,  buy_long_mean_bounds,  buy_short_mean_bounds,
                                                                   sell_percent_drop_bounds, sell_long_mean_bounds, sell_short_mean_bounds)

                # Apply mutation and add to new population
                new_population.append(mutate(child1, mutation_rate, buy_percent_drop_bounds,  buy_long_mean_bounds,  buy_short_mean_bounds,
                                                                    sell_percent_drop_bounds, sell_long_mean_bounds, sell_short_mean_bounds))
                if len(new_population) < pop_size:
                    new_population.append(mutate(child2, mutation_rate, buy_percent_drop_bounds,  buy_long_mean_bounds,  buy_short_mean_bounds,
                                                                        sell_percent_drop_bounds, sell_long_mean_bounds, sell_short_mean_bounds))

            population = new_population[:pop_size] # Ensure population size remains constant

        return best_overall_individual, best_overall_fitness





def etf_ticker_simulation(buy_percent_drop,  buy_long_mean,  buy_short_mean,
                          sell_percent_drop, sell_long_mean, sell_short_mean ):
    """
    Simulates the ETF trading strategy based on the given parameters.
    Accessed globally shared data for efficiency in multiprocessing.
    """
    # Access the globally set data for this worker process
    local_data = global_data_for_workers.copy() # Use a copy to avoid modification issues across processes

    shares = 0
    initial_cash   = 100           # Initial cash
    cash_available = initial_cash  # Initial cash
    investment     = initial_cash

    # Initialize columns for simulation results
    local_data['portfolio_value'] = 0.0
    local_data['portfolio_pct'] = 0.0
    local_data['invested_value'] = 0.0
    local_data['shares'] = 0

    trade_dates = []
    trade_performance = []
    trade_values = []
    is_more_than_period=True
    date_more_than_period=local_data.index[0]

    etf_ticker=local_data.columns[0]


    for i in range(1, len(local_data)):
        today = local_data.index[i]
        price_today = local_data[etf_ticker].iloc[i]

        if i >= buy_long_mean:
            buy_price_long_mean = np.mean(local_data[etf_ticker].iloc[i - buy_long_mean + 1:i - buy_short_mean + 1])
        else:
            buy_price_long_mean = np.mean(local_data[etf_ticker].iloc[:i + 1]) # Use all available data

        if i >= buy_short_mean:
            buy_price_short_mean = np.mean(local_data[etf_ticker].iloc[i - buy_short_mean + 1:i + 1])
        else:
            buy_price_short_mean = np.mean(local_data[etf_ticker].iloc[:i + 1]) # Use all available data

        if i >= sell_long_mean:
            sell_price_long_mean = np.mean(local_data[etf_ticker].iloc[i - sell_long_mean + 1:i - sell_short_mean + 1])
        else:
            sell_price_long_mean = np.mean(local_data[etf_ticker].iloc[:i + 1]) # Use all available data

        if i >= sell_short_mean:
            sell_price_short_mean = np.mean(local_data[etf_ticker].iloc[i - sell_short_mean + 1:i + 1])
        else:
            sell_price_short_mean = np.mean(local_data[etf_ticker].iloc[:i + 1]) # Use all available data

        trade = False
        is_more_than_period = abs(today - date_more_than_period) > timedelta(days=7)

        # Buy, Sell or Stay
        buy =(((buy_price_long_mean - buy_price_short_mean)/buy_price_long_mean)*100 < buy_percent_drop)
        sell=(((sell_price_long_mean - sell_price_short_mean)/sell_price_long_mean)*100 > sell_percent_drop)
        stay = ( buy == sell )
        
        if not stay:
            if buy and is_more_than_period:
#                cash_available += initial_cash
                qty=cash_available // price_today
                cost = qty * price_today
#                investment += cost
                shares += qty
                cash_available -= cost
                trade = True
                date_more_than_period=today
            elif sell:
                cash_available += price_today*shares
                shares = 0
                trade = True
  

        # Update daily portfolio performance
        today_value = ( shares * price_today ) + cash_available
        today_pct = (today_value - investment) / investment * 100 if investment > 0 else 0

        local_data.loc[today,'portfolio_value'] = today_value
        local_data.loc[today,'portfolio_pct']   = today_pct
        local_data.loc[today,'invested_value']  = investment
        local_data.loc[today,'shares']          = shares

        # Record buy events
        if trade:
            trade_dates.append(today)
            trade_performance.append(today_pct)
            trade_values.append(investment)

    return trade_dates, trade_performance, trade_values, local_data


def trade_simulation(params):
    """
    Fitness function for the genetic algorithm.
    It takes a tuple of parameters and returns the negative of the final performance.
    Lower values indicate better performance (since we are minimizing).
    """
    buy_percent_drop, buy_long_mean, buy_short_mean, sell_percent_drop, sell_long_mean, sell_short_mean = params

    # Constraint: long_mean must be strictly greater than short_mean
    # Also, ensure short_mean is at least 1 (to have a valid mean)
    if not ((buy_short_mean < buy_long_mean) and (sell_short_mean < sell_long_mean)):
        # Penalize invalid combinations heavily to guide the GA away from them
        return 1e10 # A very large number representing a bad fitness

    # Run the simulation
    trade_dates, trade_performance, trade_values, xdata = etf_ticker_simulation(buy_percent_drop,  buy_long_mean,  buy_short_mean,
                                                                          sell_percent_drop, sell_long_mean, sell_short_mean )

    final_value = xdata['portfolio_value'].iloc[-1]
#   final_value = np.mean(xdata['portfolio_value'])
    investment  = xdata['invested_value'].iloc[-1]

#    performance = (final_value - investment) / investment if investment > 0 else 0
#    performance = final_value / investment if investment > 0 else 0
    
    
    performance = final_value

    return round(-performance, 2)



def strategy_simulate(data, buy_percent_drop,  buy_long_mean,  buy_short_mean,
                            sell_percent_drop, sell_long_mean, sell_short_mean ):
    """
    Visualizes the performance of the trading strategy with the given parameters.
    """

    etf_ticker=data.columns[0]

    print("\n--- Optimization Complete ---")
    print(f"Optimal buy  percent_drop: {buy_percent_drop:.2f}")
    print(f"Optimal buy  long_mean:    {buy_long_mean}")
    print(f"Optimal buy  short_mean:   {buy_short_mean}")
    print(f"Optimal sell percent_drop: {sell_percent_drop:.2f}")
    print(f"Optimal sell long_mean:    {sell_long_mean}")
    print(f"Optimal sell short_mean:   {sell_short_mean}")
    
    
    init_worker(data)
    trade_dates   , trade_performance   , trade_values   , xdata = etf_ticker_simulation(buy_percent_drop,  buy_long_mean,  buy_short_mean,
                                                                                         sell_percent_drop, sell_long_mean, sell_short_mean )
    print(f"Maximized Return          : {xdata['portfolio_pct'].iloc[-1]:.2f}%")
    print(f"Maximized Value           : {xdata['portfolio_value'].iloc[-1]:.2f}€")
    
    fig,  (ax11, ax21)  = plt.subplots(2)

    ax11.set_xlabel('Date')
    ax11.plot(xdata.index, xdata['portfolio_value'], label='Portfolio Value' , color='tab:blue')
    ax11.plot(xdata.index, xdata['invested_value'],  label='Invested_value'  , color='tab:green')
    ax11.set_yscale('log')
    ax11.legend()
    ax11.grid(True)

    

    ax12 = ax11.twinx()  # Instantiate a second axes that shares the same x-axis
    ax12.plot(xdata.index, xdata[etf_ticker], label='Share Price', color='tab:red')
    ax12.tick_params(axis='y', labelcolor='tab:red')
    ax12.legend(loc='lower right')
    # ax12.grid(True)


    ax21.set_xlabel('Date')
    ax21.set_ylabel('Portfolio Performance (%)', color='tab:blue')
    ax21.plot(xdata.index, xdata['portfolio_pct'], label='Portfolio Performance (%)'          , color='tab:blue')
    ax21.legend()
    ax21.grid(True)
    ax21.set_yscale('log')

    plt.tight_layout()
    plt.show()


