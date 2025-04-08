import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime


def plot_stock_and_portfolio(stock_data):
    """
    Plot both stock price and portfolio value
    """
    fig, ax1 = plt.subplots(1, 1,  sharex=True)
    
    # Plot stock price on first axis
    color = 'tab:red'
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Stock Price (USD)')
    ax1.plot(stock_data.index, stock_data['Close'], label='Tesla Closing Price', color=color, alpha=0.7)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True)
    ax1.legend(loc='upper left')
    

    ax2 = ax1.twinx()

    # Plot portfolio value on second axis
    color = 'tab:blue'
    ax2.set_ylabel('Portfolio Value (USD)')
    ax2.plot(stock_data.index, stock_data['Portfolio Value'], label='Portfolio Value', color=color)
    ax2.grid(True)
    ax2.legend(loc='upper left')
    
    plt.tight_layout()
    plt.show()



def get_tesla_stock_data():
    """
    Fetch all available Tesla stock data from Yahoo Finance from the first available date until today
    """
    tesla = yf.Ticker("TSLA")
    tsla_data = tesla.history(period="max")
    tsla_data['Cash']=0
    tsla_data['Shares']=0
    tsla_data['Portfolio Value']=0
    return tsla_data



def portfolio_cash(stock_data, date_deposit_pairs):
    """
    Deposit money in portfolio
    """
    for date, value in date_deposit_pairs:
        try:
            # Try to get the exact index position
            position = stock_data.index.get_loc(date)
        except KeyError:
            indexer = stock_data.index.get_indexer([date], method='ffill')
            position = indexer[0]+1

        stock_data.loc[stock_data.index>=stock_data.index[position], 'Cash']   += value


    return stock_data


def stock_transaction(stock_data, date_stocks_pairs):
    """
    Update portfolio with stock transactions
    """
    for date, value in date_stocks_pairs:
        try:
            # Try to get the exact index position
            position = stock_data.index.get_loc(date)
        except KeyError:
            indexer = stock_data.index.get_indexer([date], method='ffill')
            position = indexer[0]+1

        stock_data.loc[stock_data.index>=stock_data.index[position], 'Shares'] += value
        stock_data.loc[stock_data.index>=stock_data.index[position], 'Cash']   -= value*stock_data.loc[stock_data.index[position], 'Close']
        stock_data.loc[stock_data.index, 'Portfolio Value']   = stock_data.loc[stock_data.index, 'Cash'] + stock_data.loc[stock_data.index, 'Shares']*stock_data.loc[stock_data.index, 'Close']
        
        

    return stock_data



cash_pairs = [
    ('1972-04-10', 1000)
]

date_stocks_validation = [
    ('1972-04-10', 100)
]

date_stocks_training = [
    ('1972-04-10', 820),
    ('2020-04-02', -820),
    ('2021-07-03', 30),
    ('2022-04-02', -30)
]



tsla_data = get_tesla_stock_data()
tsla_data = portfolio_cash(tsla_data, cash_pairs)

validation_value = stock_transaction(tsla_data, date_stocks_validation)['Portfolio Value'].iloc[-1]
training_value   = stock_transaction(tsla_data, date_stocks_training)['Portfolio Value'].iloc[-1]

plot_stock_and_portfolio(tsla_data)


tsla_data
