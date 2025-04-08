import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

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



def portfolio_cash(stock_data, cash=0, cash_date='2012-07-21'):
    """
    Deposit money in portfolio
    """
    for date, value in date_stocks_pairs:
        try:
            # Try to get the exact index position
            position = stock_data.index.get_loc(date)
        except KeyError:
            indexer = stock_data.index.get_indexer([date], method='ffill')
            position = indexer[0]+1

        stock_data.loc[stock_data.index[position], 'Shares'] += value
        stock_data.loc[stock_data.index[position], 'Cash']   -= value*stock_data.loc[stock_data.index[position], 'Close']


    for date, value in stock_pairs:
        stock_data.loc[stock_data.index >= cash_date, 'Cash'] += cash

    
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
        

    return stock_data



date_stocks_pairs = [
    ('2010-06-30', 4),
    ('2010-07-03', 1),
    ('2025-04-02', -5)
]

cash_pairs = [
    ('2010-06-30', 4),
    ('2010-07-03', 1),
    ('2025-04-02', -5)
]



tsla_data = get_tesla_stock_data()
tsla_data = get_tesla_stock_data()
tsla_data = stock_transaction(tsla_data, date_stocks_pairs)



# for date, value in date_stocks_pairs:
#     tsla_data.loc[tsla_data.index >= date, 'Shares'] += value

tsla_data
