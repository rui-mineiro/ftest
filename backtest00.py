import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

def get_tesla_stock_data():
    """
    Fetch all available Tesla stock data from Yahoo Finance from the first available date until today
    """
    tesla = yf.Ticker("TSLA")
    
    # Get the maximum available history
    hist = tesla.history(period="max")
    
    return hist




def deposit_portfolio(stock_data, deposit=0, deposit_date='2012-07-21'):
    """
    Deposit money in portfolio
    """
    
    # Convert dates to datetime if they're strings
    if isinstance(deposit_date, str):
        deposit_date = datetime.strptime(deposit_date, '%Y-%m-%d').date()
    
    # Process deposit
    for date, row in stock_data.iterrows():
        current_date = date.date()
        
        # Add cash from current_date util end
        if current_date >= deposit_date:
            stock_data.loc[date, 'Cash'] += deposit
            
    
    return stock_data





def shares_portfolio(stock_data, stock_shares ):
    """
    Calculate portfolio value over time based on transactions
    """

    for date,row in stock_shares.iterrows():
            if isinstance(stock_shares, str):
                stock_shares_datetime = datetime.strptime(stock_date, '%Y-%m-%d').date()

    # Convert dates to datetime if they're strings
    if isinstance(stock_date, str):
        stock_datetime = datetime.strptime(stock_date, '%Y-%m-%d').date()
    
    
    # Process transactions
    for date, row in stock_data.iterrows():
        current_datetime = date.date()
        
        # Buy shares on buy_date
        if current_datetime == stock_datetime:
            cost = row['Close'] * shares
            stock_data=deposit_portfolio(stock_data, deposit=-cost, deposit_date=stock_date)
            
        if current_datetime >= stock_datetime:
            stock_data.loc[date, 'Shares']          += shares
            stock_data.loc[date, 'Portfolio Value'] =  stock_data.loc[date, 'Portfolio Value']+stock_data.loc[date, 'Shares']*stock_data.loc[date, 'Close']
    
    return stock_data


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

def display_transaction_details(portfolio, buy_date, sell_date):
    """
    Display details about the transactions and final portfolio
    """
    # Get buy and sell prices
    buy_price = portfolio.loc[portfolio.index.date == buy_date, 'Close'].values[0]
    sell_price = portfolio.loc[portfolio.index.date == sell_date, 'Close'].values[0]
    
    # Get initial and final values
    initial_value = portfolio.iloc[0]['Portfolio Value']
    final_value = portfolio.iloc[-1]['Portfolio Value']
    return_pct = (final_value - initial_value) / initial_value * 100
    
    print("\n=== Transaction Details ===")
    print(f"Initial Deposit: ${initial_value:.2f}")
    print(f"\nBuy Date: {buy_date}")
    print(f"Bought 7 shares at ${buy_price:.2f} per share")
    print(f"Total Buy Cost: ${7 * buy_price:.2f}")
    
    print(f"\nSell Date: {sell_date}")
    print(f"Sold 4 shares at ${sell_price:.2f} per share")
    print(f"Total Sell Proceeds: ${4 * sell_price:.2f}")
    
    print("\n=== Final Portfolio ===")
    print(f"Final Cash: ${portfolio.iloc[-1]['Cash']:.2f}")
    print(f"Final Shares: {portfolio.iloc[-1]['Shares']}")
    print(f"Final Portfolio Value: ${final_value:.2f}")
    print(f"Total Return: {return_pct:.2f}%")
    
    # Calculate annualized return
    years = (portfolio.index[-1] - portfolio.index[0]).days / 365.25
    annualized_return = ((final_value / initial_value) ** (1/years) - 1) * 100
    print(f"Annualized Return: {annualized_return:.2f}%")

def main():
    print("Loading ALL available Tesla (TSLA) stock data and simulating portfolio...")
    
    # Get all available stock data
    tsla_data = get_tesla_stock_data()
    
    if tsla_data.empty:
        print("Failed to fetch data. Please check your internet connection.")
        return

    tsla_data['Cash']=0
    tsla_data['Shares']=0
    tsla_data['Portfolio Value']=0
    
    # Print data range
    print(f"\nData range: {tsla_data.index[0].date()} to {tsla_data.index[-1].date()}")
    
    
    tsla_data=deposit_portfolio(tsla_data, deposit=1000, deposit_date='2010-07-11')
    tsla_data=deposit_portfolio(tsla_data, deposit=1000, deposit_date='2025-04-01')
    tsla_data=deposit_portfolio(tsla_data, deposit=-1000, deposit_date='2025-04-03')


    # Calculate portfolio values
    tsla_data = shares_portfolio(tsla_data, stock_date='2011-08-11' , shares=4)
    tsla_data = shares_portfolio(tsla_data, stock_date='2015-08-11' , shares=-4)
    # Display transaction details
    # display_transaction_details(portfolio, buy_date, sell_date)
    
    # Plot both stock price and portfolio value
    plot_stock_and_portfolio(tsla_data)
    
    # Save data to CSV
    portfolio.to_csv('tesla_full_history_portfolio.csv')
    print("\nData saved to 'tesla_full_history_portfolio.csv'")

if __name__ == "__main__":
    try:
        import yfinance
        import matplotlib
    except ImportError:
        print("Installing required packages...")
        import subprocess
        subprocess.check_call(["pip", "install", "yfinance", "matplotlib", "pandas"])
    
    main()
