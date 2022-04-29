import yfinance as yf
import pandas as pd
import sys
from stocksymbol import StockSymbol


def stock_prices(market, number_of_stocks,start_date,end_date,interval):
    """
    Function to return a universe of stocks from a selected market for
    a given start date, end date and interval (granularity).
    
    :param market: (str) The market to select the universe of stocks from.
    :param start_date: (str) Start date of the price dataframe.
    :param end_date: (str) End date of the price dataframe.
    :param interval: (str) Interval of price data to be collected.
    :return df : (pandas.DataFrame) Dataframe of universe of stocks 
    
    """   
    
    #Getting api_key  to download stocks
    api_key = '1c2d4795-5aa2-4787-a060-f762dc0fd274'
    ss = StockSymbol(api_key)
    
    try:
        tickers = ss.get_symbol_list(market=market, 
                                     symbols_only=True)[0:number_of_stocks]
    except :
        print('Market not found in database')
        sys.exit()
        
    # Dataframe for universe of stocks where the rows are the prices for the interval
    # and the columns are the tickes of the stocks
    df_prices = pd.DataFrame()
    
    #downloading the price data fram yahoo
    for i in tickers:
    df_prices[i]= yf.download(tickers, start=start_date, end=end_date, 
                               interval = interval)['Adj Close']
          
    # Remove stocks with NaN Values - Happens if tickers are not found in database    
    df_prices = df_prices.dropna(axis=1)
    
    # Remove stocks with zero values - Sometimes happens with missing values or bankruptcy
    df_prices = df_prices.loc[(df_prices!=0).any(axis=1)]
    
    df_prices.to_csv('{}_stocks_price'.format(market))


