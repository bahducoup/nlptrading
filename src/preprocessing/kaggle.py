from os import walk
import pandas as pd

def process_owlcv_kaggle():
    names = ['forbes2000','nasdaq', 'nyse','sp500']
    markets = {}
    for name in names:
        path = 'data/stock_market_data/'+name+'/csv/'
        markets[name] = next(walk(path), (None, None, []))[2]  # [] if no file

    #Takes 30 seconds
    stocks_dataset = pd.DataFrame(columns = ['Symbol', 'Date', 'Low', 'Open', 'Volume', 'High', 'Close', 'Adjusted Close'])

    added_stocks = []
    df_list = []
    for market in markets:
        for stock in markets[market]:
            if(stock not in added_stocks):
                added_stocks.append(stock)
                csv = pd.read_csv('stock_market_data/'+ market +'/csv/'+ stock )
                csv['Symbol'] = stock[:-4]
                df_list.append(csv)
    full_df = pd.concat(df_list, ignore_index = True)

    full_df = full_df.loc[full_df['Volume'] != 0]
    full_df = full_df.loc[full_df['Open'] != 0]
    full_df = full_df.loc[full_df['Volume'].notna()]


    percentages = []
    num_days = []
    num_observations = []
    full_stocks = []
    for symbol, symbol_data in full_df.groupby(by = 'Symbol'):
        num_observations.append(len(symbol_data))
        num_days.append(symbol_data.index[-1] - symbol_data.index[0]+1)
        percentages.append(num_observations[-1]/num_days[-1])
        if((percentages[-1] > .99) and (num_days[-1] > 365)):
            full_stocks.append(symbol)

    full_df = full_df.loc[full_df['Symbol'].isin(full_stocks)]

    full_df['Date'] = pd.to_datetime(full_df['Date'], dayfirst = True)

    train_df = full_df.loc[full_df.Date.dt.year <  2016]
    test_df  = full_df.loc[full_df.Date.dt.year >= 2016]

    train_df.to_csv('../../data/train_df.csv')
    test_df.to_csv('../../data/test_df.csv')