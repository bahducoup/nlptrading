from copy import deepcopy
import pandas as pd
from pyfolio import timeseries
from pypfopt.efficient_frontier import EfficientFrontier



def get_daily_return(df, value_col_name="account_value"):
    df = deepcopy(df)
    df["daily_return"] = df[value_col_name].pct_change(1)
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True, drop=True)
    df.index = df.index.tz_localize("UTC")
    return pd.Series(df["daily_return"], index=df.index)


def backtest_stats(account_value, value_col_name="account_value"):
    dr_test = get_daily_return(account_value, value_col_name=value_col_name)
    perf_stats_all = timeseries.perf_stats(
        returns=dr_test,
        positions=None,
        transactions=None,
        turnover_denom="AGB",
    )
    print(perf_stats_all)
    return perf_stats_all

def convert_daily_return_to_pyfolio_ts(df):
    strategy_ret = df.copy()
    strategy_ret["Date"] = pd.to_datetime(strategy_ret["Date"])
    strategy_ret.set_index("Date", drop=False, inplace=True)
    strategy_ret.index = strategy_ret.index.tz_localize("UTC")
    del strategy_ret["Date"]
    return pd.Series(strategy_ret["daily_return"].values, index=strategy_ret.index)


def calculate_portfolio_minimum_variance(df, unique_trade_date, initial_capital= 1000000):
    portfolio = pd.DataFrame(index = range(1), columns = unique_trade_date)
    initial_capital = 1000000
    portfolio.loc[0,unique_trade_date[0]] = initial_capital

    for i in range(len( unique_trade_date)-1):
        df_temp = df[df.Date==unique_trade_date[i]].reset_index(drop=True)
        df_temp_next = df[df.Date==unique_trade_date[i+1]].reset_index(drop=True)
        #Sigma = risk_models.sample_cov(df_temp.return_list[0])
        #calculate covariance matrix
        #Sigma = df_temp.return_list[0].cov()
        Sigma = df_temp.return_list[0].cov()
        #portfolio allocation
        ef_min_var = EfficientFrontier(None, Sigma,weight_bounds=(0, 0.1))
        #minimum variance
        raw_weights_min_var = ef_min_var.min_volatility()
        #get weight
        cleaned_weights_min_var = ef_min_var.clean_weights()
        
        #current capital
        cap = portfolio.iloc[0, i]
        #current cash invested for each stock
        current_cash = [element * cap for element in list(cleaned_weights_min_var.values())]
        # current held shares
        current_shares = list(np.array(current_cash)
                                        / np.array(df_temp.Close))
        # next time period price
        next_price = np.array(df_temp_next.Close)
        ##next_price * current share to calculate next total account value 
        portfolio.iloc[0, i+1] = np.dot(current_shares, next_price)
        
    portfolio=portfolio.T
    portfolio.columns = ['account_value']
    return portfolio