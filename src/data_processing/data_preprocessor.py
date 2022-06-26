import pandas as pd
from stockstats import StockDataFrame as sdf
import numpy as np

class DataProcessor():
    def __init__(self):
        print('hi')
        pass

    def clean_data(self,data): 
        """
        All it does is, drops stocks if they are ever missing. 
        clean the raw data
        deal with missing values
        reasons: stocks could be delisted, not incorporated at the time step
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        df = data.copy()
        df = df.sort_values(["Date", "Symbol"], ignore_index=True)
        df.index = df.Date.factorize()[0]
        merged_closes = df.pivot_table(index="Date", columns="Symbol", values="Close") 
        merged_closes = merged_closes.dropna(axis=1)
        tics = merged_closes.columns
        df = df[df.Symbol.isin(tics)]
        return df
        
    def data_split(self, df, start, end, target_date_col="Date"):
        """
        split the dataset into training or testing using date
        :param data: (df) pandas dataframe, start, end
        :return: (df) pandas dataframe
        """
        data = df[(df[target_date_col] >= start) & (df[target_date_col] < end)]
        data = data.sort_values([target_date_col, "Symbol"], ignore_index=True)
        data.index = data[target_date_col].factorize()[0]
        return data

    def add_technical_indicator(self, data, tech_indicator_list):
        """
        calculate technical indicators
        use stockstats package to add technical inidactors
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        df = data.copy()
        df = df.sort_values(by=["Symbol", "Date"])
        stock = sdf.retype(df.copy())
        unique_ticker = stock.symbol.unique()

        for indicator in tech_indicator_list:
            indicator_df = pd.DataFrame(columns = ['symbol','date',indicator])
            for i in range(len(unique_ticker)):
                try:
                    temp_indicator = stock[stock.symbol == unique_ticker[i]][indicator]
                    temp_indicator = pd.DataFrame(temp_indicator)
                    temp_indicator["symbol"] = unique_ticker[i]
                    temp_indicator["date"] = df[df.Symbol == unique_ticker[i]][
                        "Date"
                    ].to_list()
                    indicator_df = indicator_df.append(
                        temp_indicator, ignore_index=True
                    )
                except Exception as e:
                    print('encounterd an error')
                    print(e)
            indicator_df.rename(columns = {'symbol':'Symbol','date':'Date'}, inplace = True)
            df = df.merge(
                indicator_df[["Symbol", "Date", indicator]], on=["Symbol", "Date"], how="left"
            )
        df = df.sort_values(by=["Date", "Symbol"])
        return df

    def calculate_turbulence(self, data):
        """calculate turbulence index based on dow 30"""
        # can add other market assets
        df = data.copy()
        df_price_pivot = df.pivot(index="Date", columns="Symbol", values="Close")
        # use returns to calculate turbulence
        df_price_pivot = df_price_pivot.pct_change()

        unique_date = df.Date.unique()
        # start after a year
        start = 252
        turbulence_index = [0] * start
        # turbulence_index = [0]
        count = 0
        for i in range(start, len(unique_date)):
            current_price = df_price_pivot[df_price_pivot.index == unique_date[i]]
            # use one year rolling window to calcualte covariance
            hist_price = df_price_pivot[
                (df_price_pivot.index < unique_date[i])
                & (df_price_pivot.index >= unique_date[i - 252])
            ]
            # Drop tickers which has number missing values more than the "oldest" ticker
            filtered_hist_price = hist_price.iloc[
                hist_price.isna().sum().min() :
            ].dropna(axis=1)

            cov_temp = filtered_hist_price.cov()
            current_temp = current_price[[x for x in filtered_hist_price]] - np.mean(
                filtered_hist_price, axis=0
            )
            # cov_temp = hist_price.cov()
            # current_temp=(current_price - np.mean(hist_price,axis=0))

            temp = current_temp.values.dot(np.linalg.pinv(cov_temp)).dot(
                current_temp.values.T
            )
            if temp > 0:
                count += 1
                if count > 2:
                    turbulence_temp = temp[0][0]
                else:
                    # avoid large outlier because of the calculation just begins
                    turbulence_temp = 0
            else:
                turbulence_temp = 0
            turbulence_index.append(turbulence_temp)

        turbulence_index = pd.DataFrame(
            {"Date": df_price_pivot.index, "turbulence": turbulence_index}
        )
        return turbulence_index

    def add_turbulence(self, data):
        """
        add turbulence index from a precalcualted dataframe
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        df = data.copy()
        turbulence_index = self.calculate_turbulence(df)
        df = df.merge(turbulence_index, on="Date")
        df = df.sort_values(["Date", "Symbol"]).reset_index(drop=True)
        return df

    def add_user_defined_feature(self, data):
        """
        add user defined features
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        df = data.copy()
        df["daily_return"] = df.Close.pct_change(1)
        # df['return_lag_1']=df.close.pct_change(2)
        # df['return_lag_2']=df.close.pct_change(3)
        # df['return_lag_3']=df.close.pct_change(4)
        # df['return_lag_4']=df.close.pct_change(5)
        return df
