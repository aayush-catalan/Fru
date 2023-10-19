from scipy import stats
from collections import OrderedDict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
from typing import Dict, List
from collections import OrderedDict
from models import ( 
    CatboostForecaster,
    TimeSeriesForecaster, 
    RandomForestForecaster, 
    GradientBoostingRegressorForecaster,
    # LightGBMForecaster,
    # ProphetForecaster,
    XGBoostForecaster
)
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from scipy import stats
import matplotlib.pyplot as plt
import pickle
from statsmodels.tsa.seasonal import seasonal_decompose
import pytz
from datetime import datetime

def add_inflation(new_df):
    values = [
        3.68, 3.37, 3.14, 3.13, 3.16, 3.2, 3.12, 3.1, 3.23, 3.33, 3.27, 3.18, 3.15, 3.01, 3.21, 3.25, 3.31, 3.43, 3.79, 3.75,
        3.82, 3.86, 3.84, 3.8, 3.62, 3.72, 3.86, 3.51, 2.85, 2.19, 1.97, 1.88, 1.97, 1.75, 1.49, 1.61, 1.6, 1.56, 1.51, 1.95,
        3.3, 3.63, 3.97, 4.44, 4.51, 4.58, 5.26, 5.62, 6.94, 8.01, 8.53, 9.23, 9.07, 9.67, 10.21, 10.84, 11.44, 12.22, 12.53,
        13.12, 13.25, 13.28, 13.34, 12.82, 12.36, 12.13, 11.78, 11.43,10.79
    ]

    # Create a list of dates from Jan 2018 to Aug 2023
    date_range = pd.date_range(start='2018-01-01', end='2023-09-30', freq='D')

    # Create a DataFrame

    inflation = []
    df = pd.DataFrame()
    df['date'] = date_range
    i = 0
    prev_month = df['date'][0].month
    # Populate the 'Price' column with values based on the corresponding month's value
    for date in df['date']:
        month = date.month
        if prev_month != month:
            prev_month = month
            i+=1
        inflation.append(values[i])
        # df.at[date, 'Price'] = value

    # Print the resulting DataFrame
    df['inflation'] = inflation
    # print(df)
    # df = df.set_index('date')
    df['date'] = pd.to_datetime(df['date'])
    new_df['date'] = pd.to_datetime(new_df['date'])
    # filtered2 = pd.merge(filtered2, df, how='left', left_index=True, right_index=True)
    merged = pd.merge(new_df, df, on='date', how='left')
    return merged

def get_filterd_dataset(product_name):

    new_df = pd.read_csv('historic_pilot_info_catalan.csv')
    # new_df = new_df.dropna(how='all')

    filtered = new_df[new_df['product_name']==product_name]
    filtered['order_submited_datetime'] = pd.to_datetime(filtered['order_submited_datetime'])
    return filtered

from scipy import stats
def mode(x):
    return stats.mode(x)[0][0] 

def get_papa_grande():
    df = get_filterd_dataset('Fresa Pequeño Kg')
    colombia_tz = pytz.timezone('America/Bogota')

    # Convert the 'order_submited_datetime' column to Colombia timezone
    if df['order_submited_datetime'].dt.tz is None:
    # If the datetime is naive, localize it
        df['order_submited_datetime'] = df['order_submited_datetime'].dt.tz_localize(pytz.utc).dt.tz_convert(colombia_tz)
    else:
        # If the datetime is already timezone-aware, just convert it
        df['order_submited_datetime'] = df['order_submited_datetime'].dt.tz_convert(colombia_tz)

    df['order_submited_datetime'] = pd.to_datetime(df['order_submited_datetime'])
    df['hour'] = df['order_submited_datetime'].dt.hour
    df['date'] = df['order_submited_datetime'].dt.date
    df['month'] = df['order_submited_datetime'].dt.month
    df['day_of_week_n'] = df['order_submited_datetime'].dt.day_of_week
    df['day_of_month'] = df['order_submited_datetime'].dt.day
    df['year'] = df['order_submited_datetime'].dt.year
    agg_df = df.groupby(['date', 'hour']).agg({
    'net_price': mode,
    'quantity': 'sum',
    'cost': 'first',
    'avg_bench': 'first',
    'product_id': 'first',
    'product_name': 'first',
    'gross_price': 'first',
    'day_of_week_n': 'first',
    'day_of_month': 'first',
    # 'date': 'first'
    'year':'first',
    'month': 'first'

    }).reset_index()

    min_date = pd.to_datetime(agg_df['date']).min()
    max_date = pd.to_datetime(agg_df['date']).max()

    date_range = pd.date_range(start=min_date, end=max_date, freq='D')
    hour_range = list(range(24))

    full_df = pd.DataFrame([(date, hour) for date in date_range for hour in hour_range], columns=['date', 'hour'])
    full_df['date'] = full_df['date'].dt.date.astype(str)  # Convert datetime to string to match df's 'date' format
    temp = agg_df[['date','hour']]
    temp_unique = temp.drop_duplicates()
    temp_unique['date'] = pd.to_datetime(temp_unique['date'])
    temp_unique['date'] = temp_unique['date'].dt.date.astype(str)
    # Identify missing rows
    merged_df = pd.merge(full_df, temp_unique, on=['date', 'hour'], how='left', indicator=True)
    missing_rows = merged_df[merged_df['_merge'] == 'left_only'][['date', 'hour']]
    # missing_rows.head(23)
    agg_df['date'] = pd.to_datetime(agg_df['date'])
    agg_df['date'] = agg_df['date'].dt.date.astype(str)
    
    merged_df = pd.merge(agg_df, missing_rows, on=['date', 'hour'], how='outer')
    cols_to_fill = [col for col in merged_df.columns if col not in ['quantity']]
    merged_df[cols_to_fill] = merged_df[cols_to_fill].fillna(method='ffill')
    merged_df['quantity'].fillna(0, inplace=True)
    merged_df['net_price_FP'] = merged_df['net_price']
    merged_df['quantity_FP'] = merged_df['quantity']
    merged_df['net_price_FP_lag48'] = merged_df['net_price_FP'].shift(48)
    merged_df['net_price_FP_lag49'] = merged_df['net_price_FP'].shift(49)
    merged_df['quantity_FP_lag48'] = merged_df['quantity_FP'].shift(48)
    merged_df['quantity_FP_lag49'] = merged_df['quantity_FP'].shift(49)

    # if not comp
    daily_quantity_papagrande = merged_df.groupby('date').agg({'quantity': 'sum','net_price': mode}).reset_index()

    daily_quantity_papagrande['FP_rolling_avg_quantity_7'] = (
        daily_quantity_papagrande['quantity']
        .rolling(window=7)
        .mean()
        .shift(2)  # This adds the 1-day lag
    )

    daily_quantity_papagrande['FP_rolling_avg_quantity_30'] = (
        daily_quantity_papagrande['quantity']
        .rolling(window=30)
        .mean()
        .shift(2)  # This adds the 1-day lag
    )

    daily_quantity_papagrande['FP_quantity_daily_lag2'] = (
            daily_quantity_papagrande['quantity']
            .shift(2)  # This adds the 1-day lag
        )
    daily_quantity_papagrande['FP_net_price_daily_lag2'] = (
            daily_quantity_papagrande['net_price']
            .shift(2)  # This adds the 1-day lag
        )

    daily_quantity_papagrande['FP_net_price_daily_lag1'] = (
            daily_quantity_papagrande['net_price']
            .shift(1)  # This adds the 1-day lag
        )

    daily_quantity_papagrande['FP_daily_net_price'] = daily_quantity_papagrande['net_price']
    daily_quantity_papagrande['FP_daily_quantity'] = daily_quantity_papagrande['quantity']
    
    return daily_quantity_papagrande, merged_df


def aggregate_by_hour(df, time_bucket, has_competitor_price):
    """
    Aggregate a DataFrame by a specified time bucket (e.g., 24 for daily aggregation).
    
    Args:
        df (pd.DataFrame): Input DataFrame with a datetime field.
        time_bucket (int): Number of hours in each time bucket for aggregation.
    
    Returns:
        pd.DataFrame: Aggregated DataFrame with the datetime bucketed and values aggregated.
    """
    # Create a timezone object for Colombia (CST)
    colombia_tz = pytz.timezone('America/Bogota')
    
    # Convert the 'order_submited_datetime' column to Colombia timezone
    df['order_submited_datetime'] = df['order_submited_datetime'].dt.tz_localize(pytz.utc).dt.tz_convert(colombia_tz)
    
    df['order_submited_datetime'] = pd.to_datetime(df['order_submited_datetime'])
    df['hour'] = df['order_submited_datetime'].dt.hour
    df['date'] = df['order_submited_datetime'].dt.date
    df['month'] = df['order_submited_datetime'].dt.month
    df['day_of_week_n'] = df['order_submited_datetime'].dt.day_of_week
    df['day_of_month'] = df['order_submited_datetime'].dt.day
    df['year'] = df['order_submited_datetime'].dt.year
    agg_df = df.groupby(['date', 'hour', 'net_price']).agg({
    'quantity': 'sum',
    'cost': 'first',
    'avg_bench': 'first',
    'product_id': 'first',
    'product_name': 'first',
    'gross_price': 'first',
    'day_of_week_n': 'first',
    'day_of_month': 'first',
    # 'date': 'first'
    'year':'first',
    'month': 'first'

}).reset_index()

    min_date = pd.to_datetime(agg_df['date']).min()
    max_date = pd.to_datetime(agg_df['date']).max()

    date_range = pd.date_range(start=min_date, end=max_date, freq='D')
    hour_range = list(range(24))

    full_df = pd.DataFrame([(date, hour) for date in date_range for hour in hour_range], columns=['date', 'hour'])
    full_df['date'] = full_df['date'].dt.date.astype(str)  # Convert datetime to string to match df's 'date' format
    temp = agg_df[['date','hour']]
    temp_unique = temp.drop_duplicates()
    temp_unique['date'] = pd.to_datetime(temp_unique['date'])
    temp_unique['date'] = temp_unique['date'].dt.date.astype(str)
    # Identify missing rows
    merged_df = pd.merge(full_df, temp_unique, on=['date', 'hour'], how='left', indicator=True)
    missing_rows = merged_df[merged_df['_merge'] == 'left_only'][['date', 'hour']]
    # missing_rows.head(23)
    agg_df['date'] = pd.to_datetime(agg_df['date'])
    agg_df['date'] = agg_df['date'].dt.date.astype(str)
    merged_df = pd.merge(agg_df, missing_rows, on=['date', 'hour'], how='outer')
    cols_to_fill = [col for col in merged_df.columns if col not in ['quantity']]
    merged_df[cols_to_fill] = merged_df[cols_to_fill].fillna(method='ffill')
    merged_df['quantity'].fillna(0, inplace=True)

    # if not comp
    daily_quantity = merged_df.groupby('date').agg({'quantity': 'sum', 'net_price': mode}).reset_index()

    result = seasonal_decompose(daily_quantity['quantity'], model='additive', period=7)
    daily_quantity['residual_daily'] = result.resid
    daily_quantity['trend_daily'] = result.trend
    daily_quantity['seasonal_daily'] = result.seasonal
    daily_quantity['residual_daily_lag2'] = daily_quantity['residual_daily'].shift(2)
    daily_quantity['trend_daily_lag2'] = daily_quantity['trend_daily'].shift(2)
    daily_quantity['seasonal_daily_lag2'] = daily_quantity['seasonal_daily'].shift(2)
    # Step 2: Calculate 7-Day Rolling Average with 1-day Lag
    daily_quantity['rolling_avg_quantity_7'] = (
        daily_quantity['quantity']
        .rolling(window=7)
        .mean()
        .shift(2)  # This adds the 1-day lag
    )

    daily_quantity['rolling_avg_quantity_30'] = (
        daily_quantity['quantity']
        .rolling(window=30)
        .mean()
        .shift(2)  # This adds the 1-day lag
    )

    daily_quantity['quantity_daily_lag2'] = (
        daily_quantity['quantity']
        .shift(2)  # This adds the 1-day lag
    )

    daily_quantity['net_price_daily_lag1'] = (
        daily_quantity['net_price']
        .shift(1)  # This adds the 1-day lag
    )

    daily_quantity['net_price_daily_lag2'] = (
        daily_quantity['net_price']
        .shift(2)  # This adds the 1-day lag
    )
    # daily_quantity[]

    def ewm_features(dataframe, alphas, lags):
      for alpha in alphas:
          for lag in lags:
              dataframe['quantity_ewm_alpha_' + str(alpha).replace(".", "") + "_lag_" + str(lag)] = \
                  dataframe['quantity'].transform(lambda x: x.shift(lag).ewm(alpha=alpha).mean())
      dataframe.fillna(0, inplace=True)
      return dataframe
    
    alphas = [0.95, 0.5]
    lags   = [2,7,10,14,30,60]


    daily_quantity= ewm_features(daily_quantity, alphas, lags)

    merged_df = merged_df.merge(daily_quantity[['date', 'quantity_ewm_alpha_095_lag_2', 'quantity_ewm_alpha_095_lag_7','quantity_ewm_alpha_095_lag_10','quantity_ewm_alpha_095_lag_14'
                                                ,'quantity_ewm_alpha_095_lag_30','quantity_ewm_alpha_095_lag_60','quantity_ewm_alpha_05_lag_2', 'quantity_ewm_alpha_05_lag_7','quantity_ewm_alpha_05_lag_10','quantity_ewm_alpha_05_lag_14'
                                                ,'quantity_ewm_alpha_05_lag_30','quantity_ewm_alpha_05_lag_60',
                                                'rolling_avg_quantity_30','rolling_avg_quantity_7','quantity_daily_lag2','trend_daily_lag2','seasonal_daily_lag2','residual_daily_lag2', 'net_price_daily_lag1','net_price_daily_lag2']], on='date', how='left')
    
    daily_quantity_FP, hourly_FP = get_papa_grande()
    merged_df = merged_df.merge(daily_quantity_FP[['date','FP_net_price_daily_lag1','FP_net_price_daily_lag2', 'FP_quantity_daily_lag2','FP_rolling_avg_quantity_30','FP_rolling_avg_quantity_7','FP_daily_quantity','FP_daily_net_price']], on='date', how='left')

    merged_df = merged_df.merge(hourly_FP[['date','hour','net_price_FP_lag48','quantity_FP_lag48','net_price_FP_lag49','quantity_FP_lag49','net_price_FP','quantity_FP']], on=['date', 'hour'], how='left')

    merged_df = merged_df.sort_values(['date','hour'])
    
    return merged_df

def create_features(dataframe, has_competitor_price):
    def is_weekend(day):
        return 1 if day in [5, 6] else 0

    dataframe['if_weekend'] = dataframe['day_of_week_n'].apply(is_weekend)
    for day_num in range(0, 7):
        day_col_name = f'net_price_day_{day_num}'
        dataframe[day_col_name] = dataframe.apply(lambda row: row['net_price'] if row['day_of_week_n'] == day_num else 0, axis=1)


    dataframe = add_inflation(dataframe)
    # dataframe.set_index('bucketed_datetime', inplace=True)
    # dataframe.index.freq = 'D' 
    # dataframe = dataframe.dropna()
    print('features ', dataframe.info())
    return dataframe


