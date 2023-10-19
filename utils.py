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
    LightGBMForecaster,
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

def add_to_Historic_Dataframe(new_data,date):
  # new_data = pd.read_csv('data/Frubana_2023_10_09.csv')

  historical =  pd.read_csv('historic_pilot_info_catalan.csv')
  colombia_tz = pytz.timezone('America/Bogota')
    # Convert the 'order_submited_datetime' column to Colombia timezone
  historical['order_submited_datetime'] = pd.to_datetime(historical['order_submited_datetime'])
  historical['order_submited_datetime'] = historical['order_submited_datetime'].dt.tz_localize(pytz.utc).dt.tz_convert(colombia_tz)
  
  new_data['order_submited_datetime'] = pd.to_datetime(new_data['order_submited_datetime'])
  new_data['order_submited_datetime'] = new_data['order_submited_datetime']   # .dt.tz_localize(pytz.utc).dt.tz_convert(colombia_tz)
  new_data['order_submited_date'] = new_data['order_submited_datetime'].dt.date.astype(str)
  
  new_data = new_data[new_data['order_submited_date']>'2023-09-29']


  common_columns = new_data.columns.intersection(historical.columns)
  merged_df = pd.concat([historical[common_columns], new_data[common_columns]])

  merged_df.sort_values('order_submited_datetime').to_csv(f'data/created/Catalan_{date}_temp.csv')

  return merged_df.sort_values('order_submited_datetime')



def generate_datafrane_for_profit(date, price, columns):
    test_date = pd.to_datetime(date)
    dates = [test_date.date()] * 24
    prices = [price] * 24
    hours = list(range(24))
    
 
    # Create DataFrame
    df = pd.DataFrame({'datetime': dates, 'hour': hours, 'net_price': prices})
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['date'] = df['datetime'].dt.date
    df['day_of_week_n'] = df['datetime'].dt.day_of_week

    for day_num in range(0, 7):
        day_col_name = f'net_price_day_{day_num}'
        df[day_col_name] = df.apply(lambda row: row['net_price'] if row['day_of_week_n'] == day_num else 0, axis=1)
    merged_Df = add_inflation(df)[columns[:-1]]
    
    return merged_Df

def generate_datafrane_for_profit2(date, price):
    test_date = pd.to_datetime(date)
    dates = [test_date.date()] * 24
    prices = [price] * 24
    hours = list(range(24))
    
 
    # Create DataFrame
    df = pd.DataFrame({'datetime': dates, 'hour': hours, 'net_price': prices})
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['date'] = df['datetime'].dt.date
    df['day_of_week_n'] = df['datetime'].dt.day_of_week

    for day_num in range(0, 7):
        day_col_name = f'net_price_day_{day_num}'
        df[day_col_name] = df.apply(lambda row: row['net_price'] if row['day_of_week_n'] == day_num else 0, axis=1)
    # merged_Df = add_inflation(df)[columns[:-1]]
    df['date'] = df['date'].astype(str)
    merged_Df = df
    
    return merged_Df.sort_values(['date','hour'])

def add_inflation(new_df):
    values = [
        3.68, 3.37, 3.14, 3.13, 3.16, 3.2, 3.12, 3.1, 3.23, 3.33, 3.27, 3.18, 3.15, 3.01, 3.21, 3.25, 3.31, 3.43, 3.79, 3.75,
        3.82, 3.86, 3.84, 3.8, 3.62, 3.72, 3.86, 3.51, 2.85, 2.19, 1.97, 1.88, 1.97, 1.75, 1.49, 1.61, 1.6, 1.56, 1.51, 1.95,
        3.3, 3.63, 3.97, 4.44, 4.51, 4.58, 5.26, 5.62, 6.94, 8.01, 8.53, 9.23, 9.07, 9.67, 10.21, 10.84, 11.44, 12.22, 12.53,
        13.12, 13.25, 13.28, 13.34, 12.82, 12.36, 12.13, 11.78, 11.43,10.79,10.79
    ]

    # Create a list of dates from Jan 2018 to Aug 2023
    date_range = pd.date_range(start='2018-01-01', end='2023-10-30', freq='D')

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

    # Print the resulting DataFrame
    df['inflation'] = inflation

    df['date'] = pd.to_datetime(df['date'])
    new_df['date'] = pd.to_datetime(new_df['date'])
    # filtered2 = pd.merge(filtered2, df, how='left', left_index=True, right_index=True)
    merged = pd.merge(new_df, df, on='date', how='left')
    return merged


def get_filterd_dataset(product_name):

    new_df = pd.read_csv('data/master/master_historical_till_2023_10_17.csv')
    # new_df = new_df.dropna(how='all')

    filtered = new_df[new_df['product_name']==product_name]
    filtered['order_submited_datetime'] = pd.to_datetime(filtered['order_submited_datetime'])
    return filtered

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
    agg_df = df.groupby(['date', 'hour', 'net_price']).agg({
    'quantity': 'sum',
    'cost': 'first',
    'avg_bench': 'first',
    'product_id': 'first',
    'product_name': 'first',
    'gross_price': 'first',
    'day_of_week_n': 'first',
    'day_of_month': 'first',
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
    agg_df['date'] = pd.to_datetime(agg_df['date'])
    agg_df['date'] = agg_df['date'].dt.date.astype(str)
    merged_df = pd.merge(agg_df, missing_rows, on=['date', 'hour'], how='outer')
    cols_to_fill = [col for col in merged_df.columns if col not in ['quantity']]
    merged_df = merged_df.sort_values(['date','hour'])
    
    merged_df['net_price'] = merged_df['net_price'].fillna(method='ffill')
    merged_df['quantity'].fillna(0, inplace=True)

    # if not comp
    daily_quantity_papagrande = merged_df.groupby('date').agg({'quantity': 'sum','net_price': 'mean'}).reset_index()
    daily_quantity_papagrande

    daily_quantity_papagrande['papagrande_quantity_daily_lag2'] = (
            daily_quantity_papagrande['quantity']
            .shift(2)  # This adds the 1-day lag
        )
    daily_quantity_papagrande['papagrande_net_price_daily_lag2'] = (
            daily_quantity_papagrande['net_price']
            .shift(2)  # This adds the 1-day lag
        )
    return daily_quantity_papagrande


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
    # df['order_submited_datetime'] = df['order_submited_datetime'].dt.tz_localize(pytz.utc).dt.tz_convert(colombia_tz)
    
    df['order_submited_datetime'] = df['order_submited_datetime'].dt.tz_convert(colombia_tz)

    df['order_submited_datetime'] = pd.to_datetime(df['order_submited_datetime'])
    df['hour'] = df['order_submited_datetime'].dt.hour
    df['date'] = df['order_submited_datetime'].dt.date
    df['month'] = df['order_submited_datetime'].dt.month
    df['day_of_month'] = df['order_submited_datetime'].dt.day
    df['year'] = df['order_submited_datetime'].dt.year
    agg_df = df.groupby(['date', 'hour', 'net_price']).agg({
    'quantity': 'sum',
    'cost': 'first',
    'avg_bench': 'first',
    'product_id': 'first',
    'product_name': 'first',
    'gross_price': 'first',
   
    'day_of_month': 'first',
   
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
    new_date = merged_df.iloc[-1]['date']
    new_price = merged_df.iloc[-1]['net_price']


    date_format = '%Y-%m-%d'
    date_DATE = datetime.strptime(new_date, date_format)


    # Add one day to the date
    next_date = date_DATE + timedelta(days=1)
    next_date_str = next_date.strftime(date_format)
    print('next date',next_date_str)
    test_vector_copy = generate_datafrane_for_profit2(next_date_str,new_price )
    merged_df = merged_df.sort_values(['date','hour'])
    merged_df = pd.concat([ merged_df, test_vector_copy])
   
    test_vector_copy = generate_datafrane_for_profit2('2023-10-19',new_price )
  
    merged_df = merged_df.sort_values(['date','hour'])
    merged_df = pd.concat([ merged_df, test_vector_copy])
  

    cols_to_fill = [col for col in merged_df.columns if col not in ['quantity']]
    merged_df = merged_df.sort_values(['date','hour'])
    merged_df['net_price'] = merged_df['net_price'].fillna(method='ffill')
         
    merged_df['quantity'].fillna(0, inplace=True)
    merged_df.to_csv('temp2.csv')
   
    # if not comp
    daily_quantity = merged_df.groupby('date').agg({'quantity': 'sum', 'net_price': 'mean'}).reset_index()
   
    result = seasonal_decompose(daily_quantity['quantity'], model='additive', period=7)
    daily_quantity['residual_daily'] = result.resid
    daily_quantity['trend_daily'] = result.trend
    daily_quantity['seasonal_daily'] = result.seasonal
    daily_quantity['residual_daily_lag2'] = daily_quantity['residual_daily'].shift(2)
    daily_quantity['trend_daily_lag2'] = daily_quantity['trend_daily'].shift(2)
    daily_quantity['seasonal_daily_lag2'] = daily_quantity['seasonal_daily'].shift(2)
   
    # Step 2: Calculate 7-Day Rolling Average with 1-day Lag
    daily_quantity['rolling_avg_quantity_7_lag2'] = (
        daily_quantity['quantity']
        .rolling(window=7)
        .mean()
        .shift(2)  # This adds the 1-day lag
    )

    daily_quantity['rolling_avg_quantity_10_lag2'] = (
        daily_quantity['quantity']
        .rolling(window=10)
        .mean()
        .shift(2)  # This adds the 1-day lag
    )

    daily_quantity['rolling_avg_quantity_15_lag2'] = (
        daily_quantity['quantity']
        .rolling(window=15)
        .mean()
        .shift(2)  # This adds the 1-day lag
    )

    daily_quantity['rolling_avg_quantity_30_lag2'] = (
        daily_quantity['quantity']
        .rolling(window=30)
        .mean()
        .shift(2)  # This adds the 1-day lag
    )

    daily_quantity['quantity_daily_lag2'] = (
        daily_quantity['quantity']
        .shift(2)  # This adds the 1-day lag
    )

    daily_quantity['ratio_rolling_avg_quantity_30_7_lag2'] = daily_quantity['rolling_avg_quantity_30_lag2']/daily_quantity['rolling_avg_quantity_7_lag2']
    

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

    merged_df = merged_df.merge(daily_quantity[['date', 'ratio_rolling_avg_quantity_30_7_lag2','quantity_ewm_alpha_095_lag_2', 'quantity_ewm_alpha_095_lag_7','quantity_ewm_alpha_095_lag_10','quantity_ewm_alpha_095_lag_14'
                                                ,'quantity_ewm_alpha_095_lag_30','quantity_ewm_alpha_095_lag_60','quantity_ewm_alpha_05_lag_2', 'quantity_ewm_alpha_05_lag_7','quantity_ewm_alpha_05_lag_10','quantity_ewm_alpha_05_lag_14'
                                                ,'quantity_ewm_alpha_05_lag_30','quantity_ewm_alpha_05_lag_60',
                                                'rolling_avg_quantity_30_lag2','rolling_avg_quantity_15_lag2','rolling_avg_quantity_10_lag2','rolling_avg_quantity_7_lag2','quantity_daily_lag2','trend_daily_lag2','seasonal_daily_lag2','residual_daily_lag2']], on='date', how='left')
    
    merged_df = merged_df.sort_values(['date','hour'])
    
    
    return merged_df


from scipy import stats

def aggregate_by_hour_High_Selling(df, time_bucket, has_competitor_price):
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
    df['order_submited_datetime'] = pd.to_datetime(df['order_submited_datetime'])
    
    # # Convert the 'order_submited_datetime' column to Colombia timezone
    df['order_submited_datetime'] = df['order_submited_datetime'].dt.tz_localize(pytz.utc).dt.tz_convert(colombia_tz)
    
    df['order_submited_datetime'] = pd.to_datetime(df['order_submited_datetime'])
    df['hour'] = df['order_submited_datetime'].dt.hour
    df['date'] = df['order_submited_datetime'].dt.date
    df['month'] = df['order_submited_datetime'].dt.month
    # df['day_of_week_n'] = df['order_submited_datetime'].dt.day_of_week
    df['day_of_month'] = df['order_submited_datetime'].dt.day
    df['year'] = df['order_submited_datetime'].dt.year

#     # Step 1: Aggregate total quantity for each unique 'net_price' within each 'date' and 'hour' group
    quantity_sum = df.groupby(['date', 'hour', 'net_price']).agg({'quantity': 'sum'}).reset_index()

    # Step 2: Within each 'date' and 'hour' group, find the 'net_price' with the highest total quantity
    idx = quantity_sum.groupby(['date', 'hour'])['quantity'].idxmax()
    max_quantity_net_price = quantity_sum.loc[idx, ['date', 'hour', 'net_price']]
    max_quantity_net_price['max_net_price'] = max_quantity_net_price['net_price']
    # Step 3: Merge this information back with the original DataFrame
    merged_df = df.merge(max_quantity_net_price, on=['date', 'hour'])

    print('merged columns',merged_df.columns)
    # Step 4: Group by 'date' and 'hour' again to aggregate the other columns
    agg_df = merged_df.groupby(['date', 'hour']).agg({
        'max_net_price': 'first',
        'quantity': 'sum',
        'cost': 'first',
        'avg_bench': 'first',
        'product_id': 'first',
        'product_name': 'first',
        'gross_price': 'first',
        # 'day_of_week_n': 'first',
        'day_of_month': 'first',
        'year': 'first',
        'month': 'first'
    }).reset_index()
    agg_df['net_price'] = agg_df['max_net_price']

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

    agg_df['date'] = pd.to_datetime(agg_df['date'])
    agg_df['date'] = agg_df['date'].dt.date.astype(str)
    merged_df = pd.merge(agg_df, missing_rows, on=['date', 'hour'], how='outer')
    cols_to_fill = [col for col in merged_df.columns if col not in ['quantity']]
    merged_df[cols_to_fill] = merged_df[cols_to_fill].fillna(method='ffill')
    merged_df['quantity'].fillna(0, inplace=True)


    #  if  comp
    daily_quantity = merged_df.groupby('date').agg({'quantity': 'sum'}).reset_index()

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

    merged_df = merged_df.merge(daily_quantity[['date', 'rolling_avg_quantity_30','rolling_avg_quantity_7','quantity_daily_lag2',]], on='date', how='left')    
    
    daily_quantity_papagrande = get_papa_grande()
    merged_df = merged_df.merge(daily_quantity_papagrande[['date','papagrande_net_price_daily_lag2', 'papagrande_quantity_daily_lag2']], on='date', how='left')

    merged_df = merged_df.sort_values(['date','hour'])
    
    return merged_df



def create_features(dataframe, has_competitor_price):
    def is_weekend(day):
        return 1 if day in [5, 6] else 0

    dataframe = dataframe.sort_values(['date','hour'])
        
    dataframe['quantity'].fillna(0, inplace=True)
    dataframe = add_inflation(dataframe)
   
    result = seasonal_decompose(dataframe['quantity'], model='additive', period=7)
    dataframe['residual'] = result.resid
    dataframe['trend'] = result.trend
    dataframe['seasonal'] = result.seasonal
    dataframe['residual_lag2'] = dataframe['residual'].shift(2)
    dataframe['trend_lag2'] = dataframe['trend'].shift(2)
    dataframe['seasonal_lag2'] = dataframe['seasonal'].shift(2)

    dataframe['datetime'] = pd.to_datetime(dataframe['date'])
    dataframe['day_of_week_n'] = dataframe['datetime'].dt.day_of_week

    dataframe['if_weekend'] = dataframe['day_of_week_n'].apply(is_weekend)
    dataframe['quantity_hourly_lag48'] = dataframe['quantity'].shift(48)
    dataframe['quantity_hourly_lag49'] = dataframe['quantity'].shift(49)

    dataframe['cost_lag1'] = dataframe['cost'].shift(1)
    
    for day_num in range(0, 7):
        day_col_name = f'net_price_day_{day_num}'
        dataframe[day_col_name] = dataframe.apply(lambda row: row['net_price'] if row['day_of_week_n'] == day_num else 0, axis=1)


    dataframe.to_csv('temp.csv')
    return dataframe

def symmetric_mean_absolute_percentage_error(actual, predicted):
    """
    Calculate Symmetric Mean Absolute Percentage Error (SMAPE)

    Parameters:
    actual (numpy array or list): Array of actual values
    predicted (numpy array or list): Array of predicted values

    Returns:
    float: SMAPE
    """
    actual = np.array(actual)
    predicted = np.array(predicted)
    return np.mean(
            2 * np.abs(actual - predicted) / (np.abs(actual) + np.abs(predicted))) * 100

def mda(actual, predicted):
    """ Mean Directional Accuracy """
    actual = np.array(actual)
    predicted = np.array(predicted)
    return np.mean(
        (np.sign(actual[1:] - actual[:-1]) == np.sign(predicted[1:] - actual[:-1]))
        .astype(int))

def save_feature_importances(forecaster, folder_name, sub_folder_name, 
                                product_name, feature_names=None):
    try:
        if isinstance(forecaster.model, GradientBoostingRegressor):
            feature_importances = forecaster.model.feature_importances_
            feature_names = forecaster.model.feature_names_in_
        elif isinstance(forecaster.model, RandomForestRegressor):
            feature_importances = forecaster.model.feature_importances_
            feature_names = forecaster.model.feature_names_in_
        elif hasattr(forecaster.model, 'get_feature_importance'):  # For CatBoost
            feature_importances = forecaster.model.get_feature_importance()
            feature_names = forecaster.model.feature_names_
        else:
            raise AttributeError('Feature importances not available for this model.')

        # Create a DataFrame with the feature names and their corresponding importances
        feature_importances_df = pd.DataFrame({'feature': feature_names,
                                                'importance': feature_importances})

        # Sort the feature importances in descending order
        sorted_feature_importances = feature_importances_df.sort_values(by='importance',
                                                                         ascending=False)

        # Define the output CSV file path
        output_csv_path = f'{folder_name}/Feature_Importances/' \
                          f'{sub_folder_name}/{product_name}.csv'

        # Save the sorted feature importances to a CSV file
        sorted_feature_importances.to_csv(output_csv_path, index=False)

    except AttributeError as e:
        print(f'Error: {e}')

def plot_preds(filtered_df: pd.DataFrame, Preds: list, title: str,
                end_val: int, folder: str, parent_folder_name: str, model_name: str, actual: list):
    pds = Preds
    num_days = len(pds)
    only_test = actual
    print('plot',actual)

    indexes = [i for i in range(end_val)]

    smape = symmetric_mean_absolute_percentage_error(actual, pds)

    mda_ans = mda(actual, pds)
    plt.figure()
    plt.plot(indexes, pds, '#9d52ff', label='Prediction', marker='o', markersize=5)

    # Plot the actual line
    plt.plot(indexes, actual, '#44546a', marker='x',markersize=5,label='Actual')

    # Customize the legend labels
    legend = plt.legend(loc='upper right')
    legend.get_texts()[0].set_text('Catalan')
    legend.get_texts()[1].set_text('Oxxo')
    result = pd.DataFrame()
    result['Prediction'] = pds
    result['Actual'] = actual
    result['Date'] = indexes
    result.to_csv(f'{parent_folder_name}/Predictions/{folder}/{title}_{num_days}_days.csv')
    plt.gcf().set_size_inches(15, 5)
    name = title + f"MDA={mda_ans},SMAPE={smape} {model_name}"

    plt.title(name)
    plt.savefig(f'{parent_folder_name}/Graphs/{folder}/{title}_{num_days}_days.png')
    plt.close()
    metrics = pd.DataFrame()
    metrics['MDA'] = [mda_ans]
    metrics['SMAPE'] = [smape]
    metrics.to_csv(f'{parent_folder_name}/Metrics/{folder}/{title}_{num_days}_days.csv')
    return

def process_day(forcaster, train, test):
    # Separate target from features
    X_train, y_train = train.drop('quantity', axis=1), train['quantity']
    X_test, y_test = test.drop('quantity', axis=1), test['quantity']

    forcaster.fit(y_train, X_train)

    y_pred = forcaster.predict(X_test)

    return y_test.iloc[0], y_pred



def find_best_price(test_prices, dayLevelPredictions, cost):
    profits = []
    # cost = 	3835.1849	
    for i in range(len(test_prices)):
        profit_day = (test_prices[i] - cost)*dayLevelPredictions[i]
        profits.append(profit_day)

    index = profits.index(max(profits))
    return test_prices[index], dayLevelPredictions[index]

def find_demand(test_prices, dayLevelPredictions, cost):
    profits = []
    # cost = 	3835.1849	
    for i in range(len(test_prices)):
        profit_day = (test_prices[i] - cost)*dayLevelPredictions[i]
        profits.append(profit_day)

    index = profits.index(max(profits))
    return test_prices[index], dayLevelPredictions[index]

def generate_demand_graph_daily_testing(features, columns, product_name, folder_name, model_name, sub_folder_name, test_size, upper_bound_price, lower_bound_price, test_price_step_size):

    df = features[columns]
    df = df.dropna()
    predictions = []
    test_data = []

    max_date = pd.to_datetime(df['date']).max() - timedelta(days=2)
    min_date = max_date - timedelta(days=test_size-1)

    date_range = pd.date_range(start=min_date, end=max_date, freq='D')

    for date in date_range:

        # for combined
        date_obj = date.to_pydatetime()

        # Extract the date portion
        date_only = date_obj.date()

        # Convert the date portion back to a string if needed
        date = date_only.strftime("%Y-%m-%d")

        train = df[df['date']<date][columns]
        # print('train',df[df['date']<date]['date'])

        test = df[df['date']==date][columns]
        # print('test',df[df['date']==date]['date'])
        print('modelname',model_name)
        hour_range = list(range(24))
        if model_name == 'CatBoostRegressor':
            forcaster = CatboostForecaster()
        elif model_name == 'GradientBoostingRegressor':
            forcaster = GradientBoostingRegressorForecaster()
        elif model_name == 'RandomForestRegressor':
            print('inside randomforest')
            forcaster = RandomForestForecaster()
        elif model_name == 'LightGBM':
            forcaster = LightGBMForecaster()
        else:
            print('indi')
            forcaster = XGBoostForecaster()

    
        X_train, y_train = train.drop(['quantity','date'], axis=1), train['quantity']
        X_test, y_test = test.drop(['quantity','date'], axis=1), test['quantity']

        forcaster.fit(y_train, X_train)
    
        y_pred = forcaster.predict(X_test)
        
        # best_price, best_price_demand = find_best_price(test_prices, predictions_prices, test['cost']) 

        actual_demand_day_level = sum(y_test.values)
        predicted_demand_day_level = sum(y_pred)

        predictions.append(predicted_demand_day_level)
        test_data.append(actual_demand_day_level)
        
        print('pred',predicted_demand_day_level)
        print('actual',actual_demand_day_level)
        print('name',forcaster.model_name)
        print('complete - ', date)

    model_path = f'{folder_name}/Models/{sub_folder_name}/' \
                         f'{product_name}.pkl'

    print('preds',predictions)
    print('actual',test_data)
    
    print('inside testing')
    plot_preds(features, predictions,title=product_name, end_val=test_size, 
               folder=sub_folder_name, parent_folder_name=folder_name, model_name=model_name, actual = test_data)
    
    save_feature_importances(forcaster, folder_name, 
                        sub_folder_name, product_name)



def generate_demand_for_a_day(features, columns, model_name, date):
    
    df = features[columns]
    
    df = df.dropna()
    predictions = []
    test_data = []

    train = df[df['date']<date][columns]
    # print('train',df[df['date']<date]['date'])

    test = df[df['date']==date][columns]
    
    if model_name == 'CatBoostRegressor':
        forcaster = CatboostForecaster()
    if model_name == 'GradientBoostingRegressor':
        forcaster = GradientBoostingRegressorForecaster()
    if model_name == 'RandomForestRegressor':
        forcaster = RandomForestForecaster()
    if model_name == 'LightGBM':
        forcaster = LightGBMForecaster()
    else:
        forcaster = XGBoostForecaster()


    X_train, y_train = train.drop(['quantity','date'], axis=1), train['quantity']
    X_test, y_test = test.drop(['quantity','date'], axis=1), test['quantity']

    forcaster.fit(y_train, X_train)

    # print(X_train.columns)

    y_pred = forcaster.predict(X_test)
    
    actual_demand_day_level = sum(y_test.values)
    predicted_demand_day_level = sum(y_pred)

    predictions.append(predicted_demand_day_level)
    test_data.append(actual_demand_day_level)
    print('complete - ', date)

    return predictions, test_data

def generate_profit(features, columns, model_name, date):
    # date = datetime.strptime(date_str, '%Y-%m-%d').date()
    df = features[columns]
    df = df.dropna()
    predictions = []
    test_data = []

    train = df[df['date']<date][columns]
    print('date',date)
      
    if model_name == 'CatBoostRegressor':
        forcaster = CatboostForecaster()
    if model_name == 'GradientBoostingRegressor':
        forcaster = GradientBoostingRegressorForecaster()
    if model_name == 'RandomForestRegressor':
        forcaster = RandomForestForecaster()
    if model_name == 'LightGBM':
        forcaster = LightGBMForecaster()
    else:
        forcaster = XGBoostForecaster()

    
    X_train, y_train = train.drop(['quantity','date'], axis=1), train['quantity']
    # X_test, y_test = test.drop(['quantity','date'], axis=1), test['quantity']

    forcaster.fit(y_train, X_train)
    
    max_price = df['net_price'].max()
    min_price = df['net_price'].min()

    test_price = features[features['date']==date]['net_price'].mean()
    new_test_prices = []
    # new_test_prices.append(test_price)

    min_var = test_price - 0.3*test_price
    max_var = test_price + 0.3*test_price
    # test_cost = min_var
    for i in np.arange(min_var,
                        max_var,
                        20):
        new_test_prices.append( i)

    print('new_test_prices',new_test_prices)
    # test_vector = test.iloc[0]  
    mean_profits = []
    new_pred_volumes= []
    profits = []
    revenues = []

    # print('testvecot',test_vector)
    for i in new_test_prices:

        test_vector_copy = generate_datafrane_for_profit(date, i, columns).drop('date',axis=1)
        # print(test_vector_copy.info())
        if (test_vector_copy['net_price_day_0'] != 0).all():
            test_vector_copy['net_price_day_0'] = i
        if (test_vector_copy['net_price_day_1'] != 0).all():
            test_vector_copy['net_price_day_1'] = i
        if (test_vector_copy['net_price_day_2'] != 0).all():
            test_vector_copy['net_price_day_2'] = i
        if (test_vector_copy['net_price_day_3'] != 0).all():
            test_vector_copy['net_price_day_3'] = i
        if (test_vector_copy['net_price_day_4'] != 0).all():
            test_vector_copy['net_price_day_4'] = i
        if (test_vector_copy['net_price_day_5'] != 0).all():
            test_vector_copy['net_price_day_5'] = i
        if (test_vector_copy['net_price_day_6'] != 0).all():
            test_vector_copy['net_price_day_6'] = i
        
        # print('test',test_vector_copy)
        pred_val_hourly = forcaster.predict(test_vector_copy)

        new_pred_volumes.append(sum(pred_val_hourly))
        # profits.append((i-test_cost)*sum(pred_val_hourly))
        revenues.append(sum(pred_val_hourly)*i)


    metrics = pd.DataFrame()
    metrics['pred'] = new_pred_volumes
    metrics['prices'] = new_test_prices
    metrics.to_csv('metrics.csv')        

    plt.title('profit')
    plt.savefig(f'profitgraph/profit{date}.png')

    plt.figure()
    plt.plot( new_test_prices, new_pred_volumes, '#9d52ff', label='Prediction')
    # plt.axvline(x=test_cost/0.8, label='GMV 20', color = 'purple')
      
    plt.title('volumem')
    plt.savefig(f'profitgraph/volume{date}.png')

    plt.figure()
    plt.plot( new_test_prices, revenues, '#9d52ff', label='Prediction')
    # plt.axvline(x=test_cost/0.8, label='GMV 20', color = 'purple')
      
    plt.title('reveneu')
    plt.savefig(f'profitgraph/reveneu{date}.png')


def generate_profit_new(features, columns, model_name, date):
    # date = datetime.strptime(date_str, '%Y-%m-%d').date()
    df = features[columns]
    df = df.dropna()
    predictions = []
    test_data = []
    # print(type(date))
    date_obj = date.to_pydatetime()

# Extract the date portion
    date_only = date_obj.date()

    # Convert the date portion back to a string if needed
    date = date_only.strftime("%Y-%m-%d")
    
    
    train = df[df['date']<(date_obj - timedelta(days=1)).strftime("%Y-%m-%d")][columns]
    test_cost = features[features['date']==date]['cost'].mean()
    print('cost',(date_obj - timedelta(days=1)))
    # actual_net_price = features[features['date']==date]['net_price'].iloc[0]
    # print('actual net',actual_net_price)
    # print('train',df[df['date']<date]['date'])

    # test = df[df['date']==date][columns]
    # print('test',df[df['date']==date]['date'])
    
    if model_name == 'CatBoostRegressor':
        forcaster = CatboostForecaster()
    if model_name == 'GradientBoostingRegressor':
        forcaster = GradientBoostingRegressorForecaster()
    if model_name == 'RandomForestRegressor':
        forcaster = RandomForestForecaster()
    if model_name == 'LightGBM':
        forcaster = LightGBMForecaster()
    else:
        forcaster = XGBoostForecaster()

    
    X_train, y_train = train.drop(['quantity','date'], axis=1), train['quantity']
    # X_test, y_test = test.drop(['quantity','date'], axis=1), test['quantity']

    forcaster.fit(y_train, X_train)
    

    # print(X_train.columns)

    max_price = df['net_price'].max()
    min_price = df['net_price'].min()

    # print('inside',X_train.columns)
    # print('inside',X_test.columns)
    test_price = features['net_price'].mean()
    new_test_prices = []
    # new_test_prices.append(test_price)

    # min_var = test_price - 0.3*test_price
    # max_var = test_price + 0.3*test_price
    min_var = 1934
    max_var = 2901

    # test_cost = min_var
    for i in np.arange(min_var,
                        max_var,
                        20):
        new_test_prices.append( i)

    # test_vector = test.iloc[0]  
    mean_profits = []
    new_pred_volumes= []
    profits = []
    revenues = []

    # print('testvecot',test_vector)
    for i in new_test_prices:

        # print('date',date.date())
        # print(df['date'].iloc[-1])
        # print('df',df[df['date']==date][columns].iloc[-1])
        # test_vector_copy = generate_datafrane_for_profit(date, i, columns).drop(['date','quantity'],axis=1)
        test_vector_copy = df[df['date']==date][columns].drop(['date','quantity'],axis=1)
        test_vector_copy['net_price'] = i
        # print(test_vector_copy.info())
        if (test_vector_copy['net_price_day_0'] != 0).all():
            test_vector_copy['net_price_day_0'] = i
        if (test_vector_copy['net_price_day_1'] != 0).all():
            test_vector_copy['net_price_day_1'] = i
        if (test_vector_copy['net_price_day_2'] != 0).all():
            test_vector_copy['net_price_day_2'] = i
        if (test_vector_copy['net_price_day_3'] != 0).all():
            test_vector_copy['net_price_day_3'] = i
        if (test_vector_copy['net_price_day_4'] != 0).all():
            test_vector_copy['net_price_day_4'] = i
        if (test_vector_copy['net_price_day_5'] != 0).all():
            test_vector_copy['net_price_day_5'] = i
        if (test_vector_copy['net_price_day_6'] != 0).all():
            test_vector_copy['net_price_day_6'] = i
        
        # print('test',test_vector_copy)
        pred_val_hourly = forcaster.predict(test_vector_copy)

        new_pred_volumes.append(sum(pred_val_hourly))
        # profits.append((i-test_cost)*sum(pred_val_hourly))
        revenues.append(sum(pred_val_hourly)*i)


    metrics = pd.DataFrame()
    metrics['pred'] = new_pred_volumes
    metrics['prices'] = new_test_prices
    metrics.to_csv('metrics.csv')
    # plt.figure()
    # plt.plot( new_test_prices, profits, '#9d52ff', label='Prediction')
    # plt.axvline(x=test_cost/0.8, label='GMV 20', color = 'purple')
        

    plt.title('profit')
    plt.savefig(f'profitgraph/profit{date}.png')

    plt.figure()
    plt.plot( new_test_prices, new_pred_volumes, '#9d52ff', label='Prediction')
    # plt.axvline(x=test_cost/0.8, label='GMV 20', color = 'purple')
      
    plt.title('volumem')
    plt.savefig(f'profitgraph/volume{date}.png')

    plt.figure()
    plt.plot( new_test_prices, revenues, '#9d52ff', label='Prediction')
    # plt.axvline(x=test_cost/0.8, label='GMV 20', color = 'purple')
      
    plt.title('reveneu')
    plt.savefig(f'profitgraph/reveneu{date}.png')
