import os
from utils import (get_filterd_dataset,
                   aggregate_by_hour,
                   create_features,
                #    aggregate_by_time_bucket,
                    #  generate_demand_graph,
                    #  generate_demand_graph_testing,
                     generate_demand_graph_daily_testing,
                     generate_demand_for_a_day,
                     generate_profit,
                     generate_profit_new,
                     generate_datafrane_for_profit,
                     aggregate_by_hour_High_Selling,
                    add_to_Historic_Dataframe

                #      profit_graph_without_uncertainity
                     )

from models import ( 
    CatboostForecaster,
    TimeSeriesForecaster, 
    RandomForestForecaster, 
    GradientBoostingRegressorForecaster
)

import yaml
import pandas as pd
import pytz
# data1 = pd.read_csv('data/Frubana_2023_10_16.csv')
# data2 = pd.read_csv('data/Frubana_2023_10_17.csv')

# colombia_tz = pytz.timezone('America/Bogota')
#     # Convert the 'order_submited_datetime' column to Colombia timezone

# data2['order_submited_datetime'] = pd.to_datetime(data2['order_submited_datetime'])
# data2['order_submited_datetime'] = data2['order_submited_datetime'].dt.tz_localize(pytz.utc).dt.tz_convert(colombia_tz)
# data2['order_submited_date'] = data2['order_submited_datetime'].dt.date.astype(str)
# print('date2',len(data2))
# data2 = data2[data2['order_submited_date']>'2023-10-15']
# print('date2',len(data2))
# data2['order_submited_datetime'] = pd.to_datetime(data2['order_submited_datetime'])
# data1['order_submited_datetime'] = pd.to_datetime(data1['order_submited_datetime'])


# data = pd.concat([data1,data2])
# data = data.drop_duplicates()
# data['order_submited_datetime'] = pd.to_datetime(data['order_submited_datetime'])

# # addDataframe()
# df = add_to_Historic_Dataframe(data,'2023-10-17')

with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

has_competitor_price = config['has_competitor_price']
products = config['products']
folder_name = config['folder_name']
columns = config['features']
time_bucket = config['time_bucket']
upper_bound_price = config['upperBound']
lower_bound_price = config['lowerBound']
test_price_step_size = config['test_price_step_size']
test_dataset_size = config['test_dataset_size']

for product_name in products:
    # filtered = get_filterd_dataset(product_name)
    # daily_aggregated_df = aggregate_by_hour(filtered, time_bucket=time_bucket, has_competitor_price=has_competitor_price)
    # x = daily_aggregated_df.sort_values('date')
    
    
    # features = create_features(daily_aggregated_df, has_competitor_price)
    # print('dily',features.sort_values('date').iloc[-1])
    
    # print(features.iloc[-1])

    # print(features.info)
    # models = ['RandomForestRegressor','XGBoost','GradientBoostingRegressor','CatBoostRegressor']
    models = ['LightGBM']

    features = pd.read_csv('combined_data/CBP1kg_500(mode).csv')
    features['quantity_lag_month'] = features['quantity'].shift(168*4)
    features['quantity_lag_biweek'] = features['quantity'].shift(168*2)
    features['net_price_lag_month'] = features['net_price'].shift(168*4)
    features['net_price_lag_biweek'] = features['net_price'].shift(168*2)
    features['quantity_lag_10'] = features['quantity'].shift(240)
    features['net_price_lag_10'] = features['net_price'].shift(240)
    features['FP_daily_quantity_lag_month'] = features['FP_daily_quantity'].shift(30)
    features['FP_daily_quantity_lag_10'] = features['FP_daily_quantity'].shift(10)
    features['FP_daily_quantity_lag_biweek'] = features['FP_daily_quantity'].shift(14)
    features['FP_daily_quantity_lag_15'] = features['FP_daily_quantity'].shift(15)
    features['quantity_hourly_lag48'] = features['quantity'].shift(48)
    features['quantity_hourly_lag49'] = features['quantity'].shift(49)
    features['quantity_daily_lag3'] = features['quantity_daily_lag2'].shift(1)
    features['quantity_daily_lag4'] = features['quantity_daily_lag2'].shift(2)
    features['quantity_hourly_lag72']= features['quantity_hourly_lag48'].shift(24)
    features['quantity_hourly_lag96']= features['quantity_hourly_lag48'].shift(48)
    features['quantity_hourly_lag108']= features['quantity_hourly_lag48'].shift(72)
    features['quantity_hourly_lag132']= features['quantity_hourly_lag48'].shift(96)
    features['datetime'] = pd.to_datetime(features['date'])

    # mixed_type_columns = features.select_dtypes(include=['object']).columns[features.select_dtypes(include=['object']).apply(lambda x: x.apply(type).nunique() > 1)]

    # print("Columns with mixed data types:")
    # print(mixed_type_columns)

    # print(features.iloc[-2])




    for model_name in models:

        sub_folder_name = f'{model_name}_step_binned_feats'
        os.makedirs(f'{folder_name}/Graphs/{sub_folder_name}/{product_name}', exist_ok=True)
        os.makedirs(f'{folder_name}/Feature_Importances/{sub_folder_name}/{product_name}', exist_ok=True)  # noqa: E501
        os.makedirs(f'{folder_name}/Metrics/{sub_folder_name}/{product_name}', exist_ok=True)
        os.makedirs(f'{folder_name}/Predictions/{sub_folder_name}/{product_name}', exist_ok=True)
        os.makedirs(f'{folder_name}/Models/{sub_folder_name}/{product_name}', exist_ok=True)
        os.makedirs(f'{folder_name}/Daily_Analysis_Graphs/CatBoostRegressor/{product_name}', exist_ok=True)
        
        # generate_demand_graph_daily_testing(features,columns, product_name,folder_name,model_name,sub_folder_name, test_dataset_size, upper_bound_price, lower_bound_price, test_price_step_size)
        # preds, actual = generate_demand_for_a_day(features,columns, model_name,'2023-09-01')
        date_range = pd.date_range(start='2023-10-18', end='2023-10-19', freq='D')
        for day in date_range:
             
            generate_profit_new(features,columns, model_name,day,lower_bound_price, upper_bound_price)


