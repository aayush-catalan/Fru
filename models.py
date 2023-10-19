from abc import ABC, abstractmethod
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from catboost import CatBoostRegressor

import pandas as pd
# from fbprophet import Prophet
import pandas as pd

class TimeSeriesForecaster(ABC):
    @abstractmethod
    def fit(self, X, y):
        pass
    
    @abstractmethod
    def predict(self, X):
        pass


class PipelineOptimizer(ABC):
    @abstractmethod
    def optimize(self, X, y):
        pass

# class ProphetForecaster(TimeSeriesForecaster):
#     def __init__(self):
#         self.model_name = 'Prophet'
#         self.params_grid = {
#             'seasonality_mode': 'multiplicative',
#             'daily_seasonality': True,
#             'weekly_seasonality': True,
#             'yearly_seasonality': True,
#         }
#         # Prophet doesn't use a lags_grid like the other models,
#         # but you can keep it for consistency if you want.
#         self.lags_grid = [7]  
#         self.model = Prophet(**self.params_grid)

#     def fit(self, y, regressors=None):
#         # Prepare the data in the format Prophet expects
#         data = pd.DataFrame({'ds': y.index, 'y': y.values})
#         # Add regressors if provided
#         if regressors is not None:
#             data = data.join(regressors)
#             for col in regressors.columns:
#                 self.model.add_regressor(col)
#         # Fit the model
#         self.model.fit(data)
#         return pd.DataFrame()  # Return empty DataFrame or maybe some fitting info

#     def predict(self, X):
#         # Prepare future DataFrame
#         future = self.model.make_future_dataframe(periods=len(X))
#         # Add regressors if provided
#         if X is not None:
#             future = future.join(X)
#         # Make predictions
#         forecast = self.model.predict(future)
#         return forecast['yhat']  # Return predictions

# Usage:
# forecaster = ProphetForecaster()
# forecaster.fi
import lightgbm as lgb
import pandas as pd

class LightGBMForecaster(TimeSeriesForecaster):
    def __init__(self):
        self.model_name = 'LightGBM'
        self.params_grid = {
            'learning_rate': 0.05,
            'n_estimators': 200,
            'max_depth': 5,
            'num_leaves': 31,
            'min_data_in_leaf': 20,
        }
        self.lags_grid = [7]
        self.model = lgb.LGBMRegressor(random_state=100)

    def fit(self, y, regressors):
        # The grid search forecaster comes here
        self.model.fit(regressors, y)
        return pd.DataFrame()

    def predict(self, X):
        # Perform prediction using the fitted LightGBM model
        return self.model.predict(X)

import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

class SARIMAXForecaster(TimeSeriesForecaster):
    def __init__(self):
        self.model_name = 'SARIMAX'
        # SARIMAX parameters: p, d, q for ARIMA and P, D, Q, s for seasonal component
        # self.params_grid = dict(
        #     p=[1, 2, 3],
        #     d=[0, 1],
        #     q=[1, 2, 3],
        #     P=[0, 1, 2],
        #     D=[0, 1],
        #     Q=[0, 1, 2],
        #     s=[12]  # assuming a yearly seasonal component
        # )
        self.lags_grid = [7]  # Example lag grid
        self.model = None  # Model will be initialized in fit method

    def fit(self, y, regressors):
        # The grid search forecaster can be implemented here
        # For simplicity, we'll use p=1, d=1, q=1, P=1, D=1, Q=1, s=12
        self.model = SARIMAX(y, 
                             exog=regressors, 
                             order=(1, 1, 1), 
                             seasonal_order=(1, 1, 1, 12),
                             enforce_stationarity=False,
                             enforce_invertibility=False)
        self.model_fit = self.model.fit()
        return pd.DataFrame()  # return some DataFrame if necessary

    def predict(self, X):
        # Perform prediction using the fitted SARIMAX model
        # The 'steps' parameter specifies how many steps in the future to forecast
        forecast = self.model_fit.forecast(steps=len(X), exog=X)
        return forecast


class GradientBoostingRegressorForecaster(TimeSeriesForecaster):
    def __init__(self):
        self.model_name = 'GradientBoostingRegressor'
        self.params_grid = dict(
                        learning_rate=0.05,
                        n_estimators=200,
                        max_depth=5,
                        min_samples_leaf=9,
                        min_samples_split=9,
                        )
        self.lags_grid = [7]
        self.model = GradientBoostingRegressor(loss="squared_error", **self.params_grid, random_state=100)
    
    def fit(self, y, regressors):
        # The grid search forcaster comes here
        self.model.fit(regressors,y)
        return pd.DataFrame()

    def predict(self, X):
        # Perform prediction using the fitted GradientBoosting model
        # ...
        # test_step_size = 1
        return self.model.predict(X)

class CatboostForecaster(TimeSeriesForecaster):
    def __init__(self):
        self.model_name = 'CatBoostRegressor'
        # self.params_grid = {
        #     'n_estimators': 100,
        #     'max_depth': 5,
        #     'learning_rate': 0.1
        #     }
        # self.lags_grid = [7]
        self.model = CatBoostRegressor(verbose=0, random_seed= 100) 
        #loss_function="RMSEWithUncertainty", **self.params_grid)
        
    def fit(self, y, regressors):
        # The grid search forcaster comes here
        self.model.fit(regressors,y)
        return pd.DataFrame()

    def predict(self, X):
        # Perform prediction using the fitted ARIMA model
        # ...
        # test_step_size = 1
        return list(self.model.predict(X))  #[0] ist(temp_pred_val)[0]
    

class RandomForestForecaster(TimeSeriesForecaster):
    def __init__(self):
        self.model_name = 'RandomForestRegressor'
        self.params_grid = {
        'n_estimators': 50,
        'max_depth': 5
        }
        self.lags_grid = [7]
        self.model = RandomForestRegressor(**self.params_grid, random_state= 100)
    
    def fit(self, y,regressors):
        # The grid search forcaster comes here
        self.model.fit(regressors,y)
        return pd.DataFrame()

    def predict(self, X):
        # Perform prediction using the fitted RandomForest model
        # ...
        return self.model.predict(X)
    
import xgboost as xgb
import pandas as pd

class TimeSeriesForecaster:
    # Assume this is your base class
    pass

class XGBoostForecaster(TimeSeriesForecaster):
    def __init__(self):
        self.model_name = 'XGBoost'
        self.params_grid = {
            'learning_rate': 0.05,
            'n_estimators': 200,
            'max_depth': 5,
            'min_child_weight': 1,
            'gamma': 0,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
        }
        self.lags_grid = [7]
        self.model = xgb.XGBRegressor(random_state=100, objective ='reg:squarederror')

    def fit(self, y, regressors):
        # The grid search forecaster comes here
        self.model.fit(regressors, y)
        return pd.DataFrame()

    def predict(self, X):
        # Perform prediction using the fitted XGBoost model
        return self.model.predict(X)

# Usage:
# forecaster = XGBoostForecaster()
# forecaster.fit(y_train, X_train)
# predictions = forecaster.predict(X_test)
