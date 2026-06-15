import os
import sys
import numpy as np
from dataclasses import dataclass
import mlflow


from src.ML_Ops_Project.exception import CustomExceptions
from src.ML_Ops_Project.logger import logging
from src.ML_Ops_Project.utils import save_object, evaluate_model

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression
# from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from dotenv import load_dotenv
from urllib.parse import urlparse

import dagshub
dagshub.init(repo_owner='ragnar-harsh', repo_name='ML_Ops_Project', mlflow=True)





load_dotenv()

mlflow_url = os.getenv('ML_FLOW_URL')


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:

    def __init__(self):
        # pass
        self.model_trainer_config = ModelTrainerConfig()

    
    def eval_metrics(self, actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2

    def initiate_model_trainer(self, train_arr, test_arr):

        try:
            # pass

            logging.info("Split Train/Test Input data.")

            x_train, y_train, x_test, y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:, :-1],
                test_arr[:,-1]
            )

            models = {
                "Linear Regression": LinearRegression(),
                # "Lasso": Lasso(),
                # "Ridge": Ridge(),
                "Gradient Boosting": GradientBoostingRegressor(),
                # "K-Neighbour Regressor": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "XGBoost": XGBRegressor(),
                "CatBoost": CatBoostRegressor(verbose=False),
                "AdaBoost": AdaBoostRegressor()
            }

            # models = {
            #     "Random Forest": RandomForestRegressor(),
            #     "Decision Tree": DecisionTreeRegressor(),
            #     "Gradient Boosting": GradientBoostingRegressor(),
            #     "Linear Regression": LinearRegression(),
            #     "XGBRegressor": XGBRegressor(),
            #     "CatBoosting Regressor": CatBoostRegressor(verbose=False),
            #     "AdaBoost Regressor": AdaBoostRegressor(),
            # }

            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBoost":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoost":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }

            model_report:dict = evaluate_model(x_train, y_train, x_test, y_test, models, params)

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]


            print("======== THIS IS THE BEST MODEL ============")
            print(best_model_name)

            model_names = list(models.keys())

            actual_model = ""

            for model in model_names:
                if best_model_name == model:
                    actual_model = actual_model + model

            best_params = params[actual_model]

            # ============================= MLFLOW ========================================

            mlflow.set_registry_uri(mlflow_url)
            tracking_url_type_source = urlparse(mlflow.get_tracking_uri()).scheme



            with mlflow.start_run():
                
                predicted_qualities = best_model.predict(x_test)

                (rmse, mae, r2) = self.eval_metrics(y_test, predicted_qualities)

                mlflow.log_params(best_params)
                # mlflow.log_metrics
                mlflow.log_metric('Root Mean Square Error', rmse)
                mlflow.log_metric('Mean Absolute Error', mae)
                mlflow.log_metric('R2 Score', r2 * 100)

                if tracking_url_type_source != 'file':
                    mlflow.sklearn.log_model(best_model, "model", registered_model_name=actual_model)
                else:
                    mlflow.sklearn.log_model(best_model, "model")

            # import mlflow
            # with mlflow.start_run():
            #   mlflow.log_param('parameter name', 'value')
            #   mlflow.log_metric('metric name', 1)

            # ============================= MLFLOW ========================================


            if best_model_score < 0.6:
                raise CustomExceptions("No best models found")
            
            logging.info(f"Best model found on both training & test dataset:>>> {best_model_name}: {best_model_score * 100} ===")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(x_test)

            r2_sc = r2_score(y_test, predicted)

            return r2_sc            


        except Exception as e:
            raise CustomExceptions(e, sys)
