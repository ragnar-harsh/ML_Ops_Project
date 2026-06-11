import os
import sys
from dataclasses import dataclass

from src.ML_Ops_Project.exception import CustomExceptions
from src.ML_Ops_Project.logger import logging
from src.ML_Ops_Project.utils import save_object, evaluate_model

from sklearn.metrics import r2_score

from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor



@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:

    def __init__(self):
        # pass
        self.model_trainer_config = ModelTrainerConfig()

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
