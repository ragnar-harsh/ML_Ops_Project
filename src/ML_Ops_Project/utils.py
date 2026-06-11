import sys
import os 
from dataclasses import dataclass
from src.ML_Ops_Project.exception import CustomExceptions
from src.ML_Ops_Project.logger import logging
import pandas as pd
from dotenv import load_dotenv
import pymysql
import pickle
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

load_dotenv()

host = os.getenv('DB_HOST')
port = os.getenv('DB_PORT')
user = os.getenv('DB_USERNAME')
passwd = os.getenv('DB_PASSWORD')
db = os.getenv('DB_DATABASE')


# def read_sql_data(query):

#     logging.info("...Connecting Database....")

#     try:
#         # pass
#         myDB = pymysql.connect(
#             host=host,
#             user=user,
#             password=passwd,
#             db=db
#         )

#         logging.info("===== Connection Established =====", myDB)

#         if not query:
#             logging.warning("No Query Provided... Executing Default query....")
#             df = pd.read_sql_query("Select * from Students", myDB)
#         else:
#             df = pd.read_sql_query(query, myDB)
        
#         print(df.head())


#     except Exception as ex:
#         # logging.error("Error Occured while connecting DB")
#         raise CustomExceptions(ex, sys)
    


def read_sql_data(query = None):

    logging.info("...Connecting Database....")
    myDB = None
    try:
        myDB = pymysql.connect(
            host=host, 
            user=user, 
            password=passwd, 
            db=db
        )
        # Fixed: Safe string formatting for logging
        logging.info(f"===== Connection Established ===== {myDB}") 
        # logging.info("===== Connection Established =====", myDB)

        df = None
        
        if query is None:
            logging.warning("No Query Provided... Executing Default query....")
            df = pd.read_sql_query("SELECT * FROM Students", myDB)
        else:
            df = pd.read_sql_query(query, myDB)
            
        print(df.head())
        return df

    except Exception as ex:
        logging.error(f"Error occurred while connecting DB: {ex}")
        raise CustomExceptions(ex, sys)
        
    finally:
        # Best Practice: Always close the connection
        if myDB and myDB.open:
            myDB.close()
            logging.info("===== Connection Closed =====")
            print("===== Connection Closed =====")


def save_object(file_path, obj):
    try:
        # pass
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_object:
            pickle.dump(obj, file_object)

    except Exception as e:
        raise CustomExceptions(e, sys)



def evaluate_model_accuracy(true_value, predicted):
    mae = mean_absolute_error(true_value, predicted)
    mse = mean_squared_error(true_value, predicted)
    rmse = np.sqrt(mse)
    # rmse = np.sqrt(mean_squared_error(true_value, predicted))
    r2_square = r2_score(true_value, predicted)
    return mae, rmse, r2_square

def evaluate_model(x_train, y_train, x_test, y_test, models, params):

    try:

        # pass

        logging.info("Evaluating All the Models with Best Params")
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]

            param = params[list(models.keys())[i]]

            gs = GridSearchCV(model, param, cv=3)

            gs.fit(x_train, y_train)

            logging.info(f"{model} ->> Best Params: >> {gs.best_params_}")
            model.set_params(**gs.best_params_)

            model.fit(x_train, y_train)

            # Make Predictions
            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)


            # Evaluate train/test Dataset
            model_train_mae, model_train_rmse, model_train_r2 = evaluate_model_accuracy(y_train, y_train_pred)
            model_test_mae, model_test_rmse, model_test_r2 = evaluate_model_accuracy(y_test, y_test_pred)

            report[list(models.keys())[i]] = model_test_r2
        
        return report




    except Exception as e:
        raise CustomExceptions(e, sys)







