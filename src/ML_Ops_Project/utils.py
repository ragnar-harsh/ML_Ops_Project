import sys
import os 
from dataclasses import dataclass
from src.ML_Ops_Project.exception import CustomExceptions
from src.ML_Ops_Project.logger import logging
import pandas as pd
from dotenv import load_dotenv
import pymysql

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










