import sys
import os 
from dataclasses import dataclass
from src.ML_Ops_Project.exception import CustomExceptions
from src.ML_Ops_Project.logger import logging
import pandas as pd
from dotenv import load_dotenv

host = os.getenv('DB_HOST')
port = os.getenv('DB_PORT')
user = os.getenv('DB_USERNAME')
passwd = os.getenv('DB_PASSWORD')
db = os.getenv('DB_DATABASE')


def read_sql_data():

    logging.info("Connecting Database....")

    try:
        pass
    except Exception as ex:
        # logging.error("Error Occured while connecting DB")
        raise CustomExceptions(ex, sys)








