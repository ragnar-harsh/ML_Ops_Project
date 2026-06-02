import sys
import os 
from dataclasses import dataclass
from src.ML_Ops_Project.exception import CustomExceptions
from src.ML_Ops_Project.logger import logging
import pandas as pd



@dataclass
class DataIngestionConfig:
    train_data_path = os.path.join('artifacts', 'train.csv')
    test_data_path = os.path.join('artifacts', 'test.csv')
    raw_data_path = os.path.join('artifacts', 'raw.csv')


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):

        try:
            # pass
            logging.info("Reading From MySQL Database")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            




        except Exception as ex:
            logging.error(ex)
            raise CustomExceptions(ex, sys)

