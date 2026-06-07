import sys
import os 
from dataclasses import dataclass
from src.ML_Ops_Project.exception import CustomExceptions
from src.ML_Ops_Project.logger import logging
import pandas as pd
from src.ML_Ops_Project.utils import read_sql_data
from sklearn.model_selection import train_test_split


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
            logging.info("============ Reading From MySQL Database =============")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # df = read_sql_data()
            df = pd.read_csv(os.path.join('Notebook/data', 'raw.csv'))

            df.to_csv(self.ingestion_config.raw_data_path, index = False, header = True)

            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, index = False, header = True)
            test_set.to_csv(self.ingestion_config.test_data_path, index = False, header = True)


            logging.info("============ Data Ingestion Completed ===========")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )



        except Exception as ex:
            logging.error(ex)
            raise CustomExceptions(ex, sys)

