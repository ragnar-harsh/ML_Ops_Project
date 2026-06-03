from src.ML_Ops_Project.logger import logging
import sys 
from src.ML_Ops_Project.exception import CustomExceptions
from src.ML_Ops_Project.components.data_ingestion import DataIngestion
from src.ML_Ops_Project.components.data_ingestion import DataIngestionConfig




if __name__ == "__main__":
    logging.info("The execution of logging file is successfull.")


    try:
        # a = 5/0
        data_ingestion = DataIngestion()
        data_ingestion.initiate_data_ingestion()
    except Exception as e:
        # logging.error("Custom Exception Occured")
        # logging.warning("Custom Exception Occured: WARN")
        # logging.info("Custom Exception Occured: INFO")
        raise CustomExceptions(e, sys)
