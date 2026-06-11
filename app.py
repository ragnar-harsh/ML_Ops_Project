from src.ML_Ops_Project.logger import logging
import sys 
from src.ML_Ops_Project.exception import CustomExceptions
from src.ML_Ops_Project.components.data_ingestion import DataIngestion
# from src.ML_Ops_Project.components.data_ingestion import DataIngestionConfig
from src.ML_Ops_Project.components.data_transformation import DataTransformation
from src.ML_Ops_Project.components.model_trainer import ModelTrainer




if __name__ == "__main__":
    logging.info("The execution of logging file is successfull.")


    try:
        # a = 5/0
        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

        data_transformation = DataTransformation()
        train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data_path, test_data_path)

        model_trainer = ModelTrainer()
        res = model_trainer.initiate_model_trainer(train_arr, test_arr)

        logging.info(f"====== Execution successfull ========== >> {res * 100}")

        # print(res_obj)



    except Exception as e:
        # logging.error("Custom Exception Occured")
        # logging.warning("Custom Exception Occured: WARN")
        # logging.info("Custom Exception Occured: INFO")
        raise CustomExceptions(e, sys)
