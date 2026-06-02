from src.ML_Ops_Project.logger import logging
import sys 
from src.ML_Ops_Project.exception import CustomExceptions




if __name__ == "__main__":
    logging.info("The execution of logging file is successfull.")


    try:
        a = 5/0
    except Exception as e:
        logging.error("Custom Exception Occured")
        logging.warning("Custom Exception Occured: WARN")
        logging.info("Custom Exception Occured: INFO")
        raise CustomExceptions(e, sys)
