import sys
import pandas as pd

from src.ML_Ops_Project.exception import CustomExceptions
from src.ML_Ops_Project.logger import logging
from src.ML_Ops_Project.utils import load_object



class CustomData:
    def __init__(
            self,
            gender: str,
            race_ethnicity: str,
            parental_level_education: str,
            lunch: str,
            test_preparation_course: str,
            reading_score: int,
            writing_score: int
    ):
        self.gender = gender
        self.race = race_ethnicity
        self.education = parental_level_education
        self.lunch = lunch
        self.course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    
    def get_data_as_data_frame(self):
        try:
            # pass
            custom_data_input_dict = {
                "gender": [self.gender],
                "race": [self.race],
                "education": [self.education],
                "lunch": [self.lunch],
                "course": [self.course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }

            return pd.DataFrame(custom_data_input_dict)


        except Exception as e:
            raise CustomExceptions(e, sys)


class PredictionPipeline:

    def __init__(self):
        pass

    def predict(self, features):
        try:
        # pass
            logging.info("Loading pickle Objects...")
            model_path = "artifacts/model.pkl"
            preprocessor_path = "artifacts/preprocessor.pkl"

            model = load_object(model_path)
            preprocessor = load_object(preprocessor_path)

            logging.info("Scaling Input Features...")
            scaled_data = preprocessor.transform(features)

            logging.info("Prediction Result ---- ")
            preds = model.predict(scaled_data)
            logging.info(f"Prediction Finished ---- {preds}")

            return preds


        except Exception as e:
            raise CustomExceptions(e, sys)

        





