import os
import sys
import numpy as np
import pandas as pd
# import pickle

from dataclasses import dataclass
from src.ML_Ops_Project.exception import CustomExceptions
from src.ML_Ops_Project.logger import logging
from src.ML_Ops_Project.utils import save_object


from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline




@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:

    def __init__(self):
        # pass
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This Function is responsible for Data Transformation
        '''

        # num_features = X.select_dtypes(exclude='object').columns
        # cat_features = X.select_dtypes(include='object').columns

        num_features = ['writing_score', 'reading_score']
        cat_features = ['gender', 'race', 'education', 'lunch', 'course']



        try:
            # pass

            num_pipeline = Pipeline(steps=[
                ("Imputer", SimpleImputer(strategy='median')),
                ("Scaler", StandardScaler()),
            ])

            cat_pipeline = Pipeline(steps=[
                ("Imputer", SimpleImputer(strategy='most_frequent')),
                ("Encoder", OneHotEncoder()),
                ("Scaler", StandardScaler(with_mean=False))
            ])

            logging.info(f"Numerical Features: {num_features}")
            logging.info(f"Categorical Features: {cat_features}")
            logging.info("--- Pipeline Instantiated ---")

            preprocessor = ColumnTransformer([
                ("Num_Pipeline", num_pipeline, num_features),
                ("Cat_Pipeline", cat_pipeline, cat_features),
            ])

            return preprocessor


        except Exception as ex:
            raise CustomExceptions(ex, sys)
        
    def initiate_data_transformation(self, train_path, test_path):

        try:
            # pass
            logging.info("--- Reading Train/Test Files ---")

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("--- Initiating Data Transformer ---")
            preprocess_object = self.get_data_transformer_object()

            target_column_name = 'math_score'
            numeric_columns = ['reading_score', 'writing_score']


            logging.info("--- Dividing train dataset to dependent & Independent Features ---")

            input_train_df = train_df.drop(columns=[target_column_name])
            target_train_df = train_df[target_column_name]


            logging.info("--- Dividing train dataset to dependent & Independent Features ---")

            input_test_df = test_df.drop(columns=[target_column_name])
            target_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing on Training/Test Datasets")

            input_train_arr = preprocess_object.fit_transform(input_train_df)
            input_test_arr = preprocess_object.transform(input_test_df)

            train_arr = np.c_[
                input_train_arr, np.array(target_train_df)
            ]
            test_arr = np.c_[
                input_test_arr, np.array(target_test_df)
            ]

            logging.info("Saved Preprocessing Object")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocess_object
            )


            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )








            


        except Exception as e:
            raise CustomExceptions(e, sys)