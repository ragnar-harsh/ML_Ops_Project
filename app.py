# import sys 

# from src.ML_Ops_Project.logger import logging
# from src.ML_Ops_Project.exception import CustomExceptions
from src.ML_Ops_Project.pipelines.prediction_pipeline import CustomData, PredictionPipeline


# from src.ML_Ops_Project.components.data_ingestion import DataIngestion
# from src.ML_Ops_Project.components.data_transformation import DataTransformation
# from src.ML_Ops_Project.components.model_trainer import ModelTrainer

from flask import Flask, request, render_template
import numpy as np



application = Flask(__name__)

app = application


# Route for a home page 

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict-data', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        # pass
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_education=request.form.get('parental_level_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('reading_score')),
            writing_score=float(request.form.get('writing_score'))
        )

        data_frame = data.get_data_as_data_frame()

        print(data_frame)

        pred_pipeline = PredictionPipeline()

        results = pred_pipeline.predict(data_frame)

        return render_template('home.html', result = int(results[0]))




if __name__ == "__main__":
    app.run(debug=True)
    # app.run(host='0.0.0.0', debug=True)



'''

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


'''
