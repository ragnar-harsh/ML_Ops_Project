# Student Exam Performance Predictor

An end-to-end Machine Learning project that predicts student's math score based on various demographic and academic features. Built with MLOps principles, this project demonstrates a complete ML pipeline with proper logging, exception handling, and experiment tracking.

## 🎯 Project Overview

This project predicts students' math scores using features such as:
- Gender
- Race/Ethnicity
- Parental Level of Education
- Lunch Type
- Test Preparation Course
- Reading Score (out of 100)
- Writing Score (out of 100)

The project follows MLOps best practices including:
- ✅ Modular code structure
- ✅ Comprehensive logging
- ✅ Exception handling
- ✅ Multiple algorithm comparison
- ✅ Hyperparameter tuning
- ✅ Experiment tracking with MLflow & DagsHub
- ✅ Data version control with DVC
- ✅ CI/CD ready pipeline

## 📁 Project Structure

```
student-exam-performance-predictor/
│
├── .github/
│   └── workflows/
│       └── ci-cd.yml                 # CI/CD pipeline configuration
│
├── .dvc/                             # DVC configuration
│
├── artifacts/                        # All artifacts generated
│   ├── data.csv                      # Cleaned dataset
│   ├── train.csv                     # Training data
│   ├── test.csv                      # Test data
│   ├── model.pkl                     # Best model pickle file
│   ├── preprocessor.pkl              # Preprocessor object
│   └── model_monitoring/             # Model performance metrics
│
├── config/
│   └── config.yaml                   # Configuration file
│
├── data/
│   ├── raw/                          # Raw data (DVC tracked)
│   └── processed/                    # Processed data
│
├── logs/
│   ├── app.log                       # Application logs
│   ├── data_ingestion.log
│   ├── data_transformation.log
│   ├── model_trainer.log
│   └── prediction.log
│
├── mlruns/                           # MLflow runs
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_model_evaluation.ipynb
│
├── src/
│   ├── __init__.py
│   │
│   ├── components/
│   │   ├── __init__.py
│   │   ├── data_ingestion.py         # Data ingestion component
│   │   ├── data_transformation.py    # Data preprocessing
│   │   ├── model_trainer.py          # Model training & tuning
│   │   └── model_monitoring.py       # Model performance tracking
│   │
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── training_pipeline.py      # End-to-end training pipeline
│   │   └── prediction_pipeline.py    # Prediction pipeline
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── logger.py                 # Logging configuration
│   │   ├── exception.py              # Custom exception handling
│   │   └── utils.py                  # Utility functions
│   │
│   └── config/
│       └── configuration.py          # Configuration management
│
├── templates/
│   ├── index.html                    # Landing page
│   └── predict_data.html             # Prediction form page
│
├── static/
│   └── css/
│       └── style.css                 # Custom CSS styles
│
├── tests/
│   ├── __init__.py
│   ├── test_data_ingestion.py
│   ├── test_data_transformation.py
│   └── test_model_trainer.py
│
├── .dvcignore
├── .gitignore
├── .env.example                      # Environment variables template
├── requirements.txt                  # Project dependencies
├── setup.py                          # Package setup
├── app.py                            # Flask application entry point
├── dvc.yaml                          # DVC pipeline configuration
├── dvc.lock                          # DVC lock file
├── mlflow_tracking.py                # MLflow experiment tracking
├── README.md                         # Project documentation
└── LICENSE
```

## 🛠️ Technologies & Tools

### Core Technologies
- **Python 3.8+**: Primary programming language
- **Flask**: Web framework for API and UI
- **Scikit-learn**: Machine learning library
- **Pandas & NumPy**: Data manipulation
- **Matplotlib & Seaborn**: Visualization

### MLOps Tools
- **MLflow**: Experiment tracking and model registry
- **DagsHub**: Centralized ML platform (MLflow + DVC)
- **DVC (Data Version Control)**: Data versioning and pipeline
- **Git**: Version control

### Development Tools
- **Jupyter Lab**: Interactive development
- **VS Code**: IDE (recommended)
- **GitHub Actions**: CI/CD

## 📸 Screenshots

### Web Application Interface

#### Landing Page
![alt text](<artifacts/Screenshot 2026-06-22 185925.png>)
The modern, Material Design-inspired landing page with navigation to the prediction form

#### Prediction Form
![alt text](<artifacts/Screenshot 2026-06-22 190020.png>)
User-friendly form interface for inputting student data

## 🚀 Getting Started

### Prerequisites

1. **Python 3.8 or higher**
   ```bash
   python --version
   ```

2. **Git**
   ```bash
   git --version
   ```

3. **DVC**
   ```bash
   pip install dvc
   ```

4. **DagsHub Account** (for experiment tracking)
   - Sign up at [dagshub.com](https://dagshub.com)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/student-exam-performance-predictor.git
   cd student-exam-performance-predictor
   ```

2. **Create and activate virtual environment**
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # Mac/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install the package in development mode**
   ```bash
   pip install -e .
   ```

5. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your DagsHub credentials
   ```

6. **Set up DVC remote storage**
   ```bash
   dvc remote add -d origin https://dagshub.com/yourusername/student-exam-performance-predictor.dvc
   dvc remote modify origin --local auth basic
   dvc remote modify origin --local user your_username
   dvc remote modify origin --local password your_token
   ```

## 🔄 ML Pipeline

The project follows a structured ML pipeline with the following stages:

### 1. Data Ingestion
- Loads raw data from CSV files
- Splits data into train/test sets (80/20)
- Saves artifacts for downstream use
- Logs data statistics and validation

**Component**: `src/components/data_ingestion.py`

### 2. Data Transformation
- Handles missing values
- Encodes categorical variables
- Scales numerical features
- Creates preprocessor object
- Saves processed data

**Component**: `src/components/data_transformation.py`

### 3. Model Training
- Trains multiple models:
  - Linear Regression
  - Decision Tree Regressor
  - Random Forest Regressor
  - Gradient Boosting Regressor
  - XGBoost Regressor
  - CatBoost Regressor
  - AdaBoost Regressor
  - K-Neighbors Regressor
- Performs hyperparameter tuning using GridSearchCV
- Evaluates models using multiple metrics (R², MAE, RMSE)
- Logs all experiments with MLflow
- Saves the best model

**Component**: `src/components/model_trainer.py`

### 4. Model Monitoring
- Tracks model performance over time
- Logs metrics to MLflow
- Monitors data drift
- Generates performance reports

**Component**: `src/components/model_monitoring.py`

### 5. Prediction Pipeline
- Loads the trained model and preprocessor
- Accepts user inputs via Flask API
- Transforms input data
- Returns predictions with confidence intervals

**Component**: `src/pipeline/prediction_pipeline.py`

## 🧪 Models Evaluated

| Model | Parameters Tuned | Best R² Score |
|-------|-----------------|---------------|
| Linear Regression | - | ~0.88 |
| Decision Tree | max_depth, min_samples_split | ~0.85 |
| Random Forest | n_estimators, max_depth, min_samples_split | ~0.81 |
| Gradient Boosting | n_estimators, learning_rate, max_depth | ~0.82 |
| XGBoost | n_estimators, learning_rate, max_depth, subsample | ~0.83 |
| CatBoost | iterations, depth, learning_rate | ~0.84 |
| AdaBoost | n_estimators, learning_rate | ~0.79 |
| K-Neighbors | n_neighbors, weights, p | ~0.81 |

*Note: Actual scores may vary based on data split and tuning configuration.*

## 📊 Experiment Tracking with MLflow & DagsHub

This project uses MLflow integrated with DagsHub for experiment tracking:

1. **Start MLflow UI locally:**
   ```bash
   mlflow ui --backend-store-uri mlruns/
   ```

2. **View experiments on DagsHub:**
   - Navigate to your DagsHub repository
   - Click on "Experiments" tab
   - View all runs, metrics, and parameters

3. **Log experiments programmatically:**
   ```python
   import mlflow
   
   with mlflow.start_run():
       mlflow.log_param("model_type", "RandomForest")
       mlflow.log_metric("r2_score", 0.92)
       mlflow.sklearn.log_model(model, "model")
   ```

## 🚦 Running the Application

### Training Pipeline
```bash
# Run full training pipeline
python src/pipeline/training_pipeline.py

# Or run components individually
python src/components/data_ingestion.py
python src/components/data_transformation.py
python src/components/model_trainer.py
```

### Web Application
```bash
# Start Flask application
python app.py

# Access the application
# Open browser and go to: http://localhost:5000
```

### Prediction API
```python
# Example API call
import requests

data = {
    "gender": "male",
    "ethnicity": "group A",
    "parental_level_education": "bachelor's degree",
    "lunch": "standard",
    "test_preparation_course": "completed",
    "reading_score": 85,
    "writing_score": 78
}

response = requests.post("http://localhost:5000/predict", json=data)
print(response.json())
```

## 📈 Monitoring & Logging

### Logging Configuration
The project implements comprehensive logging using Python's built-in logging module:

```python
from src.utils.logger import logging

logging.info("Model training started")
logging.error("Error occurred during prediction")
```

### Log Files
- `logs/app.log`: General application logs
- `logs/data_ingestion.log`: Data ingestion specific logs
- `logs/data_transformation.log`: Transformation logs
- `logs/model_trainer.log`: Training logs
- `logs/prediction.log`: Prediction pipeline logs

### Exception Handling
Custom exception handling with detailed error messages:

```python
from src.utils.exception import CustomException

try:
    # Some operation
    pass
except Exception as e:
    raise CustomException(e, sys)
```

## 🔧 Configuration

### config/config.yaml
```yaml
data:
  raw_data_path: data/raw/student_performance.csv
  train_data_path: artifacts/train.csv
  test_data_path: artifacts/test.csv
  
model:
  random_state: 42
  test_size: 0.2
  
logging:
  level: INFO
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  
mlflow:
  tracking_uri: "https://dagshub.com/yourusername/student-exam-performance-predictor.mlflow"
  experiment_name: "Student_Exam_Performance"
```

## 🧪 Testing

Run tests using pytest:
```bash
pytest tests/
```

Test coverage:
```bash
pytest --cov=src tests/
```

## 📦 DVC Pipeline

The DVC pipeline is defined in `dvc.yaml`:

```yaml
stages:
  data_ingestion:
    cmd: python src/components/data_ingestion.py
    deps:
      - data/raw/student_performance.csv
      - src/components/data_ingestion.py
    outs:
      - artifacts/train.csv
      - artifacts/test.csv
  
  data_transformation:
    cmd: python src/components/data_transformation.py
    deps:
      - artifacts/train.csv
      - artifacts/test.csv
      - src/components/data_transformation.py
    outs:
      - artifacts/preprocessor.pkl
  
  model_trainer:
    cmd: python src/components/model_trainer.py
    deps:
      - artifacts/train.csv
      - artifacts/test.csv
      - artifacts/preprocessor.pkl
      - src/components/model_trainer.py
    outs:
      - artifacts/model.pkl
```

Run DVC pipeline:
```bash
dvc repro
```

## 🔄 CI/CD Pipeline

The project includes GitHub Actions workflow (`.github/workflows/ci-cd.yml`):

```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  build-and-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.8'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      - name: Run tests
        run: |
          pytest tests/
      - name: Run DVC pipeline
        run: |
          dvc pull
          dvc repro
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- Harsh Singh - [GitHub Profile](https://github.com/ragnar-harsh)

## 🙏 Acknowledgments

- Data source: [Student Performance Dataset](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams)
- Inspired by MLOps best practices from various industry sources

## 📚 Additional Resources

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [DVC Documentation](https://dvc.org/doc)
- [DagsHub Documentation](https://dagshub.com/docs)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)

## 📧 Contact

For questions or support, please open an issue or contact: your.email@example.com

---

**Star ⭐ this repository if you find it useful!**