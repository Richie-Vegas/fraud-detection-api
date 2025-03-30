Fraud Detection System
Overview
This project is a machine learning-based fraud detection system that identifies fraudulent transactions using a Random Forest Classifier. The model is trained on transactional and identity data, applies SMOTE for handling class imbalance, and is deployed as a Flask API.
Features
* Data preprocessing with missing value handling and feature engineering
* One-hot encoding for categorical variables
* Feature scaling using StandardScaler
* Model training using Random Forest with Grid Search optimization
* Handling class imbalance using SMOTE
* Flask API for real-time fraud detection
* Model persistence using Joblib
Requirements
Ensure you have the following dependencies installed:
pip install pandas scikit-learn imbalanced-learn flask matplotlib seaborn joblib
Dataset
The project utilizes two datasets:
1. train_transaction.csv – Contains transaction details.
2. train_identity.csv – Contains identity details linked to transactions.
Installation
Clone the repository and navigate to the project directory:
git clone <repository_url>
cd fraud_detection_project
Ensure Git is installed before proceeding.
Model Training
If the model file (fraud_detection_model.pkl) is missing, the script will automatically train a new model.
python fraud_detection.py
Running the Flask API
To start the API, run:
python fraud_detection.py
The API will be available at http://127.0.0.1:5000/predict.
API Usage
Send a POST request to /predict with transaction details in JSON format:
{
  "TransactionAmt": 100.0,
  "ProductCD": "W",
  "card4": "visa",
  "other_features": "values"
}
The API returns a JSON response with a fraud prediction:
{
  "prediction": 1
}
where 1 indicates fraud and 0 indicates a legitimate transaction.
There are files that I cant upload because they are too big 

