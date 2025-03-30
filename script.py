import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
import joblib
from flask import Flask, request, jsonify

# Check if the model file already exists
model_file = 'fraud_detection_model.pkl'

if not os.path.exists(model_file):
    print("Model file not found. Training the model...")

    # Load transaction data with optimized data types
    dtypes = {
        'TransactionID': 'int32',
        'isFraud': 'int8',
        'TransactionAmt': 'float32',
        'ProductCD': 'category',
        'card4': 'category',
        # Add more columns with appropriate dtypes
    }
    transaction_df = pd.read_csv('train_transaction.csv', dtype=dtypes)

    # Load identity data with optimized data types
    identity_df = pd.read_csv('train_identity.csv', dtype=dtypes)

    # Display the first 5 rows of each dataset
    print(transaction_df.head())
    print(identity_df.head())

    # Check missing values in transaction data
    print(transaction_df.isnull().sum())

    # Check missing values in identity data
    print(identity_df.isnull().sum())

    # Visualize class distribution
    sns.countplot(x='isFraud', data=transaction_df)
    plt.title('Class Distribution (0 = Legitimate, 1 = Fraud)')
    plt.show()

    # Plot transaction amounts by class
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='isFraud', y='TransactionAmt', data=transaction_df)
    plt.title('Transaction Amounts by Class')
    plt.show()

    # Merge datasets
    merged_df = pd.merge(transaction_df, identity_df, on='TransactionID', how='left')

    # Display the first 5 rows of the merged dataset
    print(merged_df.head())

    # Drop columns with more than 50% missing values
    merged_df = merged_df.dropna(axis=1, thresh=len(merged_df) * 0.5)

    # Fill remaining missing values with the mean or mode
    for col in merged_df.columns:
        if pd.api.types.is_numeric_dtype(merged_df[col]):
            # Use .loc to avoid SettingWithCopyWarning
            merged_df.loc[:, col] = merged_df[col].fillna(merged_df[col].mean())
        else:
            # Use .loc to avoid SettingWithCopyWarning
            merged_df.loc[:, col] = merged_df[col].fillna(merged_df[col].mode()[0])

    # One-hot encode categorical variables
    merged_df = pd.get_dummies(merged_df, drop_first=True)

    # Scale numerical features
    scaler = StandardScaler()
    numerical_cols = merged_df.select_dtypes(include=['float32', 'int32']).columns
    merged_df[numerical_cols] = scaler.fit_transform(merged_df[numerical_cols])

    # Define features (X) and target (y)
    X = merged_df.drop('isFraud', axis=1)
    y = merged_df['isFraud']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Apply SMOTE to balance the dataset (only on the training data)
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # Train a Random Forest model
    model = RandomForestClassifier(random_state=42)

    # Define hyperparameters for grid search
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }

    # Perform Grid Search
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='f1')
    grid_search.fit(X_train_res, y_train_res)

    # Best parameters
    print("Best Parameters:", grid_search.best_params_)

    # Use the best model from grid search
    best_model = grid_search.best_estimator_

    # Make predictions using the best model
    y_pred = best_model.predict(X_test)

    # Evaluate the model
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Save the best model
    joblib.dump(best_model, model_file)
    print(f"Model saved to {model_file}")

else:
    print(f"Model file {model_file} already exists. Skipping training.")

# Load the model
model = joblib.load(model_file)

# Create Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from the request
    data = request.get_json()
    
    # Convert input data into a DataFrame (ensure it matches the training data format)
    input_data = pd.DataFrame([data])
    
    # Make a prediction
    prediction = model.predict(input_data)
    
    # Return the prediction as a JSON response
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True)
    {
    "TransactionAmt": 100.0,
    "ProductCD": "W",
    "card4": "visa",
    "other_features": "values"
}