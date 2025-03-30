# app.py
from flask import Flask, request, jsonify
import pandas as pd
import joblib

# Load the model
model = joblib.load('fraud_detection_model.pkl')

# Create Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from the request
    data = request.get_json()
    
    # Validate input data
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    required_fields = ['TransactionAmt', 'ProductCD', 'card4']
    if not all(field in data for field in required_fields):
        return jsonify({"error": "Missing required fields"}), 400
    
    # Convert input data into a DataFrame
    input_data = pd.DataFrame([data])
    
    # Make a prediction
    prediction = model.predict(input_data)
    
    # Return the prediction as a JSON response
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True)