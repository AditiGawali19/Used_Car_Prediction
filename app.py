from flask import Flask, render_template, request, jsonify
import pandas as pd
from utils import predict_selling_price
import config
import logging

app = Flask(__name__)

# Load the dataset to get dropdown options
dataset = pd.read_csv('car_data.csv')

# Homepage API
@app.route('/')
def homepage():
    return render_template('index.html')

# API for categorical column "Fuel_Type"
@app.route('/api/fuel_type_options')
def fuel_api_options():
    return jsonify(list(dataset['Fuel_Type'].unique()))

# API for categorical column "Seller_Type"
@app.route('/api/seller_type_options')
def seller_api_options():
    return jsonify(list(dataset['Seller_Type'].unique()))

# API for categorical column "Transmission"
@app.route('/api/transmission_type_options')
def transmission_api_options():
    return jsonify(list(dataset['Transmission'].unique()))

# Prediction API
@app.route('/api/selling_price_predict', methods=['POST'])
def selling_api():
    try:
        # Get data from UI in JSON format
        data = request.get_json()
        logging.info('Data received from UI or postman: %s', data)
        
        # Input validation
        required_keys = ['Year', 'Present_Price', 'Kms_Driven', 'Seller_Type', 'Transmission', 'Fuel_Type']
        if not all(key in data for key in required_keys):
            return jsonify({'error': 'Missing required keys in input data'}), 400
        
        # Convert data to appropriate types
        Year = int(data['Year'])
        Present_Price = float(data['Present_Price'])
        Kms_Driven = int(data['Kms_Driven'])
        Seller_Type = data['Seller_Type']
        Transmission = data['Transmission']
        Fuel_Type = data['Fuel_Type']

        # Predict selling price of car
        predicted_selling_price = predict_selling_price(Year, Present_Price, Kms_Driven, Seller_Type, Transmission, Fuel_Type)

        return jsonify({'predicted_selling_price': predicted_selling_price}), 200
    
    except Exception as e:
        logging.error('An error occurred during prediction: %s', str(e))
        return jsonify({'error': 'An error occurred during prediction'}), 500

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    app.run(host='0.0.0.0', port=config.PORT_NUMBER, debug=False)
