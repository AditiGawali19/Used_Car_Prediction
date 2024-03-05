from flask import Flask, render_template, request, jsonify
import pandas as pd
from utils import predict_selling_price
import config

app = Flask(__name__)

# Load the dataset to get dropdown options
dataset = pd.read_csv(r'data\car data.csv')

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

# Prediction API                                  # api for target column
@app.route('/api/selling_price_predict', methods=['GET','POST'])
def selling_api():
    print("*"*40)
    #Get data from UI in JSON format
    data = request.get_json()
    
    # Get all X (independent) variable from JSON varialbe i.e. data
    Year = int(data['Year'])
    Present_Price = float(data['Present_Price'])
    Kms_Driven = int(data['Kms_Driven'])
    Seller_Type = data['Seller_Type']
    Transmission = data['Transmission']
    Fuel_Type = data['Fuel_Type']
    print('data received from UI or postman:', data)

    # Predict selling price of car by calling below function in Utils.py
    prd_selling_price = predict_selling_price(Year, Present_Price, Kms_Driven, Seller_Type, Transmission, Fuel_Type)

   # prd_selling_price_str = str(predicted_selling_price) 

    return jsonify({'predicted_selling_price': prd_selling_price})


if __name__ == '__main__':
    app.run(host='0.0.0.0',port=config.PORT_NUMBER,debug=False)
