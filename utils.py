import pickle
import json
import numpy as np
import config


def predict_selling_price(Year, Present_Price, Kms_Driven, Seller_Type, Transmission, Fuel_Type):

    print('input received in frunction predict_selling_price ',Year, Present_Price, Kms_Driven, Seller_Type, Transmission, Fuel_Type)
    model_file_path = config.MODEL_FILE_PATH
    drug_json_path = config.COLUMN_DATA_JSON
    

    with open(model_file_path, 'rb') as f:
        model = pickle.load(f)
    col_names = model.feature_names_in_

    with open(drug_json_path, 'rb') as f:
        col_data = json.load(f)
    
    Seller_Type = col_data['Seller_Type'][Seller_Type]
    Transmission = col_data['Transmission'][Transmission] 
    #Fuel_Type = col_data['Fuel_Type'][Fuel_Type]
    Fuel_Type_index = np.where(col_names == 'Fuel_Type_'+ Fuel_Type)[0][0]
    
    print("feature names",model.feature_names_in_)
    print("number of feature in model",model.n_features_in_)

    test_array = np.zeros((1,model.n_features_in_))
    test_array[0,0] = Year
    test_array[0,1] = Present_Price
    test_array[0,2] = Kms_Driven
    test_array[0,3] = Seller_Type
    test_array[0,4] = Transmission
    test_array[0,Fuel_Type_index] = 1

    #test_array = np.array([[Year, Present_Price, Kms_Driven, Seller_Type, Transmission, Fuel_Type]])
    Selling_price = model.predict(test_array)[0]
    print("Predicte selling price", Selling_price)
    return Selling_price
