"""
Project: Bangalore House Price Prediction
Description: A Flask-based web application that serves a Machine Learning model (Linear Regression) 
             to predict real estate prices in Bangalore based on user inputs.

Authors:
  - Amey Thakur (https://github.com/Amey-Thakur)
  - Mega Satish (https://github.com/msatmod)

Repository: https://github.com/Amey-Thakur/BANGALORE-HOUSE-PRICE-PREDICTION
Release Date: August 7, 2021
License: MIT License
"""

# ------------------------------------------------------------------------------
# Import Necessary Libraries
# ------------------------------------------------------------------------------
# Flask: Micro web framework for serving the application.
# render_template: Renders HTML templates from the 'templates' directory.
# request: Handles incoming HTTP requests (GET/POST).
# url_for: Generates URLs for functions (used in routing).
from flask import Flask, render_template, request, url_for

import pickle
import numpy as np
import json


app = Flask(__name__)

# ------------------------------------------------------------------------------
# Global Variable Initialization
# ------------------------------------------------------------------------------
# __locations: Holds the list of location names extracted from columns.json.
# __data_columns: Holds the complete list of feature columns used during model training.
# model: The deserialized Linear Regression model object.
# ------------------------------------------------------------------------------
__locations = None
__data_columns = None

# Load the trained machine learning model from the serialized pickle file.
# The 'rb' mode indicates reading in binary format.
model = pickle.load(open('bangalore_home_prices_model.pickle','rb'))

# Load the data columns schema.
# This ensures that the input features for inference match the order and structure
# of the features used during training (One-Hot Encoding alignment).
f = open('columns.json')
__data_columns = json.loads(f.read())['data_columns']

# Extract locations starting from the 4th column (index 3).
# The first 3 columns are typically 'sqft', 'bath', 'bhk' based on the training data structure.
__locations = __data_columns[3:]

def get_estimated_price(input_json):
    try:
        loc_index = __data_columns.index(input_json['location'].lower())
    except:
        loc_index = -1
    x = np.zeros(len(__data_columns))
    x[0] = input_json['sqft']
    x[1] = input_json['bath']
    x[2] = input_json['bhk']
    if loc_index >= 0:
        x[loc_index] = 1
    result = round(model.predict([x])[0],2)
    return result

    
# ------------------------------------------------------------------------------
# Route: Home Page
# Method: GET
# Description: Renders the main landing page (index.html) and passes the list of
#              available locations to populate the dropdown menu.
# ------------------------------------------------------------------------------
@app.route('/')
def index():
    return render_template('index.html', locations=__locations)

# ------------------------------------------------------------------------------
# Route: Predict Price
# Method: POST
# Description: Handles the form submission, processes the input data, performs
#              inference using the loaded model, and returns the estimated price.
# ------------------------------------------------------------------------------
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Extraction of data from the HTML form using the 'name' attributes.
        input_json = {
            "location": request.form['sLocation'],
            "sqft": request.form['Squareft'],
            "bhk": request.form['uiBHK'],
            "bath": request.form['uiBathrooms']
        }
        
        # Calculate the estimated price using the helper function.
        result = get_estimated_price(input_json)

        # Formatting the result for display.
        # If the price is greater than 100 Lakhs, convert it to Crores for better readability.
        if result > 100:
            result = round(result/100, 2)
            result = str(result) + ' Crore'
        else:
            result = str(result) + ' Lakhs'

    # Render the prediction result page.
    return render_template('predict.html', result=result)


if __name__=='__main__':
    app.run(debug=True,port=5000)