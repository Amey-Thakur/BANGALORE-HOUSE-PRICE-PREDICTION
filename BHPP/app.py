from flask import Flask, render_template, request ,url_for
import pickle
import numpy as np
import json


app = Flask(__name__)

__locations = None
__data_columns = None
model = pickle.load(open('banglore_home_prices_model.pickle','rb'))

f = open('columns.json')
__data_columns = json.loads(f.read())['data_columns']
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

    
@app.route('/')

def index():
    return render_template('index.html', locations=__locations)

@app.route('/predict',methods=['POST'])
def predict():

    if request.method == 'POST':
        input_json = {
            "location": request.form['sLocation'],
            "sqft": request.form['Squareft'],
            "bhk": request.form['uiBHK'],
            "bath": request.form['uiBathrooms']
        }
        result = get_estimated_price(input_json)

        if result > 100:
            result = round(result/100, 2)
            result = str(result) + ' Crore'
        else:
            result = str(result) + ' Lakhs'

    return render_template('predict.html',result=result)


if __name__=='__main__':
    app.run(debug=True,port=5000)