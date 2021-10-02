import numpy as np
import pandas as pd
import requests
from flask import Flask, url_for, jsonify, render_template, request, redirect
import pickle
model = pickle.load(open('heart_prediction.pkl', 'rb'))
app = Flask(__name__)

@app.route('/',methods=['POST','GET])
def predict():
    float_features = [float(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)
    op_list = ["Positive","Negative"]

    return render_template('index.html', prediction_text='Patient belongs to  {} category'.format(op_list[output]))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    input= [[49.0,1,80,0,30,1,427000.0,1.0,138,0,0,12]]
    input= pd.DataFrame(input)
    output = model.predict(input)
    return "Predicted Output" + str(output[0])    


if __name__ == "__main__":
    app.run(debug=True,port=5000)