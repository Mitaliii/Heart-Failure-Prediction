from flask import Flask,render_template,request,url_for,jsonify,redirect
import requests
import pandas as pd
import numpy as np
import pickle

model = pickle.load(open('heart_prediction.pkl','rb'))

app=Flask(__name__)

@app.route('/',methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    jsson=request.form.to_dict()
    values_list=list(jsson.values())
    output=model.predict(pd.DataFrame([values_list]))
    return render_template('predict.html',prediction_text="Heart Failure Prediction : "+str(output[0]))

if __name__=="__main__":
    app.run(debug=True,port=5000)
