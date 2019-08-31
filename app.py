from flask import Flask
from flask import request
import json
import pandas as pd
from flask_cors import CORS
import pickle

app = Flask(__name__)
CORS(app)

@app.route('/userdata')
def fetchdata():
    df = pd.read_csv('testms.csv') 
    data = []
    keys = df.columns
    for i,j in df.iterrows():
        item = {}
        for index,val in enumerate(j):
            item[keys[index]]=val
        data.append(item)
    return json.dumps({"data":data})

@app.route('/test')
def test():
    return "hello"

@app.route('/predict')
def predict():
    with open('model.obj','rb') as fp:
        model = pickle.load(fp)
        data = request.get_json()
        to_extract= ['Gender', 'family_history', 'work_interfere', 'remote_work',
       'tech_company', 'benefits', 'care_options', 'wellness_program',
       'seek_help', 'mental_health_consequence', 'phys_health_consequence',
       'mental_health_interview', 'phys_health_interview',
       'mental_vs_physical', 'obs_consequence']
        input=[]
        for i in to_extract:
            input.append(data[i])
        result = model.predict(input)
        return str(result)