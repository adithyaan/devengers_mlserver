from flask import Flask
from flask import request
import json
import pandas as pd
from flask_cors import CORS
import pickle
from sklearn import preprocessing

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
            if pd.isna(val):
                item[keys[index]]=str(val)
            else:
                item[keys[index]]=val
        data.append(item)
    return json.dumps({"data":data})

@app.route('/test')
def test():
    return "hello"

@app.route('/predict',methods=['POST'])
def predict():
    with open('d_model.obj','rb') as fp:
        model = pickle.load(fp)
        data = request.get_json()
        to_extract= ['Gender', 'family_history', 'work_interfere', 'remote_work',
       'tech_company', 'benefits', 'care_options', 'wellness_program',
       'seek_help', 'mental_health_consequence', 'phys_health_consequence',
       'mental_health_interview', 'phys_health_interview',
       'mental_vs_physical', 'obs_consequence']
        input=[]
        keys=[]
        for i in to_extract:
            input.append(data[i])
            keys.append(i)
        
        test = pd.DataFrame([input],columns=keys)

        # temp = test.Age
        test['work_interfere'] = test['work_interfere'].replace(to_replace="nan",value="Maybe")
        # test.Age = temp

        # test['work_interfere']=test['work_interfere'].fillna("Maybe")
        # test['self_employed']=test['self_employed'].fillna("Dont Know")

        temp = test.Gender
        temp = temp.replace(to_replace=["Male","male","m","Malr","Male ","Cis Man"],value='M')
        temp = temp.replace(to_replace=["female","F","Female","Woman","femail","f"],value='F')
        temp = temp.replace(to_replace=["p","Female (trans)","ostensibly male, unsure what that really means"],value='T')
        test.Gender = temp
        with open('dict.obj',"rb") as fp:
            d = pickle.load(fp)
            test = test.apply(lambda x: d[x.name].transform(x)) 
        
        result = model.predict(test)
        print(result)
        result = 'Yes' if result[0]==0 else 'No'
        print(result)
        return {"data":result}