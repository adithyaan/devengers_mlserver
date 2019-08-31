from flask import Flask
from flask import request
import json
import pandas as pd
from flask_cors import CORS


app = Flask(__name__)
CORS(app)

@app.route('/userdata')
def fetchdata():
    df = pd.read_csv('testms.csv') 
    data = df.to_json()
    return json.dumps({"data":data})