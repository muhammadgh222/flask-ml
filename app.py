from flask import Flask,request,render_template,url_for
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib as joblib
import os

model=joblib.load('iris_model_LR.pkl')
scaler=joblib.load('scaler.save')

app =Flask(__name__)

# IMG_FOLDER=os.path.join('static','IMG')
# app.config['UPLOAD_FOLDER']=IMG_FOLDER


@app.route('/hi')
def index():
    return "hello"

@app.route('/',methods=['GET','POST'])
def home():
    if request.method=='GET':
        return "Hello from home"


if __name__ == '__main__':
    app.run(debug=True)