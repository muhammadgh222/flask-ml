from flask import Flask,request,render_template,url_for, jsonify
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib as joblib
import os

model=joblib.load('iris_model_LR.pkl')
scaler=joblib.load('scaler.save')

app =Flask(__name__)

IMG_FOLDER=os.path.join('static','IMG')
app.config['UPLOAD_FOLDER']=IMG_FOLDER


@app.route('/hi')
def index():
    return "hello"

@app.route('/',methods=['POST'])
def home():
    req_data=request.get_json()
    raw_data = {
        'sl':req_data['sl'],
        'sw':req_data['sw'],
        'pl':req_data['pl'],
        'pw':req_data['pw'],
    }
    data = np.array([[raw_data['sl'], raw_data['sw'], raw_data['pl'], raw_data['pw']]])
    x = scaler.transform(data)
    print(x)
    prediction = model.predict(x)
    print(prediction)
    image=prediction[0]+'.png'
    image=os.path.join(app.config['UPLOAD_FOLDER'],image)
    print(image)
    return image


if __name__ == '__main__':
    app.run(debug=True)