from flask import Flask, request, render_template, url_for, jsonify
from tensorflow import keras
from keras.layers import Dense
from keras.models import Sequential, load_model
from PIL import Image
import numpy as np

app = Flask(__name__)

def preprossing(image):
    image=Image.open(image)
    image = image.resize((150, 150))
    image_arr = np.array(image.convert('RGB'))
    image_arr.shape = (1, 150, 150, 3)
    return image_arr

classes = ['Buildings' ,'Forest', 'Glacier' ,'Mountain' ,'Sea' ,'Street']
model=load_model("Intel_Image_Classification.h5")

@app.route('/')
def index():

    return "hello"


@app.route('/predictApi', methods=["POST"])
def api():
    # Get the image from post request
    try:
        if 'fileup' not in request.files:
            return "Please try again. The Image doesn't exist"
        image = request.files.get('fileup')
        image_arr = preprossing(image)
        print("Model predicting ...")
        result = model.predict(image_arr)
        print("Model predicted", result)
        ind = np.argmax(result)
        prediction = classes[ind]
        print(prediction)
        return jsonify({'prediction': prediction})
    except Exception as e: print(e)



if __name__ == '__main__':
    app.run(debug=True)