from flask import Flask, request, render_template, url_for, jsonify
# import tensorflow
from tensorflow.keras.models import load_model
# from tensorflow import keras
# from keras.models import Sequential, load_model
from PIL import Image
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["SM_FRAMEWORK"] = "tf.keras"


app = Flask(__name__)

def preprossing(image):
    image=Image.open(image)
    image = image.resize((224, 224))
    image_arr = np.array(image.convert('RGB'))
    image_arr.shape = (1, 224, 224, 3)
    return image_arr

classes = ["Positive","Negative"]
model=load_model("covid19.h5",compile=False)

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
        print('REQUEST FILES: ', request.files)
        print('IMAGE BEFORE PROCESSING:',image)
        image_arr = preprossing(image)
        print('IMAGE AFTER PROCESSING: ',image_arr)
        print("Model predicting ...")
        result = model.predict(image_arr)
        print("Model predicted", result)
        ind = np.argmax(result)
        prediction = classes[ind]
        print(prediction)
        return jsonify({'prediction': prediction})
    except:
        return jsonify({'Error': 'Error occur'})



if __name__ == '__main__':
    app.run(debug=True)