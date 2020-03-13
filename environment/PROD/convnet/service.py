#Import Flask
# from flask import Flask, request, jsonify, redirect
from flask_cors import CORS
#Import Keras
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions
import numpy as np

import requests
import json
import os
from werkzeug.utils import secure_filename
#Import TensorFlow
import tensorflow as tf
import predict_pb2
import prediction_service_pb2
from grpc.beta import implementations

UPLOAD_FOLDER = '../../../samples/images/uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

from flask import Flask
#Initialize the application service
app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Funciones
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict(model_name):
    from flask import request
    from flask import jsonify
    from flask import redirect
    print("...... calling predict ......")
    data = {"success": False}
    if request.method == "POST":
        # check if the post request has the file part
        if 'file' not in request.files:
            print('No file part')
        file = request.files['file']
        # if user does not select file, browser also submit a empty part without filename
        if file.filename == '':
            print('No selected file')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            #loading image
            filename = UPLOAD_FOLDER + '/' + filename
            print("\nfilename:",filename)

            host = "127.0.0.1"
            port = 8500
            model_name = model_name
            model_version = 1
            request_timeout = 10.0

            image_filepaths = [filename]

            for index, image_filepath in enumerate(image_filepaths):
                image_ndarray = image.img_to_array(image.load_img(image_filepaths[0], target_size=(224, 224)))
                image_ndarray = image_ndarray / 255.

            # Create gRPC client and request
            channel = implementations.insecure_channel(host, port)
            stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
            request = predict_pb2.PredictRequest()
            request.model_spec.name = model_name
            request.model_spec.version.value = model_version
            request.inputs['input_image'].CopyFrom(tf.contrib.util.make_tensor_proto(image_ndarray, shape=[1] + list(image_ndarray.shape)))

            # Send request
            result = str(stub.Predict(request, request_timeout))
            mylist = result.split('\n')[-8:-3]
            finallist = []
            for element in mylist:
                  element = element.split(':')[1]
                  finallist.append(float("{:.6f}".format(float(element))))

            index = finallist.index(max(finallist))
            CLASSES = ['Daisy', 'Dandelion', 'Rosa', 'Girasol', 'Tulipán']

            ClassPred = CLASSES[index]
            ClassProb = finallist[index]

            print(finallist)
            print(ClassPred)
            print(ClassProb)

            label = ClassPred
            score = ClassProb

            #Results as Json
            data["predictions"] = []
            r = {"label": label, "score": float(score)}
            data["predictions"].append(r)

            #Success
            data["success"] = True

    return jsonify(data)


#Define a route
@app.route('/')
def default():
    return 'TensorFlow Serving ... Go to /flores/predict'

# Main
@app.route('/flores/predict/',methods=['POST'])
def flores():
    model_name = "flowers_model_tl-full"
    return (predict(model_name))

# Run de application
app.run(host='0.0.0.0',port=5000)
