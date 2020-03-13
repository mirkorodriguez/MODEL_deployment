#!/usr/bin/env python

import numpy as np
import tensorflow as tf
from grpc.beta import implementations

import predict_pb2
import prediction_service_pb2

from keras.preprocessing import image

tf.app.flags.DEFINE_string("host", "127.0.0.1", "gRPC server host")
tf.app.flags.DEFINE_integer("port", 8500, "gRPC server port")
tf.app.flags.DEFINE_string("model_name", "flowers_model_tl-full", "TensorFlow model name")
tf.app.flags.DEFINE_integer("model_version", 1, "TensorFlow model version")
tf.app.flags.DEFINE_float("request_timeout", 10.0, "Timeout of gRPC request")
FLAGS = tf.app.flags.FLAGS


def main():
  host = FLAGS.host
  port = FLAGS.port
  model_name = FLAGS.model_name
  model_version = FLAGS.model_version
  request_timeout = FLAGS.request_timeout

  image_filepaths = ["test-image.jpg"]

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

if __name__ == '__main__':
  main()
