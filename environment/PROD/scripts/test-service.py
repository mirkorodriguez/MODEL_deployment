import argparse
import json
import numpy as np
import requests
from keras.applications import inception_v3, vgg16, resnet50, mobilenet
from keras.applications.imagenet_utils import decode_predictions
from keras.preprocessing import image

# Argumentos
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="La ruta de la imagen es necesaria.")
ap.add_argument("-m", "--model", required=True, help="El nombre del modelo es necesario.")
args = vars(ap.parse_args())

image_path = args['image']
model_name = args['model']

print("\nModel:",model_name)
print("Image:",image_path)


# Preprocesar imagen
img = image.img_to_array(image.load_img(image_path, target_size=(224, 224)))
img = img / 255.
img = img.astype('float32')

payload = {"instances": [{'input_image': img.tolist()}]}

# URI
uri = ''.join(['http://127.0.0.1:9000','/v1/models/',model_name,':predict'])
print("URI:",uri)

# Request al modelo desplegado en TensorFlow Serving
r = requests.post(uri, json=payload)
pred = json.loads(r.content.decode('utf-8'))

# Decodificando decoder util
predictions = decode_predictions(np.array(pred['predictions']),top=1)
print("Predictions:\n",predictions)
print("Class:",predictions[0][0][1])
print("Score",predictions[0][0][2])
