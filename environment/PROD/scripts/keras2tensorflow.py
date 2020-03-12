import tensorflow as tf

path_to_load = "models/convnet/"

path_to_save = "models/convnet/tensorflow/"

# The export path contains the name and the version of the model
tf.keras.backend.set_learning_phase(0)  # Ignore dropout at inference

models = ['flowers_model_tl-full']
models_version = '1'

# Fetch the Keras session and save the model
# The signature definition is defined by the input and output tensors
# And stored with the default serving key
with tf.keras.backend.get_session() as sess:
    for model_name in models:
        print ("\nConverting keras model:", model_name, "to TensorFlow *.pb")
        model = tf.keras.models.load_model(''.join([path_to_load,model_name,'.h5']))
        model.summary()
        tf.saved_model.simple_save(sess,
                                    ''.join([path_to_save,model_name,'/',models_version]),
                                    inputs={'input_image': model.input},
                                    outputs={t.name: t for t in model.outputs})
