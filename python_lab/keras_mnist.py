'''
Title: MNIST Model creation and export to INNX.
Author: Keith Pijanowski
Date created: 2020/07/19
Last modified: 2020/07/21
Description: A simple convnet that achieves ~99% test accuracy on MNIST.
'''
import datetime
import math

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
import keras2onnx
import onnxruntime


# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)
batch_size = 128
epochs = 15
img_width, img_height = 28, 28
models_file_path = './keras'
ONNX_MODEL_FILE = 'keras_mnist.onnx'


def load_data():
    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    # Make sure images have shape (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    return x_train, x_test, y_train, y_test


def build_model():

    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    return model


def train_model(model, x_train, y_train):
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)


def evaluate_model(model, x_test, y_test):
    score = model.evaluate(x_test, y_test, verbose=0)
    return score


def save_framework_model(model):
    keras.models.save_model(model, models_file_path)


def load_framework_model():
    return keras.models.load_model(models_file_path, compile=True)


def plot_model(model):
    keras.utils.plot_model(model, to_file='plot.png', show_shapes=True)


def get_images(sample_indexes):
    image_samples = []
    for sample in sample_indexes:
        image_samples.append(x_train[sample])
    return image_samples


def plot_images(image_samples, columns=5):
    rows = math.ceil(len(image_samples)/columns)
    fig, axs = plt.subplots(rows, columns, sharex=True, sharey=True)
    fig.suptitle('MNIST Images')

    # Generate plots for samples
    index = 0
    for row in range(0, rows):
        for column in range(0, columns):
            # Generate a plot
            #reshaped_image = x_train[sample].reshape((img_width, img_height))
            #plt.imshow(sample)
            #plt.show()
            if index > len(image_samples)-1:
                axs[row,column].axis('off')
            else:
                sample = image_samples[index]
                axs[row, column].imshow(sample)
                axs[row, column].set_title('Sample #{}'.format(index), fontsize=8)
                axs[row, column].tick_params(labelsize=8)
            index += 1

    plt.show()


def predict(model, image_samples):
    # A few random samples
    samples_to_predict = []

    # Convert into Numpy array
    samples_to_predict = np.array(image_samples)
    print(samples_to_predict.shape)

    probabilities = model.predict(samples_to_predict)
    print(type(probabilities))
    print(probabilities)

    # Generate arg maxes for predictions
    classes = np.argmax(probabilities, axis=1)
    print(classes)


def export_to_onnx(model):
    # convert to onnx model
    onnx_model = keras2onnx.convert_keras(model, model.name)
    meta = onnx_model.metadata_props.add()
    meta.key = "creation_date"
    meta.value = datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    meta = onnx_model.metadata_props.add()
    meta.key = "author"
    meta.value = 'keithpij'
    onnx_model.doc_string = 'MNIST model converted from Keras'
    onnx_model.model_version = 3  # This must be an integer or long.
    keras2onnx.save_model(onnx_model, ONNX_MODEL_FILE)


def load_onnx_model(onnx_model_file):

    try:
        session = onnxruntime.InferenceSession(onnx_model_file)

    except (InvalidGraph, TypeError, RuntimeError) as e:
        # It is possible for there to be a mismatch between the onnxruntime and the
        # version of the onnx model format.
        print(e)
        raise e

    return session


def print_onnx_model_data(session):
    inputs = session.get_inputs()
    for input in inputs:
        print(input.name)
        print(input.shape)
        print(input.type)

    outputs = session.get_outputs()
    for output in outputs:
        print(output.name)
        print(output.shape)
        print(output.type)

    providers = session.get_providers()
    print(providers)

    model_metadata = session.get_modelmeta()
    print(model_metadata.custom_metadata_map)
    print(model_metadata.description)
    print(model_metadata.domain)
    print(model_metadata.graph_name)
    print(model_metadata.producer_name)
    print(model_metadata.version)


def onnx_infer(onnx_session, image_samples):
    input_name = onnx_session.get_inputs()[0].name
    result = onnx_session.run(None, {input_name: image_samples})
    print(type(result))
    probabilities = np.array(result[0])
    print(type(probabilities))
    print(probabilities)

    # Generate arg maxes for predictions
    predictions = np.argmax(probabilities, axis=1)
    return predictions


if __name__ == '__main__':
    x_train, x_test, y_train, y_test = load_data()
    print("x_train shape:", x_train.shape)
    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")

    #model = build_framework_model()
    #train_model(model, x_train, y_train)
    #save_framework_model(model)

    model = load_framework_model()
    onnx_model_file = export_to_onnx(model)
    #plot_model(model)

    #model.summary()

    #score = evaluate_model(model, x_test, y_test)
    #print("Test loss:", score[0])
    #print("Test accuracy:", score[1])

    sample_indexes = [5, 100, 150, 300, 400, 401, 900]
    image_samples = get_images(sample_indexes)
    #plot_images(image_samples)
    
    #model = load_framework_model()
    #predict(model, image_samples)

    onnx_session = load_onnx_model(ONNX_MODEL_FILE)
    print_onnx_model_data(onnx_session)
    predictions = onnx_infer(onnx_session, image_samples)
    print(predictions)
