'''
Note: torch.onnx does not needd to be installed as it comes with Pytorch.
'''
import datetime
import math
import os
from time import time

import matplotlib.pyplot as plt
import numpy as np
import onnx
import onnxruntime
import torch
from torch import nn
from torch import optim
import torch.onnx as torch_onnx
import torchvision
from torchvision import datasets, transforms


MODEL_PATH = './pytorch/pytorch_mnist_model.pt'
ONNX_MODEL_FILE = 'pytorch_mnist.onnx'

def load_data():
    # Define a transform to normalize the data
    transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                                ])

    # Download and load the training data
    train = datasets.MNIST('./mnist_data/', download=False, train=True, transform=transform)
    test = datasets.MNIST('./mnist_data/', download=False, train=False, transform=transform)
    #train = datasets.MNIST('./mnist_data/', download=False, train=True)
    #test = datasets.MNIST('./mnist_data/', download=False, train=False)
    train_loader = torch.utils.data.DataLoader(train, batch_size=64, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test, batch_size=64, shuffle=False)
    return train_loader, test_loader


def explore_data(train_loader):
    dataiter = iter(train_loader)
    images, labels = dataiter.next()
    print(type(images))
    print(images.shape)
    print(labels.shape)
    print(labels[0])
    plt.imshow(images[0].numpy().squeeze(), cmap='gray_r')
    plt.show()


def build_model():
    # Layer details for the neural network
    input_size = 784
    hidden_sizes = [128, 64]
    output_size = 10

    # Build a feed-forward network
    model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                        nn.ReLU(),
                        nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                        nn.ReLU(),
                        nn.Linear(hidden_sizes[1], output_size),
                        nn.LogSoftmax(dim=1))
    print(model)
    return model


def train_model(model, train_loader):
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)

    time0 = time()
    epochs = 15
    for e in range(epochs):
        running_loss = 0
        for images, labels in train_loader:
            # Flatten MNIST images into a 784 long vector
            images = images.view(images.shape[0], -1)
        
            # Training pass
            optimizer.zero_grad()
            
            output = model(images)
            loss = criterion(output, labels)
            
            #This is where the model learns by backpropagating
            loss.backward()
            
            #And optimizes its weights here
            optimizer.step()
            
            running_loss += loss.item()
        else:
            print("Epoch {} - Training loss: {}".format(e, running_loss/len(train_loader)))

    print("\nTraining Time (in minutes) =",(time()-time0)/60)


def evaluate_model(model, test_loader):
    correct_count, all_count = 0, 0

    for images,labels in test_loader:
        for i in range(len(labels)):
            img = images[i].view(1, 784)
            # Turn off gradients to speed up this part
            with torch.no_grad():
                logps = model(img)

            # Output of the network are log-probabilities, need to take exponential for probabilities
            ps = torch.exp(logps)
            probab = list(ps.numpy()[0])
            pred_label = probab.index(max(probab))
            true_label = labels.numpy()[i]
            if(true_label == pred_label):
                correct_count += 1
            all_count += 1

    print("Number Of Images Tested =", all_count)
    print("\nModel Accuracy =", (correct_count/all_count))


def get_images(sample_indexes, loader, flatten=False):
    dataiter = iter(loader)
    images, labels = dataiter.next()

    image_samples = []
    for index in sample_indexes:
        if flatten:
            image_samples.append(images[index].view(1, 784))
        else:
            image_samples.append(images[index])

    return image_samples


def predict(model, image_samples):
    '''
    Predict the number for a list of image samples.
    image_samples must be flattened.
    '''
    predictions = []
    for image in image_samples:
        #img = image.view(1, 784)

        # Turn off gradients to speed up this part
        with torch.no_grad():
            logps = model(image)

        # Output of the network are log-probabilities, need to take exponential for probabilities
        ps = torch.exp(logps)
        probab = list(ps.numpy()[0])
        pred_label = probab.index(max(probab))
        predictions.append(pred_label)

    return predictions


def plot_images_from_loader(loader):
    dataiter = iter(loader)
    images, labels = dataiter.next()
    figure = plt.figure()
    num_of_images = 60
    for index in range(1, num_of_images + 1):
        plt.subplot(6, 10, index)
        plt.axis('off')
        plt.imshow(images[index].numpy().squeeze(), cmap='gray_r')
    plt.show()


def plot_images(image_samples, columns=5):
    rows = math.ceil(len(image_samples)/columns)

    figure = plt.figure()
    num_of_images = len(image_samples)
    for index in range(1, num_of_images+1):
        plt.subplot(rows, columns, index)
        plt.axis('off')
        plt.imshow(image_samples[index-1].numpy().squeeze(), cmap='gray_r')

    plt.show()


def save_framework_model(model):
    torch.save(model, MODEL_PATH) 


def load_framework_model():
    model = torch.load(MODEL_PATH)
    return model


def export_to_onnx(model):
    '''
    Export the Pytorch model to the ONNX format.
    The Pytorch export requires sample input data in order to determine the input parameter.
    This must be a value that is of similar type and shape as an input image.
    '''
    #batch_size = 100
    #sample_input = torch.randn(batch_size, 1, 784, requires_grad=True)
    sample_input = torch.randn(1, 784)
    print(type(sample_input))
    torch.onnx.export(model,               # model being run
                      sample_input,                         # model input (or a tuple for multiple inputs)
                      ONNX_MODEL_FILE,   # where to save the model (can be a file or file-like object)
                      #export_params=True,        # store the trained parameter weights inside the model file
                      #opset_version=10,          # the ONNX version to export the model to
                      #do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names = ['input'],   # the model's input names
                      output_names = ['output'] # the model's output names
                      #dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                      #              'output' : {0 : 'batch_size'}}
                      )

    # Set metadata on the model.
    onnx_model = onnx.load(ONNX_MODEL_FILE)
    meta = onnx_model.metadata_props.add()
    meta.key = "creation_date"
    meta.value = datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    meta = onnx_model.metadata_props.add()
    meta.key = "author"
    meta.value = 'keithpij'
    onnx_model.doc_string = 'MNIST model converted from Pytorch'
    onnx_model.model_version = 3  # This must be an integer or long.
    onnx.save(onnx_model, ONNX_MODEL_FILE)



def check_onnx_model():
    # Load the ONNX model
    onnx_model = onnx.load(ONNX_MODEL_FILE)

    # Check that the IR is well formed
    onnx.checker.check_model(onnx_model)

    # Print a human readable representation of the graph
    #onnx.helper.printable_graph(onnx_model.graph)
    print('Model :\n\n{}'.format(onnx.helper.printable_graph(onnx_model.graph)))

    meta = onnx_model.metadata_props.add()
    meta.key = "creation_date"
    meta.value = datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    meta = onnx_model.metadata_props.add()
    meta.key = "author"
    meta.value = 'keithpij'
    onnx_model.doc_string = 'MNIST model converted from Pytorch'
    onnx_model.model_version = 3  # This must be an integer or long.

    return onnx_model


def load_onnx_model():

    try:
        session = onnxruntime.InferenceSession(ONNX_MODEL_FILE)

    except (InvalidGraph, TypeError, RuntimeError) as e:
        # It is possible for there to be a mismatch between the onnxruntime and the
        # version of the onnx model format.
        print(e)
        raise e

    return session


def onnx_infer(onnx_session, image_samples):
    input_name = onnx_session.get_inputs()[0].name

    np_samples = []
    for sample in image_samples:
        np_sample = np.array(sample)

        result = onnx_session.run(None, {input_name: np_sample})
        probabilities = np.array(result[0])
        #print(type(probabilities))
        print(probabilities)

        # Generate arg maxes for predictions
        predictions = np.argmax(probabilities, axis=1)
        print(predictions)


if __name__ == '__main__':
    train_loader, test_loader = load_data()
    #explore_data(train_loader)
    #plot_images_from_loader(test_loader)

    model = build_model()
    train_model(model, train_loader)
    save_framework_model(model)
    
    model = load_framework_model()
    export_to_onnx(model)
    check_onnx_model()
    #evaluate_model(model, test_loader)

    sample_indexes = [0]
    image_samples = get_images(sample_indexes, test_loader)
    plot_images(image_samples, columns=len(sample_indexes))
    
    #predictions = predict(model, image_samples)
    #print(predictions)

    onnx_session = load_onnx_model()
    #print_data(onnx_session)
    flat_image_samples = get_images(sample_indexes, test_loader, flatten=True)
    onnx_infer(onnx_session, flat_image_samples)
