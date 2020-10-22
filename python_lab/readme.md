pip install --upgrade pip
pip list

pip install tensorflow
pip install tensorflow_datasets
pip install tf2onnx

pip install matplotlib
pip install keras2onnx
pip install onnxruntime

pip install pydot
pip install graphviz

pip install torch torchvision


Useful pyenv commands:
brew install pyenv
brew install pyenv-virtualenv

pyenv install --list
pyenv versions
pyenv global 3.8.5
pyenv local 3.8.5
pyenv global system
pyenv local system
pyenv which pip

pyenv virtualenv 3.8.5 onnx
cd onnx
pyenv local onnx

Common Errors for Pytorch models after converted to ONNX:

onnxruntime.capi.onnxruntime_pybind11_state.InvalidArgument: [ONNXRuntimeError] : 2 : INVALID_ARGUMENT : Invalid rank for input: input.1 Got: 2 Expected: 3 Please fix either the inputs or the model.

onnxruntime.capi.onnxruntime_pybind11_state.InvalidArgument: [ONNXRuntimeError] : 2 : INVALID_ARGUMENT : Got invalid dimensions for input: input.1 for the following indices
 index: 0 Got: 6 Expected: 2
 Please fix either the inputs or the model.


https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/input_data.py

# To run a .NET app:
dotnet run