from os import listdir
from tensorflow import TensorSpec
from keras.models import load_model
from tf2onnx.convert import from_keras

file = sorted(listdir('saves'))[-1]
model = load_model(f'saves/{file}')
model.output_names = []

from_keras(model, [TensorSpec(model.inputs[0].shape, model.inputs[0].dtype)], opset=13, output_path='model.onnx')

print(f'saved saves/{file} as model.onnx')