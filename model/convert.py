
from os import listdir
from keras.models import load_model
from tensorflow import lite

file = sorted(listdir('saves'))[-1]

with open('model.tflite', 'wb') as model:
	model.write(lite.TFLiteConverter.from_keras_model(load_model(f'saves/{file}')).convert())

print(f'saved saves/{file} as model.tflite')