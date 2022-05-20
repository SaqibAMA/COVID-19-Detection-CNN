from keras.models import load_model
import cv2
from constants import MODEL_PATH, WEIGHT_PATH, IMAGE_SIZE
from utils import read_image_data
import numpy as np
import matplotlib.pyplot as plt

model = load_model(MODEL_PATH)
model.load_weights(WEIGHT_PATH)

# load validation dataset
image_data = {
    'covid': [],
    'normal': []
}

image_data['covid'] = read_image_data('/Test/COVID-19/images/')
image_data['normal'] = read_image_data('/Test/Normal/images/')

covid_predictions = model.predict(np.array(image_data['covid']))
normal_predictions = model.predict(np.array(image_data['normal']))

correct = {
    'covid': 0,
    'normal': 0
}

covid_plot = []

for p in covid_predictions:
    if np.argmax(p) == 0:
        correct['covid'] += 1
    covid_plot.append(correct['covid'])

for p in normal_predictions:
    if np.argmax(p) == 1:
        correct['normal'] += 1

print("Accuracy on detecting COVID-19: ", (correct['covid'] / len(covid_predictions)) * 100, '%')
print("Accuracy on detecting Normal: ", (correct['normal'] / len(normal_predictions)) * 100, '%')


print("Overall: ", ((correct['covid'] / len(covid_predictions)) * 100 + (correct['normal'] / len(normal_predictions)) * 100) / 2, '%')