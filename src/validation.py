from keras.models import load_model
import cv2
from constants import BATCH_NORM_MODEL_PATH, BATCH_NORM_WEIGHT_PATH, IMAGE_SIZE
from utils import read_image_data
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

model = load_model(BATCH_NORM_MODEL_PATH)
model.load_weights(BATCH_NORM_WEIGHT_PATH)

# load validation dataset
image_data = {
    'covid': [],
    'normal': []
}

image_data['covid'] = read_image_data('/Val/COVID-19/images/')
image_data['normal'] = read_image_data('/Val/Normal/images/')

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

# Plotting confusion matrix

# print("Plotting confusion matrix...")

# val_X = np.concatenate((np.array(image_data['covid']), np.array(image_data['normal'])))
# val_Y = np.concatenate((
#     np.full(len(image_data['covid']), 0),
#     np.full(len(image_data['normal']), 1)
# ))

# cf_matrix = confusion_matrix(val_Y, model.predict_classes(val_X))

# ax = sns.heatmap(cf_matrix, annot=True, cmap="YlGnBu", fmt='g')
# ax.set_xlabel('Predicted labels')
# ax.set_ylabel('True labels')
# ax.set_title('Confusion Matrix')
# plt.show()