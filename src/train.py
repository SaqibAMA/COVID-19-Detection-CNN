"""
Created by: Saqib Ali
train.py -- allows the user to train the exising network to 25 epochs.
"""

from lzma import MODE_FAST
import numpy as np
import sklearn.utils
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from keras.optimizers import Adam
from keras.losses import SparseCategoricalCrossentropy
from constants import MODEL_PATH, WEIGHT_PATH
from utils import read_image_data

# Reading training data from the dataset

image_data = {
    'covid': np.array([]),
    'normal': np.array([])
}

image_data['covid'] = read_image_data('/Train/COVID-19/images/')
image_data['normal'] = read_image_data('/Train/Normal/images/')

print("Data loaded.")

# Balancing data, if unbalanced

print("COVID Samples: ", len(image_data['covid']))
print("Normal Samples: ", len(image_data['normal']))

upper_bound = min(len(image_data['covid']), len(image_data['normal']))
print("\nSample Upper Bound: ", upper_bound)

image_data['covid'] = image_data['covid'][:upper_bound]
image_data['normal'] = image_data['normal'][:upper_bound]

print("\nCOVID Samples after balancing: ", len(image_data['covid']))
print("Normal Samples after balancing: ", len(image_data['normal']))

print("Data balanced.")

# Reshaping the data
reshaped_data = {
    'covid': [],
    'normal': []
}

# Generating X and Y

# X = np.concatenate((reshaped_data['covid'], reshaped_data['normal']))
X = np.concatenate((np.array(image_data['covid']), np.array(image_data['normal'])))
Y = np.concatenate((
    np.full(len(image_data['covid']), 0),
    np.full(len(image_data['normal']), 1)
))

# shuffling data
X, Y = sklearn.utils.shuffle(X, Y)

# Since the data we're reading is completely Test data, we don't have to split it.

# Creating our neural network

model = load_model(MODEL_PATH)
model.load_weights(WEIGHT_PATH)

## Uncomment the model if you don't have any previous training data

# model = Sequential([
#     Conv2D(32, 3, padding="same", activation="relu", input_shape=X.shape[1:]),
#     MaxPool2D(),
#     Conv2D(32, 3, padding="same", activation="relu"),
#     MaxPool2D(),
#     Conv2D(64, 3, padding="same", activation="relu"),
#     MaxPool2D(),
#     Dropout(0.5),
#     Flatten(),
#     Dense(128, activation="relu"),
#     Dense(2, activation="softmax")
# ])


# Compiling the model

model.compile(optimizer=Adam(lr=0.0001), loss=SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
history = model.fit(X, Y, epochs=25)

# Saving the model weights and model attributes
model.save_weights(WEIGHT_PATH)
model.save(MODEL_PATH)
