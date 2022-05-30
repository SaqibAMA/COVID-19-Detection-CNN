"""
Created by: Saqib Ali
train.py -- allows the user to train the exising network to 25 epochs.
"""

# Output files: model_batch_norm.h5, weight_batch_norm.h5
# Observations: The modal quickly converges to the correct solution and the accuracy jumps 60% to 95% in just the first training run.

# Total epochs: 100

import numpy as np
import sklearn.utils
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.losses import SparseCategoricalCrossentropy
from constants import BATCH_NORM_MODEL_PATH, BATCH_NORM_WEIGHT_PATH
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

# Loading validation data

validation_images = {
    'covid': np.array([]),
    'normal': np.array([])
}

validation_images['covid'] = read_image_data('/Val/COVID-19/images/')
validation_images['normal'] = read_image_data('/Val/Normal/images/')

val_X = np.concatenate((validation_images['covid'], validation_images['normal']))
val_Y = np.concatenate((
    np.full(len(validation_images['covid']), 0),
    np.full(len(validation_images['normal']), 1)
))

# Generating X and Y

# X = np.concatenate((reshaped_data['covid'], reshaped_data['normal']))
X = np.concatenate((np.array(image_data['covid']), np.array(image_data['normal'])))
Y = np.concatenate((
    np.full(len(image_data['covid']), 0),
    np.full(len(image_data['normal']), 1)
))

# shuffling data
X, Y = sklearn.utils.shuffle(X, Y)
val_X, val_Y = sklearn.utils.shuffle(val_X, val_Y)

# Since the data we're reading is completely Test data, we don't have to split it.

# Creating our neural network

model = load_model(BATCH_NORM_MODEL_PATH)
model.load_weights(BATCH_NORM_WEIGHT_PATH)

## Uncomment the model if you don't have any previous training data

# model = Sequential([
#     Conv2D(32, 3, padding="same", activation="relu", input_shape=X.shape[1:]),
#     MaxPool2D(),
#     BatchNormalization(),
#     Dropout(0.2),
#     Conv2D(32, 3, padding="same", activation="relu"),
#     MaxPool2D(),
#     BatchNormalization(),
#     Dropout(0.2),
#     Conv2D(64, 3, padding="same", activation="relu"),
#     MaxPool2D(),
#     BatchNormalization(),
#     Dropout(0.2),
#     Flatten(),
#     Dense(128, activation="relu"),
#     Dense(2, activation="softmax")
# ])


# Compiling the model

model.compile(optimizer=Adam(lr=0.000001), loss=SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
history = model.fit(X, Y, epochs=25, validation_data=(val_X, val_Y))

# Saving the model weights and model attributes
model.save_weights(BATCH_NORM_WEIGHT_PATH)
model.save(BATCH_NORM_MODEL_PATH)