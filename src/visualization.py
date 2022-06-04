from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization
import visualkeras

model = Sequential([
    Conv2D(32, 3, padding="same", activation="relu", input_shape=(50, 50, 3)),
    MaxPool2D(),
    BatchNormalization(),
    Dropout(0.2),
    Conv2D(32, 3, padding="same", activation="relu"),
    MaxPool2D(),
    BatchNormalization(),
    Dropout(0.2),
    Conv2D(64, 3, padding="same", activation="relu"),
    MaxPool2D(),
    BatchNormalization(),
    Dropout(0.2),
    Flatten(),
    Dense(128, activation="relu"),
    Dense(2, activation="softmax")
])

visualkeras.layered_view(model, to_file='model_visualization.png', legend=True).show()