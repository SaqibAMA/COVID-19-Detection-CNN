from keras.models import load_model
import cv2

model = load_model('src/model.h5')
model.load_weights('src/weight.h5')

# load validation dataset
