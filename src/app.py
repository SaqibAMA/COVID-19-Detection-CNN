from keras.models import load_model

model = load_model('model.h5')
model.load_weights('weight.h5')
