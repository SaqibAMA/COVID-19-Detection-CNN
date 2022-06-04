# execution command: /Users/saqib/.conda/envs/covid-classification/bin/python src/app.py 

from keras.models import load_model
from tkinter import *
from tkinter import filedialog
import cv2
import numpy as np
from constants import BATCH_NORM_MODEL_PATH, IMAGE_SIZE, BATCH_NORM_WEIGHT_PATH

model_batch_norm = load_model(BATCH_NORM_MODEL_PATH)
model_batch_norm.load_weights(BATCH_NORM_WEIGHT_PATH)


def upload_image(root):
    # take image upload from user (only png) and convert it to a numpy array
    root.filename = filedialog.askopenfilename(initialdir="/", title="Select a file", filetypes=(("png files", "*.png"), ("all files", "*.*")))
    if (root.filename == ""):
        return
    img = cv2.resize(cv2.imread(root.filename), IMAGE_SIZE)
    img = img.reshape(1, IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
    img = img.astype('float32')

    # predict the image
    prediction_batch_norm = model_batch_norm.predict(img)

    print("Prediction (Batch Normalization Model): ", prediction_batch_norm)

    prediction_options = ['COVID', 'Normal']

    Label(root, text="Filename: " + root.filename.split('/')[-1]).pack(padx=2, pady=2)
    Label(root, text="Prediction: " + prediction_options[np.argmax(prediction_batch_norm)]).pack(padx=2)
    # Label(root, text="Prediction (Batch Normalized): " + prediction_options[np.argmax(prediction_batch_norm)]).pack(padx=2)
    Label(root, text="--").pack(padx=2, pady=2)


def main():
    root = Tk()
    root.title("COVID-19 Detection - Saqib Ali")
    root.geometry("480x800")

    # show heading "COVID-19 Detection App" at the top of the screen
    heading = Label(root, text="COVID-19 Detection App", font=("Arial Bold", 22))
    heading.pack(padx=10, pady=10)

    # show an image panel to show the selected image
    img = PhotoImage(file="src/assets//covid.png")
    Label(root, image=img).pack(padx=10, pady=10)

    # show a button to upload an image
    upload_button = Button(root, text="Upload Image", command=lambda: upload_image(root))
    upload_button.config(width=25, height=2)
    upload_button.pack(padx=10, pady=10)

    root.mainloop()


if __name__ == '__main__':
    main()
