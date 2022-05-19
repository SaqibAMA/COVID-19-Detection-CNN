# execution command: /Users/saqib/.conda/envs/covid-classification/bin/python src/app.py 

from keras.models import load_model
from tkinter import *
from tkinter import filedialog
import cv2
import time

model = load_model('src/model.h5')
model.load_weights('src/weight.h5')


def upload_image(root):
    # take image upload from user (only png) and convert it to a numpy array
    root.filename = filedialog.askopenfilename(initialdir="/", title="Select a file", filetypes=(("png files", "*.png"), ("all files", "*.*")))
    if (root.filename == ""):
        return
    img = cv2.resize(cv2.imread(root.filename), (50, 50))
    img = img.reshape(1, 50, 50, 3)
    img = img.astype('float32')

    # predict the image
    prediction = model.predict(img)
    is_covid = prediction[0][0]


    Label(root, text="Filename: " + root.filename.split('/')[-1]).pack(padx=2, pady=2)
    Label(root, text="Prediction: " + ("COVID" if is_covid > 0.4 else "Normal"), fg="red" if is_covid > 0.5 else "green").pack(padx=2)
    Label(root, text="--").pack(padx=2, pady=2)


def main():
    root = Tk()
    root.title("COVID-19 Detection - Saqib Ali")
    root.geometry("480x800")

    # show heading "COVID-19 Detection App" at the top of the screen
    heading = Label(root, text="COVID-19 Detection App", font=("Arial Bold", 22))
    heading.pack(padx=10, pady=10)

    # show an image panel to show the selected image
    img = PhotoImage(file="src/covid.png")
    Label(root, image=img).pack(padx=10, pady=10)

    # show a button to upload an image
    upload_button = Button(root, text="Upload Image", command=lambda: upload_image(root))
    upload_button.config(width=25, height=2)
    upload_button.pack(padx=10, pady=10)

    root.mainloop()


if __name__ == '__main__':
    main()
