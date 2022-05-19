from constants import dataset_path, IMAGE_SIZE
import cv2
import os

def read_image_data(sub_path):
    image_data = []
    for dirname, _, files in os.walk(dataset_path + sub_path):
        for file in files:
            image_path = dataset_path + sub_path + file
            image_data.append(cv2.resize(cv2.imread(image_path)[..., ::-1], IMAGE_SIZE) / 255)
    return image_data