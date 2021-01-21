import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# image_path = os.path.join("dataset", "train_images", "6103.jpg")
# img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
# cv2.imshow('Image', img)
# plt.imshow(img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# print(img.shape)
# print(img.size)
# print(img[0][799])

# scale_percent = 25
# width = int(img.shape[1] * (scale_percent / 100))
# height = int(img.shape[0] * (scale_percent / 100))
# dim = (width, height)

# resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
# print(resized.shape)
# plt.imshow(resized)
# plt.show()

###############################################################################
# Getting the data and modify it into the correct format

# Read the training information csv file
def read_train_info():
    train_path = os.path.join("dataset", "train.csv")
    return pd.read_csv(train_path)

train_info = read_train_info()

# Return a cv2 image from the file path
def get_image(file_name):
    image_path = os.path.join("dataset", "train_images", file_name)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    scale_percent = 25
    width = int(image.shape[1] * (scale_percent / 100))
    height = int(image.shape[0] * (scale_percent / 100))
    dim = (width, height)
    resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized_image

# Convert cv2 image into list of pixels intensity
def convert_img_to_pixel(image):
    pixels_intensity = []
    
    for col in range(image.shape[1]):
        for row in range(len(image)):
            pixels_intensity.append(image[row][col])

    return pixels_intensity

# Loop through all the available training data
# and modify it into the correct array format
def create_training_data():
    X_train = [[i for i in range(30000)]]
    y_train = [[30000]]
    for ind in range(500):
        file_name = train_info["image_id"].loc[ind]
        image = get_image(file_name)
        converted_image = convert_img_to_pixel(image)
        X_train.append(converted_image)
        y_train.append([train_info["label"].loc[ind]])
        print("Converting Done", ind)

    return X_train, y_train

X_train, y_train = create_training_data()

# train = np.c_[X_train, y_train]

# Create columns name
# column_name = [i for i in range(480001)]
# for i in range(480000):
#     name = "Pixel" + str(i)
#     column_name.append(name)

# column_name.append("target")

# train = np.insert(train, 0, column_name, axis=0)

X_train_data_path = os.path.join("dataset", "X_train_data.csv")
y_train_data_path = os.path.join("dataset", "y_train_data.csv")
np.savetxt(X_train_data_path, X_train, delimiter=",", fmt="%i")
np.savetxt(y_train_data_path, y_train, delimiter=",", fmt="%i")

print("Done Modifying Data")

###############################################################################
# Feature Engineering and Training the model

# Read the training data csv file
def read_train_set():
    train_path = os.path.join("dataset", "train_data.csv")
    return pd.read_csv(train_path)

# train = read_train_set()
# print(train.iloc[:3, train.shape[1]-1])