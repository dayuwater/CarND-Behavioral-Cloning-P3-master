import csv
import cv2
import numpy as np
import sklearn
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Dropout
from keras.layers import Cropping2D
from sklearn.model_selection import train_test_split

samples = []
# directory = "./data" # change to other dataset to test
# directory_special_case = "./curv2/curve"

# change this to the directories of your training data
directories = ["./data", "./curv2/curve", "./cw"]

def transform_path(directory, line):
    for i in range(0,3):
        source_path = line[i]
        last_part = source_path.split("/")[-1]
        source_path = directory + "/IMG/" + last_part
        line[i] = source_path
        
    return line


for dirr in directories:
    with open(dirr + "/driving_log.csv") as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            line = transform_path(dirr, line)
            samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples[1:], test_size=0.2)


# A generator is required because the memory is severely limited on AWS
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            measurements = []
            for line in batch_samples:
                for i in range(0,3):
                    source_path = line[i]
                    image = cv2.imread(source_path)
                    images.append(image)
                    measurement = float(line[3])
                    # if the image is from left camera, add +0.2 turning angle (to right)
                    if i == 1:
                        measurement += 0.2
                    # if the image is from right camera, add -0.2 turning angle (to left)
                    elif i == 2:
                        measurement -= 0.2
                    measurements.append(measurement)

            augmented_images, augmented_measurements = [], []
            for image, measurement in zip(images, measurements):
                augmented_images.append(image)
                augmented_measurements.append(measurement)
                augmented_images.append(cv2.flip(image,1))
                augmented_measurements.append(-1.0*measurement)
                
            X_train = np.array(augmented_images)
            y_train = np.array(augmented_measurements)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)


# NVidia Model
# Need to add dropout layers to reduce overfitting

model = Sequential()
model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Dropout(0.5))
model.add(Convolution2D(24,5,5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(64,3,3, activation="relu"))
model.add(Convolution2D(64,3,3, activation="relu"))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))





model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch= \
            len(train_samples), validation_data=validation_generator, \
            nb_val_samples=len(validation_samples),  nb_epoch=20)
model.save("model_data_improved.h5")