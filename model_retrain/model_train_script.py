import matplotlib
matplotlib.use(backend="Agg")

# importing required packages
from my_model.Jagga import Jagga

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.utils import np_utils
from imutils import paths

import matplotlib.pyplot as plt
import numpy as np
#from PIL import Image
import pickle
#import PIL
import cv2
import os


class JaggaRetrain:
    @staticmethod
    def model_retrain(learn_rate, batch_size, epochs, data, labels):
        # Initializing the learning rate, batch size and # of epoch to train

        learn_rate = learn_rate
        batch_size = batch_size
        epochs = epochs

        # loading the image dataset and initializing the list of data (images) and class images (labels)

        print("[INFO] Loading Train Images...")
        train_imagePaths = list(paths.list_images("D:/adity/Projects/image_classification/dataset/seg_train"))
        train_data = []
        train_labels = []

        for train_imagePath in train_imagePaths:
            train_label = train_imagePath.split(os.path.sep)[-2]
            train_image = cv2.imread(train_imagePath)
            train_image = cv2.resize(train_image, (32, 32))

            # updating data and labels list respectively
            train_data.append(train_image)
            train_labels.append(train_label)
        print(len(train_labels))
        print(len(train_data))

        # loading the image dataset and initializing the list of data (images) and class images (labels)
        # for testing images

        print("[INFO] Loading Test Images...")
        test_imagePaths = list(paths.list_images("D:/adity/Projects/image_classification/dataset/seg_test"))
        test_data = []
        test_labels = []

        for test_imagePath in test_imagePaths:
            test_label = test_imagePath.split(os.path.sep)[-2]
            test_image = cv2.imread(test_imagePath)
            test_image = cv2.resize(test_image, (32, 32))

            # updating data and labels list respectively
            test_data.append(test_image)
            test_labels.append(test_label)
        print(len(test_labels))
        print(len(test_data))

        # Appending the new dataset to training dataset
        print("[INFO] Appending new dataset to retrain the model...")
        for i in range(len(data)):
            train_data.append(data[i])
            train_labels.append(labels[i])

        print(len(train_labels))
        print(len(train_data))

        # convert my dataset into numpy array and preprocess by scaling pixel intensities to range [0, 1]

        # train dataset
        train_data = np.array(train_data, dtype="float") / 255.0

        # test dataset
        test_data = np.array(test_data, dtype="float") / 255.0

        # Encoding the train labels currently as strings, to integers and then one-hot encode them

        train_le = LabelEncoder()
        train_labels = train_le.fit_transform(train_labels)
        train_labels = np_utils.to_categorical(y=train_labels, num_classes=len(set(train_labels)))

        # Encoding the test labels currently as strings, to integers and then one-hot encode them

        test_le = LabelEncoder()
        test_labels = test_le.fit_transform(test_labels)
        test_labels = np_utils.to_categorical(y=test_labels, num_classes=len(set(test_labels)))

        # construct the training image generator for data augmentation
        # aug will be used further to generate images from our data

        aug = ImageDataGenerator(rotation_range=36, zoom_range=0.2, width_shift_range=0.25, height_shift_range=0.25,
                                 shear_range=0.2, horizontal_flip=True, fill_mode="nearest")

        # Initializing the optimizer and compiling modelX

        print("[INFO] Compiling Model for re-training...")
        opt = Adam(learning_rate=learn_rate, decay=learn_rate / epochs)
        model = Jagga.build(width=32, height=32, depth=3, classes=len(train_le.classes_))
        model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

        # training the network
        try:
            print("[INFO] Training networkX for : {} epochs...".format(epochs))
            res = model.fit_generator(aug.flow(train_data, train_labels, batch_size=batch_size),
                                      validation_data=(test_data, test_labels), steps_per_epoch=len(train_data) // batch_size,
                                      epochs=epochs)
        except ValueError:
            print("ValueError")

        print("[INFO] Evaluating Network...")
        predictions = model.predict(test_data, batch_size=batch_size)
        print(classification_report(test_labels.argmax(axis=1), predictions.argmax(axis=1), target_names=test_le.classes_))

        # Save the Retrained Model to Disk as it is more efficient

        print("[INFO] Serializing re-trained model to disk...")
        model.save("jagga.model")

        # Saving the Label Encoder to disk as well

        print("[INFO] Serializing Label Encoder of re-trained model to disk...")
        f = open("le.pickle", "wb")
        f.write(pickle.dumps(train_le))
        f.close()

        # lets plot the training loss and accuracy for re-trained model

        print("[INFO] Serializing the plotted graph of re-trained model to disk...")
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(np.arange(0, epochs), res.history["loss"], label="train_loss")
        plt.plot(np.arange(0, epochs), res.history["val_loss"], label="val_loss")
        plt.plot(np.arange(0, epochs), res.history["accuracy"], label="train_accuracy")
        plt.plot(np.arange(0, epochs), res.history["val_accuracy"], label="val_accuracy")
        plt.title("Training Loss and Accuracy on Dataset")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend(loc="lower left")
        plt.savefig("plotR.png")
        print("[INFO] Saved plot as plotR.png...")
        print("[INFO] Re-training Completed...")
