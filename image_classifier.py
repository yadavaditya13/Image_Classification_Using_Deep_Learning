from keras.preprocessing.image import img_to_array
from keras.models import load_model
from model_retrain.model_train_script import JaggaRetrain
from imutils import paths

import numpy as np
import imutils
import tkinter
import pymongo
import random
import pickle
import cv2
import os

# print("hello1")
# mw = tkinter.Tk()

# creating connection to MongoDB and create a database
print("[INFO] Making connection with MongoDB server...")
my_client = pymongo.MongoClient("mongodb://127.0.0.1:27017/?compressors=disabled&gssapiServiceName=mongodb")
print("[INFO] Since no error(p.s.: ignore warnings)... Connection Successful...")
my_db = my_client["image_classification"]
print("[INFO] Creating Database Completed...")

# creating a collection for my database
my_coln = my_db["images_info"]
print("[INFO] Creating Collection Successful...")

# Initializing the learning rate, batch size and # of epoch to retrain

learn_rate = 1e-4
batch_size = 20
epochs = 75

data = []
data_labels = []

# This will be used to save the newly appended images in train images dir
count = 0


def image_run_inference():
    imagePaths = list(paths.list_images("D:/adity/Projects/image_classification/dataset/seg_pred"))
    # Loading my model and label encoder

    model = load_model("jagga.model")
    le = pickle.loads(open("le.pickle", "rb").read())

    # shuffling the images for randomization and selecting first 128 images for predictions

    random.shuffle(imagePaths)
    test_images = list(imagePaths[:128])

    images = []
    labels = []
    img_labels = []

    # Initializing my_list to store the images and labels in json-format
    my_list = []

    for test_image in test_images:
        # preprocessing the image one-by-one

        image = cv2.imread(test_image)
        images.append(image)
        image = cv2.resize(image, (32, 32))
        data.append(image)

        image = image.astype("float") / 255.0
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)

        # passing img through my model Jagga

        preds = model.predict(image)[0]
        index = np.argmax(preds)
        label = le.classes_[index]
        labels.append(label)

        img_label = "{}: {:.2f}%".format(label, preds[index] * 100)
        img_labels.append(img_label)

    global count
    # this loop is for displaying images and allowing user to change labels
    for i in range(len(images)):

        # Initializing my_dict and storing the images and labels in json-format
        my_dict = {}

        # Lets try to save a copy of image in train dir
        _path = "D:\\adity\Projects\image_classification\dataset\seg_train" + "\\" + labels[i] + "\\"
        _path = os.path.sep.join([_path, "{}.jpg".format(count)])

        cv2.imwrite(_path, images[i])
        print("[INFO] The images are saved in train dir...")

        images[i] = imutils.resize(images[i], width=500, height=500)
        cv2.imshow(img_labels[i], images[i])
        key = cv2.waitKey(0) & 0xFF

        if key == ord("r"):
            w = tkinter.Tk()
            w.title("Enter the correct Label: ")

            labelX = tkinter.Entry(w)
            labelX.pack()
            labelX.focus_set()

            def save():
                labels[i] = labelX.get()

                # appending updates values to my_dict
                my_dict[test_images[i]] = labels[i]
                print(labels[i])

            b = tkinter.Button(w, text='Submit', command=save)
            b.pack(side='bottom')
            w.mainloop()

        # appending values to database documents
        my_dict = {"image": test_images[i], "label": labels[i]}
        my_list.append(my_dict)
        data_labels.append(labels[i])

        if key == ord("c"):
            continue

        if key == ord("q"):
            break

    print("[INFO] Inserting values to Database...")
    x = my_coln.insert_many(my_list)
    print(x.inserted_ids)

    r = tkinter.Tk()

    retrain_button = tkinter.Button(r, text="Retrain", width=20, command=retrain_model)
    retrain_button.pack()

    r.mainloop()
    cv2.destroyAllWindows()


# we will be calling the retraining model script for retraining the model along with the new tested images
def retrain_model():
    n = tkinter.Tk()
    n.title("Enter the parameters: ")

    label1 = tkinter.Label(n, text="Learning Rate : ")
    label1.pack()

    lr = tkinter.Entry(n)
    lr.pack()
    lr.focus_set()
    # lr.focus_set()

    label2 = tkinter.Label(n, text="Batch Size : ")
    label2.pack()

    bs = tkinter.Entry(n)
    bs.pack()
    bs.focus_set()
    # bs.focus_set()

    label3 = tkinter.Label(n, text="Epoch # : ")
    label3.pack()

    epo = tkinter.Entry(n)
    epo.pack()
    epo.focus_set()

    # epo.focus_set()

    def submit_call():
        global learn_rate
        global batch_size
        global epochs

        learn_rate = float(lr.get())
        batch_size = int(float(bs.get()))
        epochs = int(float(epo.get()))

        JaggaRetrain.model_retrain(learn_rate=learn_rate, batch_size=batch_size, epochs=epochs, data=data,
                                   labels=data_labels)
        # n.destroy()

    submit_button = tkinter.Button(n, text="Submit", width=15, command=submit_call)
    submit_button.pack()

    n.mainloop()


# print("hello2")
flag = True

while True:
    # print("hello3")

    root = tkinter.Tk()
    root.title("Image_Classification")

    labelR = tkinter.Label(root, text="Run Inference : ")
    labelR.pack()

    image_button = tkinter.Button(root, text="OK", width=20, command=image_run_inference)
    image_button.pack()


    def close_window():
        global flag
        flag = False
        root.destroy()


    close_button = tkinter.Button(root, text="Close", width=20, command=close_window)
    close_button.pack()

    if not flag:
        break

    root.mainloop()

cv2.destroyAllWindows()
# mw.mainloop()
