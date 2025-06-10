import os
from random import shuffle
from os import listdir
from tabnanny import verbose
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from sklearn.utils import shuffle

class ExtractFeatures():

    def __init__(self, path, IMAGE_SIZE, SEQUENCE_LENGTH, model_name, nb_classes, mode):

        self.path = path
        self.IMAGE_SIZE = IMAGE_SIZE
        self.SEQUENCE_LENGTH = SEQUENCE_LENGTH
        self.model_name = model_name
        self.mode = mode
        self.nb_classes = nb_classes

        if self.model_name == 'Xception':
            self.model = self.xception()

        elif self.model_name == 'VGG16':
            self.model = self.vgg16()

        elif self.model_name == 'VGG19':
            self.model = self.vgg19()
            
        elif self.model_name == 'InceptionV3':
            self.model = self.inceptionv3()

        elif self.model_name == 'DenseNet121':
            self.model = self.densenet121()

        elif self.model_name == 'DenseNet201':
            self.model = self.densenet201()

        elif self.model_name == 'ResNet50':
            self.model = self.resnet50()

        elif self.model_name == 'ResNet152':
            self.model = self.resnet152()
        
        else:
            print("Error: Invalid Entry: Enter a valid number between 1-8")
            return

        self.extract()


    def xception(self):
        from tensorflow.keras.applications import Xception
        
        base_model = Xception(weights='imagenet', include_top=False, input_shape=(self.IMAGE_SIZE, self.IMAGE_SIZE, 3))

        output = base_model.output
        output = Flatten(name="flatten")(output)

        transfer_model = Model(inputs=base_model.input, outputs=output)

        for layer in transfer_model.layers:
            layer.trainable = False

        transfer_model.summary()

        return transfer_model

    def vgg16(self):

        from tensorflow.keras.applications import VGG16

        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(self.IMAGE_SIZE, self.IMAGE_SIZE, 3))

        output = base_model.output
        output = Flatten(name="flatten")(output)

        transfer_model = Model(inputs=base_model.input, outputs=output)

        for layer in transfer_model.layers:
            layer.trainable = False

        transfer_model.summary()

        return transfer_model

    def vgg19(self):
        from tensorflow.keras.applications import VGG19

        base_model = VGG19(weights='imagenet', include_top=False, input_shape=(self.IMAGE_SIZE, self.IMAGE_SIZE, 3))

        output = base_model.output
        output = Flatten(name="flatten")(output)

        transfer_model = Model(inputs=base_model.input, outputs=output)

        for layer in transfer_model.layers:
            layer.trainable = False

        transfer_model.summary()

        return transfer_model

    def inceptionv3(self):
        from tensorflow.keras.applications import InceptionV3

        base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(self.IMAGE_SIZE, self.IMAGE_SIZE, 3))

        output = base_model.output
        output = Flatten(name="flatten")(output)

        transfer_model = Model(inputs=base_model.input, outputs=output)

        for layer in transfer_model.layers:
            layer.trainable = False

        transfer_model.summary()

        return transfer_model

    def densenet121(self):
        from tensorflow.keras.applications import DenseNet121

        base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(self.IMAGE_SIZE, self.IMAGE_SIZE, 3))

        output = base_model.output
        output = Flatten(name="flatten")(output)

        transfer_model = Model(inputs=base_model.input, outputs=output)

        for layer in transfer_model.layers:
            layer.trainable = False

        transfer_model.summary()

        return transfer_model
    
    def densenet201(self):
        from tensorflow.keras.applications import DenseNet201
        from tensorflow.keras.applications.densenet import preprocess_input

        base_model = DenseNet201(weights='imagenet', include_top=False, input_shape=(self.IMAGE_SIZE, self.IMAGE_SIZE, 3))

        output = base_model.output
        output = Flatten(name="flatten")(output)

        transfer_model = Model(inputs=base_model.input, outputs=output)

        for layer in transfer_model.layers:
            layer.trainable = False

        transfer_model.summary()

        return transfer_model
    
    def resnet50(self):
        from tensorflow.keras.applications import ResNet50
        
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(self.IMAGE_SIZE, self.IMAGE_SIZE, 3))

        output = base_model.output
        output = Flatten(name="flatten")(output)

        transfer_model = Model(inputs=base_model.input, outputs=output)

        for layer in transfer_model.layers:
            layer.trainable = False

        transfer_model.summary()

        return transfer_model

    def resnet152(self):
        from tensorflow.keras.applications import ResNet152
        
        base_model = ResNet152(weights='imagenet', include_top=False, input_shape=(self.IMAGE_SIZE, self.IMAGE_SIZE, 3))

        output = base_model.output
        output = Flatten(name="flatten")(output)

        transfer_model = Model(inputs=base_model.input, outputs=output)

        for layer in transfer_model.layers:
            layer.trainable = False

        transfer_model.summary()

        return transfer_model
    
    def extract(self):
        x = []
        y = []

        # TO ADD EXTRA FRAMES
        zero_img = np.zeros([self.IMAGE_SIZE, self.IMAGE_SIZE, 3], np.float32)
        zero_img = np.expand_dims(zero_img, axis=0)
        features0 = self.model.predict(zero_img)
        features0 = features0[0]
        
        for folder in listdir(self.path):
            label = folder
            class_path = self.path + '\\' + folder

            for video_path in listdir(class_path):
                video_frames_path = class_path + '\\' + video_path

                y.append(label)

                features_sequence = []
                for frame in listdir(video_frames_path):
                    frame_path = video_frames_path + '\\' + frame

                    img = load_img(frame_path, target_size=(self.IMAGE_SIZE, self.IMAGE_SIZE))
                    im = img_to_array(img)
                    blur = cv2.GaussianBlur(im, (3, 3), 0)
                    imb = np.expand_dims(blur, axis=0)

                    if self.model_name == 'Xception':
                        from tensorflow.keras.applications.xception import preprocess_input
                    elif self.model_name == 'VGG16':
                        from tensorflow.keras.applications.vgg16 import preprocess_input
                    elif self.model_name == 'VGG19':
                        from tensorflow.keras.applications.vgg19 import preprocess_input
                    elif self.model_name == 'InceptionV3':
                        from tensorflow.keras.applications.inception_v3 import preprocess_input
                    elif self.model_name == 'DenseNet121' or self.model_name == 'DenseNet201':
                        from tensorflow.keras.applications.densenet import preprocess_input
                    elif self.model_name == 'ResNet50' or self.model_name == 'ResNet152':
                        from tensorflow.keras.applications.resnet import preprocess_input

                    imb = preprocess_input(imb)

                    features = self.model.predict(imb)
                    features = features[0]
                    features_sequence.append(features)

                nb_missing_frames = self.SEQUENCE_LENGTH - len(listdir(video_frames_path))
                
                for i in range(nb_missing_frames):
                    features_sequence.append(features0)
                
                x.append(features_sequence)

        le = LabelEncoder()
        y = le.fit_transform(y)

        x = np.array(x)
        y = np.array(y)

        y = to_categorical(y, self.nb_classes)

        X, Y = shuffle(x, y, random_state=42)
        
        print(X.shape)
        print(Y.shape) 
        
        parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
        if not os.path.isdir(parent_dir + "\\Extracted Features"):
            os.mkdir(parent_dir + "\\Extracted Features")
        dest_path = parent_dir + "\\Extracted Features\\" + self.model_name
        if not os.path.isdir(dest_path):
            os.mkdir(dest_path)
        
        np.save(dest_path + "\\x_" + self.mode + ".npy", X)
        np.save(dest_path + "\\y_" + self.mode + ".npy", Y)

if __name__ == '__main__':
    
    SEQUENCE_LENGTH = 100
    IMAGE_SIZE = 128
    nb_classes = 5

    path = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + "\\Data\\Train" 

    print("1: Xception\n2: VGG16\n3: VGG19\n4: InceptionV3\n5: DenseNet121\n6: DenseNet201\n7: ResNet50\n8: ResNet152\n")
    n = int(input("Enter the number corresponding to the model required for feature extraction: "))
    model_name = ""

    if n == 1:
        model_name = "Xception"
    elif n == 2:
        model_name = "VGG16"
    elif n == 3:
        model_name = "VGG19"
    elif n == 4:
        model_name = "InceptionV3"
    elif n == 5:
        model_name = "DenseNet121"
    elif n == 6:
        model_name = "DenseNet201"
    elif n == 7:
        model_name = "ResNet50"
    elif n == 8:
        model_name = "ResNet152"


    ef = ExtractFeatures(path, IMAGE_SIZE, SEQUENCE_LENGTH, model_name, nb_classes, mode='train')
    #arr1 = np.load("C:\\Minor Project\\Project Code\\Extracted Features\\"+ model_name +"\\x_train.npy")
    #print(arr1.shape)
    #arr2 = np.load("C:\\Minor Project\\Project Code\\Extracted Features\\Xception\\y_train.npy")
    #print(arr2.shape)