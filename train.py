import os
import numpy as np
from extract_features import ExtractFeatures
from train_generator import TrainGenerator
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from keras.callbacks import ModelCheckpoint

class Train():

    def __init__(self, SEQUENCE_LENGTH, IMAGE_SIZE, model_name, train_path, batch_size, nb_classes):

        self.IMAGE_SIZE = IMAGE_SIZE
        self.SEQUENCE_LENGTH = SEQUENCE_LENGTH
        self.batch_size = batch_size
        self.model_name = model_name
        self.nb_classes = nb_classes

        self.train_path = train_path
        self.saved_model_path = ""

        self.x_train, self.y_train = self.check_features()
        self.model_nb = self.check_model_path()

        self.train_generator = TrainGenerator(self.x_train, self.y_train, self.batch_size)
        
        print((self.train_generator[0])[0].shape)
        print((self.train_generator[0])[1].shape)
        
        self.lstm()

    def check_features(self):
        features_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + "\\Extracted Features\\" + model_name

        if os.path.isfile(features_path + "\\x_train.npy"):
            x_train = np.load(features_path + "\\x_train.npy")
            y_train = np.load(features_path + "\\y_train.npy")

            return x_train, y_train
        
        else:
            ef = ExtractFeatures(self.train_path, self.IMAGE_SIZE, self.SEQUENCE_LENGTH, self.model_name, self.nb_classes, mode='train')
            
            x_train = np.load(features_path + "\\x_train.npy")
            y_train = np.load(features_path + "\\y_train.npy")
            
            return x_train, y_train
    
    def check_model_path(self):
        path = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + "\\Saved Models" #+ self.model_name
        if not os.path.isdir(path):
            os.mkdir(path)
        path = path + "\\" + self.model_name
        if not os.path.isdir(path):
            os.mkdir(path)
        
        self.saved_model_path = path

        return len(os.listdir(path)) + 1
    
    def lstm(self):

        lstm_model = Sequential()
        lstm_model.add(Bidirectional(LSTM(158, return_sequences=False, activation='tanh', dropout=0.5), input_shape=(self.SEQUENCE_LENGTH, self.x_train.shape[2])))
        lstm_model.add(Dense(128)) #, activation='relu'
        lstm_model.add(LeakyReLU(alpha=0.05))
        lstm_model.add(Dropout(0.5))
        lstm_model.add(Dense(64)) #, activation='relu'
        lstm_model.add(LeakyReLU(alpha=0.03))
        lstm_model.add(Dropout(0.5))
        lstm_model.add(Dense(self.nb_classes, activation='softmax'))

        optimizer = Adam(learning_rate=1e-4, decay=1e-7) #, decay=1e-6
        lstm_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"])

        path = self.saved_model_path + "\\m_" + str(self.model_nb) + ".h5"
        checkpoint = ModelCheckpoint(path, monitor="loss", mode="min", save_best_only=True, verbose=1)
        
        lstm_model.fit(self.train_generator, epochs = 40, verbose = 2, callbacks=[checkpoint])

if __name__ == '__main__':
    SEQUENCE_LENGTH = 100
    IMAGE_SIZE = 128
    batch_size = 4
    nb_classes = 5

    train_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + "\\Data\\Train"
    
    print("1: Xception\n2: VGG16\n3: VGG19\n4: InceptionV3\n5: DenseNet121\n6: DenseNet201\n7: ResNet50\n8: ResNet152\n")
    n = int(input("Enter the number corresponding to the model for training: "))
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
    
    t = Train(SEQUENCE_LENGTH, IMAGE_SIZE, model_name, train_path, batch_size, nb_classes)