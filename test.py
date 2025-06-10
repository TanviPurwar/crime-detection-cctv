import os
import numpy as np
from extract_features import ExtractFeatures
from train_generator import TrainGenerator
from test_generator import TestGenerator
import tensorflow as tf
from sklearn.metrics import confusion_matrix

class Test():
    def __init__(self, SEQUENCE_LENGTH, IMAGE_SIZE, model_name, model_path, test_path, batch_size, nb_classes):
        
        self.IMAGE_SIZE = IMAGE_SIZE
        self.SEQUENCE_LENGTH = SEQUENCE_LENGTH
        self.batch_size = batch_size
        self.model_name = model_name
        self.model_path = model_path
        self.nb_classes = nb_classes

        self.test_path = test_path

        self.x_test, self.y_test = self.check_features()

        self.test_generator = TrainGenerator(self.x_test, self.y_test, self.batch_size)
        self.pred_generator = TestGenerator(self.x_test, self.batch_size)
        
        print((self.test_generator[0])[0].shape)
        print((self.test_generator[0])[1].shape)
        
        self.validate_model()

    def check_features(self):
            features_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + "\\Extracted Features\\" + model_name

            if os.path.isfile(features_path + "\\x_test.npy"):
                x_train = np.load(features_path + "\\x_test.npy")
                y_train = np.load(features_path + "\\y_test.npy")

                return x_train, y_train
            
            else:
                ef = ExtractFeatures(self.test_path, self.IMAGE_SIZE, self.SEQUENCE_LENGTH, self.model_name, self.nb_classes, mode='test')

                x_train = np.load(features_path + "\\x_test.npy")
                y_train = np.load(features_path + "\\y_test.npy")

                return x_train, y_train

    def validate_model(self):
        print(os.path.isfile(self.model_path))
        lstm_model = tf.keras.models.load_model(self.model_path)
        
        loss, acc = lstm_model.evaluate(self.test_generator, verbose=1)
        
        print("test accuracy: ", acc)
        print("test loss: ", loss)
        
        ypred = lstm_model.predict(self.pred_generator)
        
        cm=confusion_matrix(self.y_test.argmax(axis=1), ypred.argmax(axis=1))
        print(cm)
    

if __name__ == '__main__':
    SEQUENCE_LENGTH = 100
    IMAGE_SIZE = 128
    batch_size = 5
    nb_classes = 5

    test_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + "\\Data\\Test" 

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
    
    #manually put in model name
    model_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + "\\Saved Models\\" + model_name + "\\m.h5"
    
    t = Test(SEQUENCE_LENGTH, IMAGE_SIZE, model_name, model_path, test_path, batch_size, nb_classes)