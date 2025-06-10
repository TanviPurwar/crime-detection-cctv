import sys, os
import shutil
from sklearn.model_selection import train_test_split

class TrainTestSplit():
    
    def __init__(self, path, split_factor):

        self.path = path
        self.split_factor = split_factor

        self.train_path = os.path.abspath(os.path.join(self.path, os.pardir)) + "\\Train"
        self.test_path = os.path.abspath(os.path.join(self.path, os.pardir)) + "\\Test"

        os.mkdir(self.train_path)
        os.mkdir(self.test_path) 
        os.mkdir(self.val_path) 

        self.split_data()
    
    def split_data(self):   

        file_paths = []
        labels = []

        for folder in os.listdir(self.path):
            folder_files = self.path + '\\' + folder

            for file in os.listdir(folder_files):
                file_paths.append(folder_files + "\\" + file)
                labels.append(folder)
    
        #x_train, x_test, y_train, y_test = train_test_split(file_paths, labels, test_size = self.split_factor, shuffle = True, random_state = 42, stratify = labels)
        x_train, x_test, y_train, y_test = train_test_split(file_paths, labels, test_size = self.split_factor, shuffle = True, random_state = 42, stratify = labels)
        
        for index in range(len(x_train)):
            shutil.copytree(x_train[index], self.train_path + "\\" + y_train[index] + "\\" + x_train[index].split('\\')[-1:][0])
        
        for index in range(len(x_test)):
            shutil.copytree(x_test[index], self.test_path + "\\" + y_test[index] + "\\" + x_test[index].split('\\')[-1:][0])
        

        shutil.rmtree(self.path)
