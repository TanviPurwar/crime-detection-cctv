import numpy as np
import keras
import math
#from tensorflow.keras.utils import Sequence

class TestGenerator(keras.utils.Sequence):
    
    def __init__(self, x, batch_size):
        self.x = x
        self.batch_size = batch_size
        self.num_batches = math.ceil(len(self.x) // batch_size)
        #print(self.batch_idx[0])
    
    def __len__(self):
        return (self.num_batches)
    
    def __getitem__(self, idx):
        x_batch = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        return x_batch