import os
import sys
import cv2
import numpy as np
import pandas as pd
from keras.models import load_model
from os import listdir #, fpathconf
import subprocess
import math
import glob
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

video_file = 'C:\\Users\\Tanvi\\Desktop\\UCF Crime Dataset\\Entire Dataset\\Arson\\Arson040_x264.mp4'
output_path = 'C:\\Users\\Tanvi\\Desktop\\Frames'

if os.path.isfile(video_file):
                    result = subprocess.run([
                        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=avg_frame_rate,r_frame_rate,nb_frames,width,height", video_file,],
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,)
                    ffprobe_out = str(result.stdout, 'utf-8')

                    nb_frames = int(ffprobe_out.split('\n')[5][10:])
                    interval = math.ceil(nb_frames/100)

                    if not os.path.isdir(output_path):
                        os.mkdir(output_path)
                        os.system('ffmpeg -loglevel quiet -i "{}" -vf "select=not(mod(n\,{}))" -vsync vfr -q:v {} "{}/%03d.jpg"'.format(video_file, interval, 0.2, 
                                                                                                                        output_path))

from tensorflow.keras.applications import DenseNet121

base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
output = base_model.output
output = Flatten(name="flatten")(output)
transfer_model = Model(inputs=base_model.input, outputs=output)
for layer in transfer_model.layers:
    layer.trainable = False
    
    
x = []

# TO ADD EXTRA FRAMES
zero_img = np.zeros([128, 128, 3], np.float32)
zero_img = np.expand_dims(zero_img, axis=0)
features0 = transfer_model.predict(zero_img)
features0 = features0[0]

for frame in listdir(output_path):
    print(frame)
    
    img = load_img(frame, target_size=(128, 128))
    im = img_to_array(img)
    blur = cv2.GaussianBlur(im, (3, 3), 0)
    imb = np.expand_dims(blur, axis=0)
    
    from tensorflow.keras.applications.densenet import preprocess_input
    
    imb = preprocess_input(imb)

    features = self.model.predict(imb)
    features = features[0]
    x.append(features)
    
nb_missing_frames = self.SEQUENCE_LENGTH - len(listdir(video_frames_path))
for i in range(nb_missing_frames):
    x.append(features0)

lstm_model = tf.keras.models.load_model('C:\\Minor Project\\Project Code\\Saved Models\\DenseNet121\\m.h5')

ypred = lstm_model.predict(x)
print(ypred)

