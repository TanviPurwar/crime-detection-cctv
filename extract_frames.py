import os, sys
from os import listdir #, fpathconf
import subprocess
import math
import glob

class ExtractFrames():

    def __init__(self, dataset_path, scale_factor, dest_dir, SEQUENCE_LENGTH):
        
        self.dataset_path = dataset_path
        self.scale_factor = scale_factor 
        self.dest_dir = dest_dir
        self.SEQUENCE_LENGTH = SEQUENCE_LENGTH

        self.extractFrames()

    def extractFrames(self):
       
        os.mkdir(self.dest_dir)

        for folder in listdir(self.dataset_path):

            frame_path = self.dest_dir + "\\" + folder
            os.mkdir(frame_path)

            folder_url = self.dataset_path + "\\" + folder
            print("Extracting frames from ", folder)

            for file in listdir(folder_url):
                file_url = folder_url + '\\' + file

                if os.path.isfile(file_url):
                    result = subprocess.run([
                        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=avg_frame_rate,r_frame_rate,nb_frames,width,height", file_url,],
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,)
                    ffprobe_out = str(result.stdout, 'utf-8')

                    nb_frames = int(ffprobe_out.split('\n')[5][10:])
                    interval = math.ceil(nb_frames/self.SEQUENCE_LENGTH)

                    if not os.path.isdir(frame_path + "\\" + file[:-4]):
                        os.mkdir(frame_path + "\\" + file[:-4])
                        os.system('ffmpeg -i "{}" -vf "select=not(mod(n\,{}))" -vsync vfr -q:v {} "{}/%03d.jpg"'.format(file_url, interval, self.scale_factor, 
                                                                                                                        frame_path + "\\" + file[:-4]))
