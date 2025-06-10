from extract_frames import ExtractFrames
from train_test_split import TrainTestSplit
import sys, os

def prepareDataset():

    parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    os.mkdir(parent_dir + '\\Data')
    dataset_path = 'C:\\Users\\Tanvi\\Desktop\\UCF Crime Dataset\\Entire Dataset'
    dest_dir = os.path.join(os.path.join(parent_dir, 'Data'), 'UCF Crime Frames')
    
    scale_factor = 2
    SEQUENCE_LENGTH = 100
    split_factor = 0.2

    ExtractFrames(dataset_path, scale_factor, dest_dir, SEQUENCE_LENGTH)
    TrainTestSplit(dest_dir, split_factor)


if __name__ == '__main__':
    
    prepareDataset()