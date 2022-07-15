import cv2
from test_i3d import *
import pickle

with open('datasets.pkl', 'rb') as f:
    datasets = pickle.load(f)

mode = 'rgb'
num_classes = 2000
save_model = './checkpoints/'

# Change to where the videos are located
root = {'word': 'videos'}

train_split = 'preprocess/nslt_2000.json'

weights = './checkpoints/nslt_2000_065846_0.447803.pt'

run(mode=mode, root=root, train_split=train_split, weights=weights, datasets=datasets, num_classes=num_classes)
