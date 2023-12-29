import numpy as np
import cv2
import os
from PIL import Image

class dmb():
    def __init__(self,x,y,z ):
        pass
    def __call__(self,x):
        return x

files = os.listdir('./data2/')
files = sorted(files, key=lambda e: int(e.split('/')[-1].split('.')[0]))

for f in files:
    arr = np.array(Image.open(os.path.join('./data2',f)))
    h,w = arr.shape
    arr = arr[:,w//2:]
    if(np.mean(arr)<200):
        print(f)

