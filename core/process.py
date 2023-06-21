import os

import cv2 as cv
import numpy as np

def pre_process(data_path,num,ext):
    file_name = os.path.split(data_path)[1].split('.')[0]
    img=cv.imdecode(np.fromfile(data_path,dtype=np.uint8),-1)
    if num == 0:
        img=AddSaltPepperNoise(img,0.5)

    cv.imencode(".png",img)[1].tofile(r'./tmp/draw/{}.{}'.format(file_name,ext))

    return data_path, file_name

def AddSaltPepperNoise(src, rate):
    srcCopy = src.copy()
    height, width = srcCopy.shape[0:2]
    noiseCount = int(rate*height*width/2)
    # add salt noise
    X = np.random.randint(width,size=(noiseCount,))
    Y = np.random.randint(height,size=(noiseCount,))
    srcCopy[Y, X] = 255
    # add black peper noise
    X = np.random.randint(width,size=(noiseCount,))
    Y = np.random.randint(height,size=(noiseCount,))
    srcCopy[Y, X] = 0
    return srcCopy