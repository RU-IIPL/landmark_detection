import os
import numpy as np
import glob

def encodeKeypointsScale(_data, _maxImageSize):
    data = np.array(_data).astype(np.float32)
    data = (data - _maxImageSize/2) / (_maxImageSize/2)
    return data

def decodeKeypointsScale(_data, _maxImageSize):
    data = np.array(_data).astype(np.float32)
    data = data * _maxImageSize/2 + _maxImageSize/2
    return data

def changeImageScale(_data, _imgSize, val = 255):
    data = np.array(_data)
    data = data / val
    data = data.reshape(-1, _imgSize[0], _imgSize[1], 1)
    return data

def saveMeanData(_data, _filename='mean.csv'):
    data = np.mean(_data, axis=0)
    np.savetxt(_filename, data, delimiter=',')

def loadMeanData(_filename):
    data = np.loadtxt(_filename, delimiter=',')
    return data

def getDataPath(_path):
    files = None
    if type(_path) == str:
        files = glob.glob(_path)
    elif type(_path) == list:
        files = _path
    return files
