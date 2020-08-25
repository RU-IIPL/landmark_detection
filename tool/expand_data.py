# -*- coding: utf-8 -*-
"""
@author: Terada
"""

import numpy as np
import matplotlib.pyplot as plt

import os
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from collections import OrderedDict

import glob
import cv2
from collections import deque
import csv
import copy
from natsort import natsorted

## Translation x4
def doTranslation(_img):
    rows, cols = _img.shape
    x_value = [-cols / 10, cols / 10]
    y_value = [-rows / 10, rows / 10]
    img_list = deque([])
    for xv in x_value:
        for yv in y_value:
            M = np.float32([[1, 0, xv], [0, 1, yv]])
            dst = cv2.warpAffine(_img, M, (cols, rows))
            img_list.append(dst)
    return img_list

## Scale x8
def doScale(_img):
    rows, cols = _img.shape
    x_value = [1.1, 1.0, 0.9]
    y_value = [1.1, 1.0, 0.9]
    img_list = deque([])
    for xv in x_value:
        for yv in y_value:
            if xv == 1.0 and yv == 1.0:
                continue
            M = np.float32([[xv, 0, 0], [0, yv, 0]])
            dst = cv2.warpAffine(_img, M, (cols, rows))
            img_list.append(dst)
    return img_list

## Scale -> Flip x8
def doScale_yFlip(_img):
    rows, cols = _img.shape
    x_value = [1.1, 1.0, 0.9]
    y_value = [1.1, 1.0, 0.9]
    img_list = deque([])
    for xv in x_value:
        for yv in y_value:
            if xv == 1.0 and yv == 1.0:
                continue
            M = np.float32([[xv, 0, 0], [0, yv, 0]])
            dst = cv2.warpAffine(_img, M, (cols, rows))
            dst_yflip = cv2.flip(dst, 1)
            img_list.append(dst_yflip)
    return img_list

def getFullPathFiles(_dir, _ext = "*"):
    files = None
    if type(_dir) == str:
        # get fullpath files
        files = glob.glob(_dir + "//" + _ext)
        files = natsorted(files)
    elif type(_dir) == list:
        files = _dir
    return files

def writeImageList(_list, _dir, _filename, _name):
    # get filename
    fname = os.path.basename(_filename).split('.')[0]
    # get root directory
    rootDir = os.path.dirname(os.path.abspath(_dir))
    # set directory name
    dir = rootDir + '/' + os.path.basename(_dir) + _name
    # create directory
    os.makedirs(dir, exist_ok=True)
    # write image
    for ino, img in enumerate(_list):
        cv2.imwrite("{0}/{1}{2}_{3}.jpg".format(dir, fname, _name, ino), img)

def generateImages(_srcPath, _dstPath = None, _isDraw = False, _isWrite = False):
    if _dstPath == None:
        _dstPath = _srcPath
    files = getFullPathFiles(_srcPath, _ext="*.jpg")

    for i, filename in enumerate(files):
        # read image
        img = cv2.imread(filename, 0)

        # pre-processing
        img_yflip = cv2.flip(img, 1) #x:0, y:1, xy:-1
        img_trans_list = doTranslation(img)
        img_scale_list = doScale(img)
        img_scale_ylip_list = doScale_yFlip(img)

        # draw
        if _isDraw:
            cv2.imshow("Image", img)
            cv2.imshow("yFlip", img_yflip)
            for ino, pimg in enumerate(img_trans_list):
                cv2.imshow("Translation {0}".format(ino), pimg)
            for ino, pimg in enumerate(img_scale_list):
                cv2.imshow("Scale {0}".format(ino), pimg)
            for ino, pimg in enumerate(img_scale_ylip_list):
                cv2.imshow("Scale Flip {0}".format(ino), pimg)
            cv2.waitKey(0)

        # write
        if _isWrite:
            writeImageList(list([img_yflip]), _dstPath, filename, '_yflip')
            writeImageList(img_trans_list, _dstPath, filename, '_trans')
            writeImageList(img_scale_list, _dstPath, filename, '_scale')
            writeImageList(img_scale_ylip_list, _dstPath, filename, '_scale_yflip')

###---------------------------------------------

def readText(filename):
    data = deque([])
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            data.append(row)
    return header, data

### x, y, z ->  d, x, y
def read3DText(filename):
    data = deque([])
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            data.append(row)
    return data

## Flip
def doTextFlip(_data, cols, rows):
    _list_data = copy.deepcopy(list(_data))
    for i in range(len(_list_data)):
        i_flip = yflipID[i]
        _list_data[i_flip][2] = str(cols - int(_list_data[i][2]))
        _list_data[i_flip][3] = str(int(_list_data[i][3]))
    return _list_data

### x, y, z ->  d, x, y
def do3DTextFlip(_data, cols, rows):
    _list_data = copy.deepcopy(list(_data))
    for i in range(len(_list_data)):
        i_flip = yflipID[i]
        _list_data[i_flip][1] = str(cols - float(_list_data[i][1]))
        _list_data[i_flip][2] = str(float(_list_data[i][2]))
    return _list_data

## Translation
def doTextTranslation(_data, cols, rows):
    _list_data = copy.deepcopy(list(_data))
    x_value = [-cols / 10, cols / 10]
    y_value = [-rows / 10, rows / 10]
    txt_list = deque([])

    for xv in x_value:
        for yv in y_value:
            dst = copy.deepcopy(_list_data) #
            for i in range(len(_list_data)):
                dst[i][2] = str(float(_list_data[i][2]) + float(xv))
                dst[i][3] = str(float(_list_data[i][3]) + float(yv))
            txt_list.append(dst)
    return txt_list

### x, y, z ->  d, x, y
def do3DTextTranslation(_data, cols, rows):
    _list_data = copy.deepcopy(list(_data))
    x_value = [-cols / 10, cols / 10]
    y_value = [-rows / 10, rows / 10]
    txt_list = deque([])

    for xv in x_value:
        for yv in y_value:
            dst = copy.deepcopy(_list_data) #
            for i in range(len(_list_data)):
                dst[i][1] = str(float(_list_data[i][1]) + float(xv))
                dst[i][2] = str(float(_list_data[i][2]) + float(yv))
            txt_list.append(dst)
    return txt_list

## Scale
def doTextScale(_data, cols, rows):
    _list_data = copy.deepcopy(list(_data))
    x_value = [1.1, 1.0, 0.9]
    y_value = [1.1, 1.0, 0.9]
    txt_list = deque([])

    for xv in x_value:
        for yv in y_value:
            if xv == 1.0 and yv == 1.0:
                continue
            dst = copy.deepcopy(_list_data) #
            for i in range(len(_list_data)):
                dst[i][2] = str(int(float(_list_data[i][2]) * xv))
                dst[i][3] = str(int(float(_list_data[i][3]) * yv))
            txt_list.append(dst)
    return txt_list

### x, y, z ->  d, x, y
def do3DTextScale(_data, cols, rows):
    _list_data = copy.deepcopy(list(_data))
    x_value = [1.1, 1.0, 0.9]
    y_value = [1.1, 1.0, 0.9]
    txt_list = deque([])

    for xv in x_value:
        for yv in y_value:
            if xv == 1.0 and yv == 1.0:
                continue
            dst = copy.deepcopy(_list_data) #
            for i in range(len(_list_data)):
                dst[i][1] = str(float(float(_list_data[i][1]) * xv))
                dst[i][2] = str(float(float(_list_data[i][2]) * yv))
            txt_list.append(dst)
    return txt_list

## Scale -> Flip x8
def doTextScale_yFlip(_data, cols, rows):
    _list_data = copy.deepcopy(list(_data))
    x_value = [1.1, 1.0, 0.9]
    y_value = [1.1, 1.0, 0.9]
    txt_list = deque([])

    for xv in x_value:
        for yv in y_value:
            if xv == 1.0 and yv == 1.0:
                continue
            dst = copy.deepcopy(_list_data) #
            for i in range(len(_list_data)):
                i_flip = yflipID[i]
                dst[i_flip][2] = str(cols - float(float(_list_data[i][2]) * xv))
                dst[i_flip][3] = str(float(float(_list_data[i][3]) * yv))
            txt_list.append(dst)
    return txt_list

### x, y, z ->  d, x, y
def do3DTextScale_yFlip(_data, cols, rows):
    _list_data = copy.deepcopy(list(_data))
    x_value = [1.1, 1.0, 0.9]
    y_value = [1.1, 1.0, 0.9]
    txt_list = deque([])

    for xv in x_value:
        for yv in y_value:
            if xv == 1.0 and yv == 1.0:
                continue
            dst = copy.deepcopy(_list_data) #
            for i in range(len(_list_data)):
                i_flip = yflipID[i]
                dst[i_flip][1] = str(cols - float(float(_list_data[i][1]) * xv))
                dst[i_flip][2] = str(float(float(_list_data[i][2]) * yv))
            txt_list.append(dst)
    return txt_list

def writeTextList(header, _list, _dir, _filename, _name):
    # get filename
    fname = os.path.basename(_filename).split('.')[0]
    # get root directory
    rootDir = os.path.dirname(os.path.abspath(_dir))
    # set directory name
    dir = rootDir + '/' + os.path.basename(_dir) + _name
    # create directory
    os.makedirs(dir, exist_ok=True)

    for ino, txt in enumerate(_list):
        with open("{0}/{1}{2}_{3}.txt".format(dir, fname, _name, ino), 'w') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(header)
            writer.writerows(txt)

### 
def write3DTextList(_list, _dir, _filename, _name):
    # get filename
    fname = os.path.basename(_filename).split('.')[0]
    # get root directory
    rootDir = os.path.dirname(os.path.abspath(_dir))
    # set directory name
    dir = rootDir + '/' + os.path.basename(_dir) + _name
    # create directory
    os.makedirs(dir, exist_ok=True)

    for ino, txt in enumerate(_list):
        with open("{0}/{1}{2}_{3}.txt".format(dir, fname, _name, ino), 'w') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerows(txt)

def generateTexts(_srcPath, _dstPath = None, _isDraw = False, _isWrite = False):
    if _dstPath == None:
        _dstPath = _srcPath
    files = getFullPathFiles(_srcPath, _ext="*.txt")

    for i, filename in enumerate(files):
        # read text
        header, data = readText(filename)
        width = 360
        height = 400

        # pre-processing
        txt_yflip = doTextFlip(data, width, height)
        txt_trans_list = doTextTranslation(data, width, height)
        txt_scale_list = doTextScale(data, width, height)
        txt_scale_yflip_list = doTextScale_yFlip(data, width, height)

        writeTextList(header, list([txt_yflip]), _dstPath, filename, "_yflip")
        writeTextList(header, txt_trans_list, _dstPath, filename, "_trans")
        writeTextList(header, txt_scale_list, _dstPath, filename, "_scale")
        writeTextList(header, txt_scale_yflip_list, _dstPath, filename, "_scale_yflip")

def generateTexts3D(_srcPath, _dstPath = None, _isDraw = False, _isWrite = False):
    if _dstPath == None:
        _dstPath = _srcPath
    files = getFullPathFiles(_srcPath, _ext="*.txt")

    for i, filename in enumerate(files):
        # read text
        data = read3DText(filename)
        width = 360
        height = 400

        # pre-processing
        txt_yflip = do3DTextFlip(data, width, height)
        txt_trans_list = do3DTextTranslation(data, width, height)
        txt_scale_list = do3DTextScale(data, width, height)
        txt_scale_yflip_list = do3DTextScale_yFlip(data, width, height)

        write3DTextList(list([txt_yflip]), _dstPath, filename, "_yflip")
        write3DTextList(txt_trans_list, _dstPath, filename, "_trans")
        write3DTextList(txt_scale_list, _dstPath, filename, "_scale")
        write3DTextList(txt_scale_yflip_list, _dstPath, filename, "_scale_yflip")


# parts swap
yflipID = {0:3, 1:2, 2:1, 3:0, 4:6, 5:7, 6:4, 7:5, 8:8, 9:9, 10:12, 11:11, 12:10, 13:13}


# usage: python .\tool\expand_data.py
if __name__ == "__main__":

    datasetNames = ["set1", "set2", "set3"]

    for datasetName in datasetNames:
        src_path = 'data/images/edge/{0}'.format(datasetName)
        generateImages(_srcPath=src_path, _isDraw=False, _isWrite=True)

        src_path = 'data/images/before/{0}'.format(datasetName)
        #generateImages(_srcPath=src_path, _isDraw=False, _isWrite=True)

        src_path = 'data/images/after/{0}'.format(datasetName)
        #generateImages(_srcPath=src_path, _isDraw=False, _isWrite=True)

        src_path = 'data/labels/before/{0}'.format(datasetName)
        #generateTexts(_srcPath=src_path, _isDraw=False, _isWrite=True)

        src_path = 'data/labels/after/{0}'.format(datasetName)
        #generateTexts(_srcPath=src_path, _isDraw=False, _isWrite=True)

        src_path = 'data/labels3d/{0}'.format(datasetName)
        #generateTexts3D(_srcPath=src_path, _isDraw=False, _isWrite=True)
