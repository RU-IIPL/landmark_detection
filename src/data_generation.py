# -*- coding: utf-8 -*-
"""
@author: Terada
"""

import numpy as np
import os
import random as rn
import re
from keras.utils import np_utils

import src.generate_label as gl
import src.common as cmn
import src.read_image as imread_mod
import src.read_text as txtread_mod

class DatasetGeneration():
    def __init__(self, str1='Image', str2='land', str3='R'):
        self.filename1_str = str1
        self.filename2_str = str2
        self.filename3_str = str3
        self.filename = None

    def shuffleDataset(self, data1, data2):
        # shuffle
        l = list(zip(data1, data2))
        np.random.shuffle(l)
        data1, data2 = zip(*l)
        data1 = np.array(data1)
        data2 = np.array(data2)
        return data1, data2

    def common_filename_check(self, _fname1, _fname2, _fname1_str='Image', _fname2_str='land', _fname1_ext='jpg', _fname2_ext='txt'):

        _1to2name = [re.sub(_fname1_str, _fname2_str, _fname1[i].split(_fname1_ext)[0]) for i in range(len(_fname1))]
        _com_name = []

        for i in range(len(_1to2name)):
            for j in range(len(_fname2)):
                if _1to2name[i] in _fname2[j]:
                    _com_name.append(_1to2name[i].split(_fname2_str)[1])
                    break

        iext = np.array([_fname1_ext] * len(_com_name))
        _com_fname1 = np.core.defchararray.add(_com_name, iext)
        _com_fname1 = [re.sub('^', _fname1_str, _com_fname1[i]) for i in range(len(_com_fname1))]

        text = np.array([_fname2_ext] * len(_com_name))
        _com_fname2 = np.core.defchararray.add(_com_name, text)
        _com_fname2 = [re.sub('^', _fname2_str, _com_fname2[i]) for i in range(len(_com_fname2))]
        return _com_fname1, _com_fname2

    def load_data(self, _image_path, _label_path, _io_fname, _img_size, _input_size, data_check=True):
        X, iname = imread_mod.main(_image_path, _input_size) # read image
        y, tname = txtread_mod.main(_label_path, _io_fname, _img_size, False) # read text

        if data_check:
            iname, tname = self.common_filename_check(iname, tname, self.filename1_str, self.filename2_str)
            print("Common Files : {0}".format(len(iname)))
            _iname_path = []
            _tname_path = []
            for i in range(len(iname)):
                _iname_path.append(os.path.dirname(_image_path) + "//" + iname[i])
                _tname_path.append(os.path.dirname(_label_path) + "//" + tname[i])
            X, iname = imread_mod.main(_iname_path, _input_size)
            y, tname = txtread_mod.main(_tname_path, _io_fname, _img_size, False)

        self.filename = tname
        return X, y

    ## add
    def load_3ddata(self, _image_path, _label_path, _io_fname, _img_size, _input_size, data_check=True):
        X, iname = imread_mod.main(_image_path, _input_size) # read image
        y, tname = txtread_mod.main(_label_path, _io_fname, _img_size, False, READ_FORMAT=3) # read text

        if data_check:
            iname, tname = self.common_filename_check(iname, tname, self.filename1_str, 'yland')
            print("Common Files : {0}".format(len(iname)))
            _iname_path = []
            _tname_path = []
            for i in range(len(iname)):
                _iname_path.append(os.path.dirname(_image_path) + "//" + iname[i])
                _tname_path.append(os.path.dirname(_label_path) + "//" + tname[i])
            X, iname = imread_mod.main(_iname_path, _input_size)
            y, tname = txtread_mod.main(_tname_path, _io_fname, _img_size, False, READ_FORMAT=3)

        #self.filename = tname
        return y


    def load_label(self):

        fp = open("data/attribute.csv")
        data = fp.read()
        fp.close()
        gen_dict = {}
        age_dict = {}
        lines = data.split('\n')
        for i, line in enumerate(lines):
            if i == 0:  # header
                continue
            if line == "":
                break
            vals = line.split(',')
            id = int(vals[0].split('R')[1])
            gen_dict[id] = int(vals[1]) - 1
            age_dict[id] = int(vals[2]) - 2

        gen_list = []
        age_list = []
        pattern = '([+-]?[0-9]+\.?[0-9]*)'
        for i in range(len(self.filename)):
            num = re.findall(pattern, os.path.basename(self.filename[i]))[0]
            num = int(num.split('.')[0])
            gen_list.append(gen_dict[num])
            age_list.append(age_dict[num])
        gen_list.insert(0, 1)  # max value
        age_list.insert(0, 4)  # max value
        one_hot_y1 = np_utils.to_categorical(gen_list)
        one_hot_y2 = np_utils.to_categorical(age_list)
        one_hot_y1 = np.delete(one_hot_y1, 0, 0)  # delete max value
        one_hot_y2 = np.delete(one_hot_y2, 0, 0)  # delete max value

        return  one_hot_y1, one_hot_y2

    def load_data2_(self, _path, _input_size, _output):
        # set
        file_path = _path
        label_name = _output + "label.csv"

        # label function
        if not os.path.isfile(label_name):
            gl.mergeLabel(file_path, label_name)
        name, pos2d, confidence = gl.getLabel(label_name)

        # shuffle
        l = list(zip(name, pos2d, confidence))
        np.random.shuffle(l)
        name, pos2d, confidence = zip(*l)
        name = np.array(name)
        pos2d = np.array(pos2d)
        confidence = np.array(confidence)

        # image function
        images = gl.getImage(name, _input_size)
        cmn.saveMeanData(images, _output + 'mean.csv')

        return images, pos2d, confidence
