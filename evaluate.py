# -*- coding: utf-8 -*-
"""
@author: Terada
"""

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import os
import random as rn
import shutil
import datetime
import csv
import glob
import pandas as pd
import re
from configparser import ConfigParser, ExtendedInterpolation

from keras.models import Sequential, model_from_json
from keras.optimizers import SGD, Adadelta, Adagrad, Adam, Adamax, RMSprop, Nadam
from keras.utils import np_utils
from keras import backend as K
import tensorflow as tf

import src.read_image as imread_mod
import src.read_text as txtread_mod
from src.data_generation import DatasetGeneration
import src.build_network as net

class Evaluate():
    def __init__(self, str1='Image', str2='land'):
        self.gd = DatasetGeneration(str1, str2)
        self.config = ConfigParser(interpolation=ExtendedInterpolation())
        self.flag_of_start = True

    def calcDiffLabel(self, _label_csv, _predict, _path):
        name = _path.split('\\')[len(_path.split('\\')) - 1]
        # get label data
        label_df = pd.read_csv(_label_csv, index_col=0)
        label_index = label_df.index
        label_column = label_df.columns
        # set column and index from label
        pred_df = pd.DataFrame(_predict, columns=label_column, index=label_index)
        # calc diff (abs)
        diff_df = abs(label_df - pred_df)
        diff_df.to_csv(self.result_path + "diff_{0}.csv".format(name))
        # calc cols mean
        #plt.figure()
        diff_df_mean = diff_df.mean()
        diff_df_mean.to_csv(self.result_path + "col_mean_{0}.csv".format(name))
        #diff_df_mean.plot.bar()
        #plt.savefig(self.result_path + "fig_cols_mean_{0}.png".format(name))
        # calc rows mean
        #plt.figure()
        diffT_df_mean = diff_df.T.mean()
        diffT_df_mean.to_csv(self.result_path + "row_mean_{0}.csv".format(name))
        #diffT_df_mean.plot.bar()
        #plt.savefig(self.result_path + "fig_rows_mean_{0}.png".format(name))

    def prediction_multi(self, _model, _X_test, _path):
        y_predict_all = _model.predict(_X_test)
        y_predict = y_predict_all[0]
        y1_predict = y_predict_all[1]
        y2_predict = y_predict_all[2]
        y3_predict = y_predict_all[3] # add
        half_size = self.input_size.max() / 2
        y_predict[0::2] = (y_predict[0::2] * half_size + half_size) * self.size_ratio[1] + 0.5
        y_predict[1::2] = (y_predict[1::2] * half_size + half_size) * self.size_ratio[0] + 0.5
        y_predict = y_predict.astype(np.int)
        dirname = _path.split('\\')[len(_path.split('\\')) - 1]
        with open(self.result_path + "predict_{0}.csv".format(dirname), 'w', newline='') as f:
            writer = csv.writer(f, lineterminator='\n')
            header = [landmark_dict[str(i)] for i in range(len(landmark_dict))]
            writer.writerow(header)
            for row in y_predict:
                writer.writerow(row)
        with open(self.result_path + "predict1_{0}.csv".format(dirname), 'w', newline='') as f:
            writer = csv.writer(f, lineterminator='\n')
            header = [genderlabel_dict[str(i)] for i in range(len(genderlabel_dict))]
            writer.writerow(header)
            for row in y1_predict:
                writer.writerow(row)
        with open(self.result_path + "predict2_{0}.csv".format(dirname), 'w', newline='') as f:
            writer = csv.writer(f, lineterminator='\n')
            header = [agelabel_dict[str(i)] for i in range(len(agelabel_dict))]
            writer.writerow(header)
            for row in y2_predict:
                writer.writerow(row)
        ## add d, x, y
        y3_predict[0::3] = (y3_predict[0::3] * half_size + half_size)
        y3_predict[1::3] = (y3_predict[1::3] * half_size + half_size) * self.size_ratio[0]
        y3_predict[2::3] = (y3_predict[2::3] * half_size + half_size) * self.size_ratio[1]
        with open(self.result_path + "predict3_{0}.csv".format(dirname), 'w', newline='') as f:
            writer = csv.writer(f, lineterminator='\n')
            header = [landmark3d_dict[str(i)] for i in range(len(landmark3d_dict))]
            writer.writerow(header)
            for row in y3_predict:
                writer.writerow(row)

        return y_predict

    def prediction(self, _model, _X_test, _path):
        y_predict = _model.predict(_X_test)
        half_size = self.input_size.max() / 2
        y_predict[0::2] = (y_predict[0::2] * half_size + half_size) * self.size_ratio[1] + 0.5
        y_predict[1::2] = (y_predict[1::2] * half_size + half_size) * self.size_ratio[0] + 0.5
        y_predict = y_predict.astype(np.int)
        dirname = _path.split('\\')[len(_path.split('\\')) - 1]
        with open(self.result_path + "predict_{0}.csv".format(dirname), 'w', newline='') as f:
            writer = csv.writer(f, lineterminator='\n')
            lm_label = [landmark_dict[str(i)] for i in range(len(landmark_dict))]
            writer.writerow(lm_label)
            for row in y_predict:
                writer.writerow(row)

        return y_predict

    def calcImageRatio(self):
        self.size_ratio = np.array([self.img_size[0] / self.input_size[0], self.img_size[1] / self.input_size[1]])

    def model_evaluation(self, _model, _X_test, _y_test, _path):
        score = _model.evaluate(_X_test, _y_test, batch_size=self.batch_size, verbose=1)
        dirname = _path.split('\\')[len(_path.split('\\'))-1]
        with open(self.save_score, 'a', newline='') as f:
            writer = csv.writer(f, lineterminator='\n')
            if self.flag_of_start:
                self.firstWrite = False
                writer.writerows([["",_model.metrics_names[0], _model.metrics_names[1]]])
            writer.writerow(list([dirname, score[0], score[1]]))

    def run(self):
        ## Read Data
        print("Read File ...")
        self.calcImageRatio()
        X_test, y_test = self.gd.load_data(self.val_image_path, self.val_label_path, self.save_labelset, self.img_size, self.input_size)
        y1_test, y2_test = self.gd.load_label() #self.result_path)
        y3_test = self.gd.load_3ddata(self.val_image_path, self.val_label3d_path, self.save_label3dset, self.img_size, self.input_size)

        models_dir = glob.glob(self.model_path)
        print(self.model_path)
        print(models_dir)
        for model_dir in models_dir:
            model_dirname = os.path.dirname(model_dir)

            ## Load Model
            print("Load Model : {0}".format(model_dirname))
            json_name = '{0}/{1}'.format(model_dirname, self.load_architecture)
            weights_name = '{0}/{1}'.format(model_dirname, self.load_weights_checkpoint) #
            if not os.path.exists(weights_name):
                weights_name = '{0}/{1}'.format(model_dirname, self.load_weights) #load_weights_checkpoint

            model = net.select(self.network, self.input_size, 28)
            #model = model_from_json(open(json_name).read())
            model.load_weights(weights_name)
            model.compile(optimizer=Adam(), loss='mean_squared_error', metrics=['accuracy'])

            ## Predict TestData
            print("Predict TestData ...")
            if not self.isMultiTask:
                self.model_evaluation(model, X_test, y_test, model_dirname)
                y_pre = self.prediction(model, X_test, model_dirname)
            else:
                #self.model_evaluation(model, X_test, [y_test, y1_test, y2_test], model_dirname)
                self.model_evaluation(model, X_test, [y_test, y1_test, y2_test, y3_test], model_dirname)
                y_pre = self.prediction_multi(model, X_test, model_dirname)

            ## Evaluate Model
            print("Evaluate Model ...")
            self.calcDiffLabel(self.save_labelset, y_pre, model_dirname)

    def setParameter(self, _conf_file):
        self.config.read(_conf_file)

        ## Image Data
        conf_io = self.config['IO']
        self.img_size = np.array([conf_io.getint('image_height'),conf_io.getint('image_width')])
        self.input_size = np.array([conf_io.getint('input_height'),conf_io.getint('input_width')])
        self.input_size = self.input_size.astype(np.int)

        ## Dataset
        conf_d = self.config['Dataset']
        testName = conf_d['base_test']
        testOther = conf_d['base_test_plus']
        self.val_image_path = '{0}//{1}{2}//*.jpg'.format(conf_d['image_path'], testName, testOther)
        self.val_label_path = '{0}//{1}{2}//*.txt'.format(conf_d['label_path'], testName, testOther)
        self.val_label3d_path = '{0}//{1}{2}//*.txt'.format(conf_d['label3d_path'], testName, testOther)
        comment=conf_d['comment']
        targetPath = conf_d['target_path']

        ## Result
        now = datetime.datetime.now()
        time = "{0:02d}{1:02d}{2:02d}{3:02d}".format(now.month, now.day, now.hour, now.minute)
        self.result_path = './result/{0}/{1}_{2}_{3}/'.format(targetPath, testName, comment, time)
        os.makedirs(self.result_path, exist_ok=True)

        ## Parameter
        conf_p = self.config['Parameter']
        self.batch_size = conf_p.getint('batch_size') #128
        self.network = conf_p['net']

        ## Malti-Task
        conf_m = self.config['MaltiTask']
        self.isMultiTask = conf_m.getboolean('multi_task_on')

        ## Filename
        conf_s = self.config['File']
        self.load_architecture = conf_s['architecture'] #'architecture.json'
        self.load_weights = conf_s['weights'] #'weights.h5'
        self.load_weights_checkpoint = conf_s['weights_checkpoint'] #'weights_checkpoint.h5'
        self.save_labelset = self.result_path + conf_s['labelset'] #'label.csv'
        self.save_label3dset = self.result_path + conf_s['label3dset'] 
        self.save_score = self.result_path + conf_s['score'] #'score.csv'
        self.model_path = conf_s['model_path']

landmark_dict = {'0':'left_eye_outer_corner_x',
                 '1':'left_eye_outer_corner_y',
                 '2':'left_eye_inner_corner_x',
                 '3':'left_eye_inner_corner_y',
                 '4':'right_eye_inner_corner_x',
                 '5':'right_eye_inner_corner_y',
                 '6': 'right_eye_outer_corner_x',
                 '7': 'right_eye_outer_corner_y',
                 '8': 'left_nose_top_x',
                 '9': 'left_nose_top_y',
                 '10': 'left_nose_bottom_x',
                 '11': 'left_nose_bottom_y',
                 '12': 'right_nose_top_x',
                 '13': 'right_nose_top_y',
                 '14': 'right_nose_bottom_x',
                 '15': 'right_nose_bottom_y',
                 '16': 'nose_root_x',
                 '17': 'nose_root_y',
                 '18': 'mouth_center_top_lip_x',
                 '19': 'mouth_center_top_lip_y',
                 '20': 'mouth_left_corner_x',
                 '21': 'mouth_left_corner_y',
                 '22': 'mouth_center_bottom_lip_x',
                 '23': 'mouth_center_bottom_lip_y',
                 '24': 'mouth_right_corner_x',
                 '25': 'mouth_right_corner_y',
                 '26': 'mouth_center_lip_x',
                 '27': 'mouth_center_lip_y'}

genderlabel_dict = {'0':'man',
                    '1':'woman'}

agelabel_dict = {'0':'20',
                 '1':'30',
                 '2':'40',
                 '3':'50',
                 '4':'60'}

landmark3d_dict = {'0':'left_eye_outer_corner_x',
                 '1':'left_eye_outer_corner_y',
                 '2':'left_eye_outer_corner_z',
                 '3':'left_eye_inner_corner_x',
                 '4':'left_eye_inner_corner_y',
                 '5':'left_eye_inner_corner_z',
                 '6':'right_eye_inner_corner_x',
                 '7':'right_eye_inner_corner_y',
                 '8':'right_eye_inner_corner_z',
                 '9': 'right_eye_outer_corner_x',
                 '10': 'right_eye_outer_corner_y',
                 '11': 'right_eye_outer_corner_z',
                 '12': 'left_nose_top_x',
                 '13': 'left_nose_top_y',
                 '14': 'left_nose_top_z',
                 '15': 'left_nose_bottom_x',
                 '16': 'left_nose_bottom_y',
                 '17': 'left_nose_bottom_z',
                 '18': 'right_nose_top_x',
                 '19': 'right_nose_top_y',
                 '20': 'right_nose_top_z',
                 '21': 'right_nose_bottom_x',
                 '22': 'right_nose_bottom_y',
                 '23': 'right_nose_bottom_z',
                 '24': 'nose_root_x',
                 '25': 'nose_root_y',
                 '26': 'nose_root_z',
                 '27': 'mouth_center_top_lip_x',
                 '28': 'mouth_center_top_lip_y',
                 '29': 'mouth_center_top_lip_z',
                 '30': 'mouth_left_corner_x',
                 '31': 'mouth_left_corner_y',
                 '32': 'mouth_left_corner_z',
                 '33': 'mouth_center_bottom_lip_x',
                 '34': 'mouth_center_bottom_lip_y',
                 '35': 'mouth_center_bottom_lip_z',
                 '36': 'mouth_right_corner_x',
                 '37': 'mouth_right_corner_y',
                 '38': 'mouth_right_corner_z',
                 '39': 'mouth_center_lip_x',
                 '40': 'mouth_center_lip_y',
                 '41': 'mouth_center_lip_z'}


if __name__ == "__main__":

    ev = Evaluate('Image', 'yland')
    config_files = glob.glob("./param/batch_ev/base/*.ini")
    for conf in config_files:
        ev.setParameter(conf)
        ev.run()

