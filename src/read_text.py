# -*- coding: utf-8 -*-
"""
@author: Terada
"""

import numpy as np
import glob
import pandas as pd
from natsort import natsorted

## 顔器官ID
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

## 2D点クラス
class Point2D():
    def __init__(self, _x, _y):
        self.x = _x
        self.y = _y

class Point3D():
    def __init__(self, _x, _y, _z):
        self.x = _x
        self.y = _y
        self.z = _z

## 特徴点群
class LandmarkData():
    def __init__(self):
        self.idx = []
        self.Point = []

    def add_parts_index(self, _parts_idx):
        self.idx.append(_parts_idx)

    def add_Point(self, _Point):
        self.Point.append(_Point)

def extract_filename(filepath, ext = None):
    _name = filepath.split('\\')[len(filepath.split('\\'))-1]

    if ext == None:
        filename = _name
    else:
        filename = _name.split(ext)[0]

    return filename

'''
 テキストファイル読み込み
'''
def read_landmark(_data_path, _format = 1):

    files = None
    if type(_data_path) == str:
        files = glob.glob(_data_path)
        files = natsorted(files) ### add
    elif type(_data_path) == list:
        files = []
        for path in _data_path:
            try:
                files.append(glob.glob(path)[0])
            except:
                pass

    filename_dict = {}
    landmarks_list = []

    for i, filename in enumerate(files):
        ## make file ID : ファイルID生成
        fname = extract_filename(filename, None) #.txt
        filename_dict[i] = fname

        ## for Landmark
        fp = open(filename)
        data = fp.read()
        fp.close()

        if _format == 1:
            landmarks = LandmarkData()
            flag_of_start = False
            lines = data.split('\n')
            for line in lines:
                if "ID" in line:
                    flag_of_start = True
                elif flag_of_start and len(line) > 0:
                    #[No][ID][X][Y]
                    vals = line.split(',')
                    landmarks.add_parts_index(1) # start zero(0)
                    landmarks.add_Point( Point2D( float(vals[2]), float(vals[3]) ))
            landmarks_list.append(landmarks)
        elif _format == 2:
            landmarks = LandmarkData()
            flag_of_start = False
            lines = data.split('\n')
            for line in lines:
                if "Format" in line:
                    flag_of_start = True
                elif '#' in line and flag_of_start:
                    #[Area_Number][Index_Numer_in_Area][Index_Numer][X][Y]
                    vals = line.split(' ')
                    landmarks.add_parts_index(int(vals[4]) + 1) # start zero(0)
                    landmarks.add_Point( Point2D( int(vals[5]), int(vals[6]) ))
            landmarks_list.append(landmarks)
        elif _format == 3:
            landmarks = LandmarkData()
            lines = data.split('\n')
            for key_num, line in enumerate(lines):
                if line == '':
                    continue
                pr = line.split(',')
                pr = [float(k) for k in pr]
                landmarks.add_parts_index(key_num) # start zero(0)
                landmarks.add_Point( Point3D( float(pr[0]), float(pr[1]), float(pr[2])))
            landmarks_list.append(landmarks)
    return filename_dict, landmarks_list

'''
 lmlist[ff].Point[lmlist[oo].idx[yy]].(x/y)
 [ff] -> ファイルIDを指定する
 [oo] -> 顔器官IDを指定する
 (x/y) -> x か yで座標値を取得する
''' 
def make_landmark_df(_lm_list, _fname_dict, _landmark_dict):
    lmptlist = []
    for i in range(len(_lm_list)):
        lmpt = np.arange(0)
        for j in range(len(_lm_list[i].idx)):
            lmpt = np.append(lmpt, _lm_list[i].Point[j].x)
            lmpt = np.append(lmpt, _lm_list[i].Point[j].y)
        lmptlist.append(lmpt)
        
    ## ファイル名を列indexとして設定する
    fn_label = [_fname_dict[i] for i in range(len(_fname_dict))]

    ## 器官名を行indexとして設定する
    lm_label = [_landmark_dict[str(i)] for i in range(len(_landmark_dict))]
    df = pd.DataFrame(np.array(lmptlist), index = fn_label, columns = lm_label)

    return df

def make_3dlandmark_df(_lm_list, _fname_dict, _landmark_dict):
    lmptlist = []
    for i in range(len(_lm_list)):
        lmpt = np.arange(0)
        for j in range(len(_lm_list[i].idx)):
            lmpt = np.append(lmpt, _lm_list[i].Point[j].x)
            lmpt = np.append(lmpt, _lm_list[i].Point[j].y)
            lmpt = np.append(lmpt, _lm_list[i].Point[j].z)
        lmptlist.append(lmpt)

    ## ファイル名を列indexとして設定する
    fn_label = [_fname_dict[i] for i in range(len(_fname_dict))]

    ## 器官名を行indexとして設定する
    lm_label = [_landmark_dict[str(i)] for i in range(len(_landmark_dict))]
    #print("{}  {}  {}".format(np.array(lmptlist).shape, np.array(fn_label).shape, np.array(lm_label).shape))
    df = pd.DataFrame(np.array(lmptlist), index = fn_label, columns = lm_label)

    return df

'''
 出力     
'''
def write_landmark_df(_df_list, _save_fname):
    _df_list.to_csv(_save_fname)

'''
 -1から1の値に変換 
'''
def landmark_change_scale(_df_lm, _img_size):
    np_lmptlist = np.array(_df_lm).astype(np.float32)
    np_lmptlist = (np_lmptlist - _img_size/2) / (_img_size/2)
    return np_lmptlist


'''
 メイン
'''
def main(filename_path, io_fname, img_size, READ_CSV = False, READ_FORMAT = 1):
    
    fname_dict = None
    ### ラベル入力 ###
    if(READ_CSV):
        ### ラベルデータ読み込み
        lm_df = pd.read_csv(io_fname, index_col=0)
    else:
        ### ファイル名、特徴点の読み込み (リスト化)
        fname_dict, lm_list = read_landmark(filename_path, READ_FORMAT)
        print("Read Landmark Texts : {0}".format(len(fname_dict)))
        ### ラベル整形    
        if READ_FORMAT == 3:
            lm_df = make_3dlandmark_df(lm_list, fname_dict, landmark3d_dict)
        else:
            lm_df = make_landmark_df(lm_list, fname_dict, landmark_dict)

        ### ラベル出力 (CSV)
        save_fname = io_fname
        write_landmark_df(lm_df, save_fname)
        print("Write Landmark Dataset : OK")
        
    ### ラベル変換 
    y = landmark_change_scale(lm_df, img_size.max())

    return y, fname_dict

def convertNewFormat(fname_dict, lm_list, dst_dict = "./convert"):
    import csv
    import os
    os.makedirs(dst_dict, exist_ok=True)
    for i in range(len(lm_list)):
        with open(dst_dict + "//" + fname_dict[i], 'w', newline='') as f:
            writer = csv.writer(f, lineterminator='\n')
            lm_label = ["No.", "ID", "POS(X)", "POS(Y)"]
            writer.writerow(lm_label)
            for j in range(len(lm_list[i].idx)):
                row = [j + 1, 0, lm_list[i].Point[j].x, lm_list[i].Point[j].y]
                writer.writerow(row)

if __name__ == "__main__":
    ### ラベル入力 ###
    READ_CSV = False
    input_fname = 'data//label.csv'
    img_size = np.array([400, 360])  #[h, w]

    if(READ_CSV):
        ### ラベルデータ読み込み
        lm_df = pd.read_csv(input_fname, index_col=0)
        print("Read CSV: {0}".format(input_fname))
        
    else:
        text_format = -1 # before(-1)
        ### ファイル名、特徴点の読み込み (リスト化)
        filename_path = 'data//txt_after//after_tmp//*.txt'
        fname_dict, lm_list = read_landmark(filename_path, text_format)
        print("Read Files: {0}".format(filename_path))
        convertNewFormat(fname_dict, lm_list, 'data//txt_after//convert')

        ### ラベル整形    
        lm_df = make_landmark_df(lm_list, fname_dict, landmark_dict)

        ### ラベル出力 (CSV)
        #save_fname = input_fname
        #write_landmark_df(lm_df, save_fname)

    ### ラベル変換 
    y = landmark_change_scale(lm_df, img_size.max())
    print("y.shape == {}; y.min == {:.3f}; y.max == {:.3f}".format(y.shape, y.min(), y.max()))
    
