# -*- coding: utf-8 -*-
"""
@author: Terada
"""
import numpy as np
import glob
import cv2
import os
from natsort import natsorted

#X = X.reshape(-1, 96, 96, 1)

def extract_filename(filepath, ext = None):
    _name = filepath.split('\\')[len(filepath.split('\\'))-1]
    
    if ext == None:
        filename = _name
    else:
        filename = _name.split(ext)[0]
        
    return filename

'''
 画像群読み込み
'''       
def read_image(_data_path, _img_size): 
 
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
                print(path)

    filename_dict = {}
    image_list = []

    for i, filename in enumerate(files):
        ## make file ID : ファイルID生成
        #fname = extract_filename(filename, None) #.txt
        fname = os.path.basename(filename)
        filename_dict[i] = fname

        ## pre-processing
        img = cv2.imread(filename, 0)
        img = cv2.resize(img, (_img_size[1], _img_size[0])) # (w,h)
        
        image_list.append(img)
        #print(img.shape)

    return filename_dict, image_list

'''
 値スケールの変換
 0から1の値に変換 
'''
def image_change_scale(_img_list, _img_size, val = 255):
    np_img_list = np.array(_img_list)
    np_img_list = np_img_list / val 
    np_img_list = np_img_list.reshape(-1, _img_size[0], _img_size[1], 1)
    return np_img_list

'''
 メイン
'''
def main(filename_path, img_size):
    
    fname_dict = None
    ### ファイル名、画像の読み込み (リスト化)
    fname_dict, img_list = read_image(filename_path, img_size)
    print("Read Images : {0}".format(len(fname_dict)))

    ### リシェイプ    
    X = image_change_scale(img_list, img_size)    
    return X, fname_dict

if __name__ == "__main__":
    ### 画像サイズ ###
    img_size = np.array([400, 360])  #[h, w]
    img_size = img_size / 4
    img_size = img_size.astype(np.int)

    ### ファイル名、画像の読み込み (リスト化)
    path = 'data//sample_img//before_*//'
    filename_path = os.path.dirname(path) + "//" +"*.jpg"
    fname_dict, img_list = read_image(filename_path, img_size)
    print("Read Files: {0}".format(filename_path))

    ### リシェイプ
    X = image_change_scale(img_list, img_size)
    print("X.shape == {}; X.min == {:.1f}; X.max == {:.1f}".format(X.shape, X.min(), X.max()))
    
