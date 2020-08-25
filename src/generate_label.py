import os
import numpy as np
import glob
import csv
from collections import deque

import cv2


def getImage(_path, check = False):
    images = deque([])
    for i, filename in enumerate(_path):
        img = cv2.imread(filename, 0)
        #img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        if not img is None:
            if check:
                cv2.imshow("Draw Image", img)
                key = cv2.waitKey(1) & 0xff
                if key == ord('q'):
                    check = False
                    cv2.destroyAllWindows()
            images.append(img)  # .reshape(1, -1))
        else:
            print("read file error : {0}".format(filename))
    return np.array(images)#:,:,:,np.newaxis]

def getLabel(_name):
    names = deque([])
    pos = deque([])
    confidence = deque([])
    with open(_name, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        #print(header)
        for row in reader:
            names.append(row[0])
            pos.append(row[1:15])
            confidence.append(np.eye(2)[np.array(row[15:22]).astype(int)])

    return np.array(names), np.array(pos), np.array(confidence).reshape(-1, 14)


def mergeLabel(_path, _name = "label.csv"):
    ### read
    files = None
    if type(_path) == str:
        files = glob.glob(_path)
    elif type(_path) == list:
        files = _path

    ### input
    isGetHeader = True
    write_list = []
    for i, filename in enumerate(files):

        fp = open(filename)
        data = fp.read()
        fp.close()

        lines = data.split('\n')
        line_list = []
        header_list = []
        for line in lines:
            if len(line) > 1:
                line_list.append(line.split(','))
        if isGetHeader:
            isGetHeader = False
            header_list.append("filename")
            for j in range(len(line_list)):
                header_list.append(line_list[j][0]+"_x")
                header_list.append(line_list[j][0]+"_y")
            for j in range(len(line_list)):
                header_list.append(line_list[j][0]+"_c")
            write_list.append(header_list)

        pos_list = []
        confidence_list = []
        for j in range(len(line_list)):
            pos_list.append(line_list[j][1:3])
            confidence_list.append(line_list[j][3:4])

        name = [filename.replace("\\","//").replace(".csv",".jpg").replace("Text","Image")]
        #name = [os.path.basename(filename).replace(".csv",".jpg")]
        pos_list = np.array(pos_list).reshape(1, -1)
        confidence_list = np.array(confidence_list).reshape(1, -1)

        write_list.append(np.r_[name, pos_list[0], confidence_list[0]])


    ### output
    with open(_name, 'w') as f:
        for i in range(len(write_list)):
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(write_list[i])

    return write_list

if __name__ == "__main__":

    path = "//"
    file_ext = "*.csv"
    file_path = path + file_ext
    label_name = "label2.csv"

    mergeLabel(file_path, label_name)
    name, y, c  = getLabel(label_name)

    X = getImage(name)
    #print(X)
    #print(np.eye(2)[[0,1,1,0]])
