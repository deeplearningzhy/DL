#coding=utf-8
from PIL import Image
import tensorflow as tf
import random
import numpy as np
import os
def ImageToMatrix(filename):
    im = Image.open(filename)
    # shape=im.shape()
    # print(shape)no attribute shape
    width, height = im.size
    im = im.convert("L")
    data = im.getdata()
    data = np.matrix(data, dtype="float") / 255.0#原图
    data_flatten = np.reshape(data, (height * width))  #flatten (1,10000)
    # new_data2 = np.reshape(new_data1, (height,width))#从flatten还原
    return data_flatten

def MatrixToImage(data):
    data = data * 255
    new_im = Image.fromarray(data.astype(np.uint8))
    return new_im

def ImageToArray(filedir):
    im=Image.open(filedir)
    width, height = im.size
    imarray=np.array(im)
    imarray_flat=np.reshape(imarray,height*width)
    # imarray_recover=np.reshape(imarray_flat,(height,width))
    return imarray_flat

def GetFlatPix(filedir):
    picarr=[]
    for img_i in os.listdir(filedir):
        pathname=filedir+img_i
        data_flatten=ImageToMatrix(pathname)#matrix 1D
        arr_flatten=np.array(data_flatten)
        picarr.append(arr_flatten)
    return picarr

def GetFlatPix2(filedir):
    picarr=[]
    for img_i in os.listdir(filedir):
        pathname=filedir+img_i
        data_flatten=ImageToArray(pathname)#array 1D
        picarr.append(data_flatten)
    picarr_toarray=np.array(picarr)
    return picarr_toarray


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    ###img-matrix-array############
    fanzhuan_dir = "D:\\deep_keyan\\homework\\2018_3_28\\img_test\\"
    arr_picarr=GetFlatPix2(fanzhuan_dir)
    a1=np.reshape(arr_picarr[0,:],(100,100))
    new_im = Image.fromarray(a1.astype(np.uint8))
    new_im.show()
    # plt.show()
    print(np.shape(arr_picarr))
    print(arr_picarr)

    ################################
    # ####img-array############
    # fanzhuan_dir = "D:\\deep_keyan\\homework\\2018_3_28\\img_test\\"
    # list_picarr = GetFlatPix2(fanzhuan_dir)
    # arr_picarr = np.array(list_picarr)
    # print(np.shape(arr_picarr))
    # print(list_picarr)
    # print(arr_picarr)
    #
    # #######
    # ####show()####
    # fanzhuan_dir = "D:\\deep_keyan\\homework\\2018_3_28\\img_test\\a001_0_10_0_12.jpg"
    # flat_matrix, re_matrix = ImageToMatrix(fanzhuan_dir)
    # flat_im = MatrixToImage(flat_matrix)
    # re_im = MatrixToImage(re_matrix)
    # flat_im.show()
    # flat_im.save('2.jpg')







    # for img_file in os.listdir(fanzhuan_dir):
    #     name = fanzhuan_dir + img_file
    #     new_name = name.split('.')[0]
    #     data, data1, data2 = ImageToMatrix(name)
    #     f0 = MatrixToImage(data)
    #     f1 = MatrixToImage(data1)
    #     f2 = MatrixToImage(data2)
    #     # f3=f2.resize((100,100,1))#TypeError: argument 1 must be sequence of length 2, not 3
    #     # new_im1=new_im.resize((100,100))#error
    #     f1.save(str(new_name) + "0.png")
    #     f2.save(str(new_name) + "1.png")

