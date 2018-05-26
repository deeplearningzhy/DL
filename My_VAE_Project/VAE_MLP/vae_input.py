#coding=utf-8
import os
import numpy as np
import scipy.misc as io
from PIL import Image
#get the image_list&label_list
def read_images(new_dir):
    img=[]
    lab=[]
    for d in os.listdir(new_dir):
        newpath=new_dir+d
        img.append(newpath)
        lab.append(int(newpath.split('_')[-1].split('.')[0]))

    #list->array->Transpose
    temp=np.array( [img, lab] )#2D
    temp_T=temp.transpose()

    #shuffle
    np.random.shuffle(temp_T)
    image_list=list(temp_T[:, 0])
    label_list=list(temp_T[:, 1])

    return image_list,label_list

#——————————————————————————————————————————————————————————————
def GetFlatPix(filedir):

    image_list,label_list=read_images(filedir)

    #img->array
    imarrs=[]
    for im in image_list:
        imfig=Image.open(im)
        width, height = imfig.size
        imarray = np.array(imfig)
        imarray_flat=np.reshape(imarray,height*width)
        imarrs.append(imarray_flat)

    image_array=np.array(imarrs)
    image_array=image_array/255.0

    #lab->(n,)->one-hot
    label_array = np.array(label_list)
    y_vec = np.zeros((len(label_array), 15), dtype=np.float)
    for i, label in enumerate(label_array):
        y_vec[i, label_array[i]] = 1.0

    return image_array,y_vec