#!/usr/bin/env python

import os
import sys
import logging
import argparse
import traceback
import importlib
import pathlib
import csv


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib
from glob import glob
from random import shuffle
from scipy import stats
from skimage.io import imsave
from skimage.draw import line, polygon

from aicssegmentation.core.utils import histogram_otsu
from aicsimageio import AICSImage, omeTifWriter
from aicsmlsegment.utils import input_normalization

import pickle
import numpy as np
import PySimpleGUI as sg
from datetime import datetime
from skimage  import io as skio
from skimage.transform  import resize
from scipy.ndimage.filters import median_filter, gaussian_filter
from PIL import Image
import base64
from io import BytesIO
import pandas




layout = [

    [sg.Graph(canvas_size=(500, 500),graph_bottom_left=(0, 500),graph_top_right=(500, 0),key="rawGraph",change_submits=True,drag_submits=False),
    sg.Graph(canvas_size=(500, 500),graph_bottom_left=(0, 500),graph_top_right=(500, 0),key="seg1Graph",change_submits=True,drag_submits=False),
    sg.Graph(canvas_size=(500, 500),graph_bottom_left=(0, 500),graph_top_right=(500, 0),key="seg2Graph",change_submits=True,drag_submits=False)],

    
    [sg.Frame(layout=[[sg.Button('Exclusion Mask', key='exMask', enable_events=True),
                       sg.Button('Merging Mask', key='mergeMask', enable_events=True)],
                      [sg.Button('Delete Last', key='delExMask', enable_events=True),
                       sg.Button('Delete Last', key='delMergeMask', enable_events=True)]],
            title='Create New Mask',
            relief=sg.RELIEF_SUNKEN),
    sg.Frame(layout=[[sg.Radio(text='Middle Slice', key='mid', group_id='projMethod', default=True, enable_events=True), 
                    sg.Radio(text='Max Projection', key='max', group_id='projMethod', enable_events=True), 
                    sg.Radio(text='Average Projection', key='ave', group_id='projMethod',enable_events=True)]],
            title='Projection Options',
            relief=sg.RELIEF_SUNKEN),
    sg.Checkbox('Enhance Contrast', key='enhCon', enable_events=True),
    sg.Button('Next Image', key='next', enable_events=True),
    sg.Button('Save', key='save', enable_events=True),
    sg.Exit()]
]

window = sg.Window("Curator", layout)



#################################
### Button Functions
#################################

def EnhanceContrast(img,target_shape=(100,100)):
    if window.Element('enhCon').Get():
        qinf, qsup = np.percentile(img, [5,95])
        img = np.clip(img,qinf,qsup)
        img = (255*(img-qinf)/(qsup-qinf))
    else:
        im_min = np.amin(img)
        im_max = np.amax(img-im_min)
        img = (img-im_min)/im_max*255

    return img





#################################
### GUI Display Functions
#################################

def displayImage(img,channel,dim,segChannel,target_shape,config,graph,scale=None):
    graph.Erase()

    img = projectStack(img,channel,dim)
    img = EnhanceContrast(img)

    if scale is None:
        ratios = np.divide(target_shape,np.shape(img))
        scale = np.amin(ratios)

    new_size = np.multiply(np.shape(img), scale).astype(np.uint64)
    new_size = (new_size[0], new_size[1])
    img = resize(img,new_size)

    img = numpy2bytes(img)
    graph.DrawImage(data=img, location=(0,0))
    return scale



def numpy2bytes(img):
    #must be used to display images without saving
    pil_img = Image.fromarray(img)
    buff = BytesIO()
    pil_img.save(buff, format="GIF")
    return base64.b64encode(buff.getvalue()).decode("utf-8")


def projectStack(img_full, channel, dim, isSeg=False):
    #outputs middle slice of stack along axis dim
    #method assumes ZCYX dimension arrangement
    img = img_full[channel,:,:,:]
    if dim==2:
        img = np.transpose(img, (1,0,2))
    elif dim==3:
        img = np.transpose(img, (2,1,0))

    if window.Element('mid').Get():
        numSlices = np.shape(img)[0]
        mid = np.floor(numSlices/2).astype(np.uint8)
        img = img[mid,:,:]
    elif window.Element('max').Get():
        img = np.max(img, axis=0)
    elif window.Element('ave').Get():
        if not isSeg:
            img = np.mean(img, axis=0)
        else:
            img = np.max(img, axis=0)

    return img




###################################
### Button Event Handling
###################################


while True:
    event, values = window.Read()