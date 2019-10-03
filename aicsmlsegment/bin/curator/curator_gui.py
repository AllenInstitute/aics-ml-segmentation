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



def main():
    layout = [
        [sg.Text("User: ", key="userText", size=(20, 1))],
        [sg.Input('', key="username", size=(20, 1))],
        [sg.Text("Dataset CSV: ", key="csvText", size=(20, 1))],

        [sg.Input("/allen/aics/assay-dev/computational/data/mitotic_annotation_gui/pseudo_real_data/gui_sample_dataset.csv", key="csvPath", size=(40, 1)), 
         sg.FileBrowse(key='csvSelect', file_types=(('CSV Files','*.csv'),), initial_folder='/allen/aics/assay-dev/computational/data/'),
         sg.Text("", key="loadError", text_color='red', size=(60, 1))],

        [sg.Button("Load DSDB", key="b_load"), 
         sg.Text('Cell ID: ', key='fn', size=(9, 1)),
         sg.Text('', key='filename', size=(20, 1))],

        [sg.Graph(canvas_size=(100, 400),graph_bottom_left=(0, 400),graph_top_right=(400, 0),key="graph_dna_yz",change_submits=True,drag_submits=True),
        sg.Graph(canvas_size=(400, 400),graph_bottom_left=(0, 400),graph_top_right=(400, 0),key="graph_dna_xy",change_submits=True,drag_submits=True),
        sg.Graph(canvas_size=(400, 400),graph_bottom_left=(0, 400),graph_top_right=(400, 0),key="graph_mem_xy",change_submits=True,drag_submits=True),
        sg.Graph(canvas_size=(100, 400),graph_bottom_left=(0, 400),graph_top_right=(100, 0),key="graph_mem_yz",change_submits=True,drag_submits=True)],

        [sg.Graph(canvas_size=(100, 100),graph_bottom_left=(0, 100),graph_top_right=(100, 0),key="graph_dna_blank",change_submits=True,drag_submits=True),
        sg.Graph(canvas_size=(400, 100),graph_bottom_left=(0, 400),graph_top_right=(400, 0),key="graph_dna_xz",change_submits=True,drag_submits=True),
        sg.Graph(canvas_size=(400, 100),graph_bottom_left=(0, 400),graph_top_right=(400, 0),key="graph_mem_xz",change_submits=True,drag_submits=True),
        sg.Graph(canvas_size=(100, 100),graph_bottom_left=(0, 100),graph_top_right=(100, 0),key="graph_mem_blank",change_submits=True,drag_submits=True)],

        [sg.Frame(layout=[[sg.Checkbox('Enhance Contrast', key='enhCon', default=False, enable_events=True),
                            sg.Checkbox('Median Filter', key='medFilt',default=True, enable_events=True),
                            sg.Checkbox('Gaussian Filter', key='gFilt', default=False, enable_events=True),
                            sg.Checkbox(text='Show Segmentation', key='showSeg', default=False, enable_events=True)]],
                title='Image Options',
                relief=sg.RELIEF_SUNKEN),
        sg.Frame(layout=[[sg.Radio(text='Middle Slice', key='mid', group_id='projMethod', default=True, enable_events=True), 
                        sg.Radio(text='Max Projection', key='max', group_id='projMethod', enable_events=True), 
                        sg.Radio(text='Average Projection', key='ave', group_id='projMethod',enable_events=True)]],
                title='Projection Options',
                relief=sg.RELIEF_SUNKEN)],

        [sg.Text("", key="info", size=(60, 1))],
        [
            sg.Button('<<', key='bbf'),
            sg.Button("<", key="bb"),
            sg.Button(">", key="bf"),
            sg.Button(">>", key="bff"),
            sg.Text("", key="spacer", size=(5, 1)),
            sg.Button("0", key="b0"),
            sg.Button("1", key="b1"),
            sg.Button("2", key="b2"),
            sg.Button("3", key="b3"),
            sg.Button("4", key="b4"),
            sg.Frame(layout=[[sg.Checkbox(" UNSURE", key="unsure", default=False, enable_events=True)]], title='',relief=sg.RELIEF_SUNKEN),
            sg.Frame(layout=[[sg.Checkbox("Activate Hotkeys", key="hotkeysActive", default=False, enable_events=True, disabled=True)]], title='',relief=sg.RELIEF_SUNKEN),
            sg.Text("Current Image:", key="currImText", size=(15, 1)),
            sg.Text("", key="currIm", size=(10, 1)),
            sg.Button("3D", key="b3d"),
            sg.Button("save", key="bs")
        ]
    ]