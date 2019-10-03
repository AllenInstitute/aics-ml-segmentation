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
import cv2