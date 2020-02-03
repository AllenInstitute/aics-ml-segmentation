#!/usr/bin/env python

import sys
import argparse
import logging
import traceback
import os
import pathlib
import numpy as np

from skimage.morphology import remove_small_objects
from skimage.io import imsave
from aicsimageio import AICSImage #, omeTifWriter
from aicsimageprocessing import resize

from aicsmlsegment.utils import load_config, load_single_image, input_normalization, image_normalization
from aicsmlsegment.utils import get_logger, simple_norm
from aicsmlsegment.model_utils import build_model, load_checkpoint, model_inference, apply_on_image, apply_on_full_image, model_inference_full_img
# Debugger
import pdb

# for post porcessing
from scipy.ndimage.morphology import binary_opening, binary_dilation, binary_erosion
from skimage.morphology import remove_small_objects, erosion, ball, dilation, remove_small_holes
from skimage.measure import label

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    config = load_config(args.config)

    # declare the model
    model = build_model(config)

    # load the trained model instance
    model_path = config['model_path']
    print(f'Loading model from {model_path}...')
    load_checkpoint(model_path, model)

    # extract the parameters for running the model inference
    args_inference=lambda:None
    args_inference.size_in = config['size_in']
    args_inference.size_out = config['size_out']
    args_inference.OutputCh = config['OutputCh']
    args_inference.nclass =  config['nclass'] 
    if config['RuntimeAug'] <=0:
        args_inference.RuntimeAug = False
    else:
        args_inference.RuntimeAug = True
    # run
    inf_config = config['mode']
    if inf_config['name'] == 'file':
        fn = inf_config['InputFile']
        data_reader = AICSImage(fn)
        img0 = data_reader.data

        if inf_config['timelapse']:
            assert img0.shape[0]>1

            for tt in range(img0.shape[0]):
                # Assume:  dimensions = TCZYX
                img = img0[tt, config['InputCh'],:,:,:].astype(float)
                img = image_normalization(img, config['Normalization'])

                if len(config['ResizeRatio'])>0:
                    img = resize(img, (1, config['ResizeRatio'][0], config['ResizeRatio'][1], config['ResizeRatio'][2]), method='cubic')
                    for ch_idx in range(img.shape[0]):
                        struct_img = img[ch_idx,:,:,:]
                        struct_img = (struct_img - struct_img.min())/(struct_img.max() - struct_img.min())
                        img[ch_idx,:,:,:] = struct_img

                # apply the model
                output_img = apply_on_image(model, img, model.final_activation, args_inference)

                # extract the result and write the output
                if len(config['OutputCh']) == 2:
                    out = output_img[0]
                    out = (out - out.min()) / (out.max()-out.min())
                    if len(config['ResizeRatio'])>0:
                        out = resize(out, (1.0, 1/config['ResizeRatio'][0], 1/config['ResizeRatio'][1], 1/config['ResizeRatio'][2]), method='cubic')
                    out = out.astype(np.float32)
                    if config['Threshold']>0:
                        out = out > config['Threshold']
                        out = out.astype(np.uint8)
                        out[out>0]=255
                    imsave(config['OutputDir'] + os.sep + pathlib.PurePosixPath(fn).stem + '_T_'+ f'{tt:03}' +'_struct_segmentation.tiff', out)
                else:
                    for ch_idx in range(len(config['OutputCh'])//2):
                        out = output_img[ch_idx]
                        out = (out - out.min()) / (out.max()-out.min())
                        if len(config['ResizeRatio'])>0:
                            out = resize(out, (1.0, 1/config['ResizeRatio'][0], 1/config['ResizeRatio'][1], 1/config['ResizeRatio'][2]), method='cubic')
                        out = out.astype(np.float32)
                        if config['Threshold']>0:
                            out = out > config['Threshold']
                            out = out.astype(np.uint8)
                            out[out>0]=255
                        imsave(config['OutputDir'] + os.sep + pathlib.PurePosixPath(fn).stem + '_T_'+ f'{tt:03}' +'_seg_'+ str(config['OutputCh'][2*ch_idx])+'.tiff',out)
        else:
            # If the model is 2D model
            if len(config['nclass']) == 1:
                if img0.shape[3] == 1: # when it is single channel
                    img = img0[0,0,0,0,:,:]
                else:
                    img = img0[0,0,0,config['InputCh'],:,:]
                img = simple_norm(img, 1, 6)
                # img = image_normalization(img, config['Normalization'])
                print(f'processing one image of size {img.shape}')

                output_img = model_inference(model, img, model.final_activation, args_inference)

                out = output_img[0] > config['Threshold']
                out = out.astype(np.uint8)
                out[out>0]=255
                imsave(config['OutputDir'] + os.sep + pathlib.PurePosixPath(fn).stem +'_struct_segmentation.tiff',out)
                print(f'Image {fn} has been segmented')
            else:
                img = img0[0,:,:,:,:].astype(float)
                print(f'processing one image of size {img.shape}')
                if img.shape[1] < img.shape[0]:
                    img = np.transpose(img,(1,0,2,3))
                img = img[config['InputCh'],:,:,:]
                img = image_normalization(img, config['Normalization'])

                if len(config['ResizeRatio'])>0:
                    img = resize(img, (1, config['ResizeRatio'][0], config['ResizeRatio'][1], config['ResizeRatio'][2]), method='cubic')
                    for ch_idx in range(img.shape[0]):
                        struct_img = img[ch_idx,:,:,:] # note that struct_img is only a view of img, so changes made on struct_img also affects img
                        struct_img = (struct_img - struct_img.min())/(struct_img.max() - struct_img.min())
                        img[ch_idx,:,:,:] = struct_img

                # apply the model
                output_img = apply_on_image(model, img, model.final_activation, args_inference)

                # extract the result and write the output
                if len(config['OutputCh']) == 2:
                    out = output_img[0] 
                    out = (out - out.min()) / (out.max()-out.min())
                    if len(config['ResizeRatio'])>0:
                        out = resize(out, (1.0, 1/config['ResizeRatio'][0], 1/config['ResizeRatio'][1], 1/config['ResizeRatio'][2]), method='cubic')
                    out = out.astype(np.float32)
                    if config['Threshold']>0:
                        out = out > config['Threshold']
                        out = out.astype(np.uint8)
                        out[out>0]=255
                    imsave(config['OutputDir'] + os.sep + pathlib.PurePosixPath(fn).stem +'_struct_segmentation.tiff', out)
                else:
                    for ch_idx in range(len(config['OutputCh'])//2):
                        out = output_img[ch_idx] 
                        out = (out - out.min()) / (out.max()-out.min())
                        if len(config['ResizeRatio'])>0:
                            out = resize(out, (1.0, 1/config['ResizeRatio'][0], 1/config['ResizeRatio'][1], 1/config['ResizeRatio'][2]), method='cubic')
                        out = out.astype(np.float32)
                        if config['Threshold']>0:
                            out = out > config['Threshold']
                            out = out.astype(np.uint8)
                            out[out>0]=255
                        imsave(config['OutputDir'] + os.sep + pathlib.PurePosixPath(fn).stem +'_seg_'+ str(config['OutputCh'][2*ch_idx])+'.tiff', out)
                print(f'Image {fn} has been segmented')

    elif inf_config['name'] == 'folder':
        from glob import glob
        filenames = glob(inf_config['InputDir'] + '/*' + inf_config['DataType'])
        filenames.sort()
        print('files to be processed:')
        print(filenames)

        for _, fn in enumerate(filenames):

            # load data
            data_reader = AICSImage(fn)
            img0 = data_reader.data

            # If the model is 2D model run this script. This is for temp
            if len(config['nclass']) == 1:
                if img0.shape[3] == 1: # when it is single channel
                    img = img0[0,0,0,0,:,:]
                else:
                    img = img0[0,0,0,config['InputCh'][0],:,:]
                # img = img0[0,0,0,config['InputCh'][0],:,:]
                img = simple_norm(img, 1, 6)
                print(f'processing one image of size {img.shape}')
                
                # for full size testing
                if config['mode']['apply_on_full_image']:
                    output_img = apply_on_full_image(model, img, model.final_activation, args_inference)

                else:
                    output_img = model_inference(model, img, model.final_activation, args_inference)

                out = output_img[:,:,1] > config['Threshold']
                out = out.astype(np.uint8)
                out[out>0]=255

                out = post_processing_2d(out).astype('int8')
                imsave(config['OutputDir'] + os.sep + pathlib.PurePosixPath(fn).stem +'_struct_segmentation.tiff',out)
                print(f'Image {fn} has been segmented')
                continue

            img = img0[0,:,:,:,:].astype(float)
            if img.shape[1] < img.shape[0]:
                img = np.transpose(img,(1,0,2,3))
            img = img[config['InputCh'],:,:,:]
            img = image_normalization(img, config['Normalization'])

            if len(config['ResizeRatio'])>0:
                img = resize(img, (1, config['ResizeRatio'][0], config['ResizeRatio'][1], config['ResizeRatio'][2]), method='cubic')
                for ch_idx in range(img.shape[0]):
                    struct_img = img[ch_idx,:,:,:] # note that struct_img is only a view of img, so changes made on struct_img also affects img
                    struct_img = (struct_img - struct_img.min())/(struct_img.max() - struct_img.min())
                    img[ch_idx,:,:,:] = struct_img

            # apply the model
            output_img = apply_on_image(model, img, model.final_activation, args_inference)

            # extract the result and write the output
            if len(config['OutputCh'])==2:
                if config['Threshold']<0:
                    out = output_img[0]
                    out = (out - out.min()) / (out.max()-out.min())
                    if len(config['ResizeRatio'])>0:
                        out = resize(out, (1.0, 1/config['ResizeRatio'][0], 1/config['ResizeRatio'][1], 1/config['ResizeRatio'][2]), method='cubic')
                    out = out.astype(np.float32)
                    out = (out - out.min()) / (out.max()-out.min())
                else:
                    out = remove_small_objects(output_img[0] > config['Threshold'], min_size=2, connectivity=1) 
                    out = out.astype(np.uint8)
                    out[out>0]=255
                imsave(config['OutputDir'] + os.sep + pathlib.PurePosixPath(fn).stem + '_struct_segmentation.tiff', out)
            else:
                for ch_idx in range(len(config['OutputCh'])//2):
                    if config['Threshold']<0:
                        out = output_img[ch_idx]
                        out = (out - out.min()) / (out.max()-out.min())
                        out = out.astype(np.float32)
                    else:
                        out = output_img[ch_idx] > config['Threshold']
                        out = out.astype(np.uint8)
                        out[out>0]=255
                    imsave(config['OutputDir'] + os.sep + pathlib.PurePosixPath(fn).stem + '_seg_'+ str(config['OutputCh'][2*ch_idx])+'.ome.tif', out)
            
            print(f'Image {fn} has been segmented')

def post_processing_2d(img):
    '''
    Post processing for 2D
    '''
    ball_mat_open = ball_matrix(5)
    ball_mat_hole = ball_matrix(5)
    ball_mat_dilation = ball_matrix(8)
    ball_mat_erosion = ball_matrix(5)

    processed_img = binary_opening(img, structure=ball_mat_open).astype(np.int)
    processed_img = binary_dilation(processed_img, structure=ball_matrix(2),iterations=2)

    object_mat = label(processed_img) # object detection

    # process per individual object
    final_seg = np.zeros(processed_img.shape)
    for individual_label in np.unique(object_mat):
        temp = (object_mat==individual_label)*1
        temp = binary_dilation(temp, structure=ball_mat_dilation)
        temp = binary_erosion(temp, structure=ball_mat_erosion)
        temp = remove_small_holes(temp, area_threshold=700)*1

        final_seg[temp == 1] = individual_label
    return final_seg

def ball_matrix(r):
    # making 2D ball shape matrix
    ball_mat = ball(r)
    return np.amax(ball_mat, axis = 0)

if __name__ == '__main__':
    main()