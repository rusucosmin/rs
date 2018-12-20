import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import os,sys
from PIL import Image
import re
# Helper functions

PIXEL_DEPTH = 255


def load_image(infilename):
    data = mpimg.imread(infilename)
    return data


def img_float_to_uint8(img):
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * 255).round().astype(np.uint8)
    return rimg


def concatenate_images(img, gt_img):
    '''
    Concatenate an image and its groundtruth
    input:  img - simple image
            gt_img - groundtruth segmentation for the image
    output: concatenation between the two images
    '''
    nChannels = len(gt_img.shape)
    w = gt_img.shape[0]
    h = gt_img.shape[1]
    if nChannels == 3:
        # Concatenate images
        cimg = np.concatenate((img, gt_img), axis=1)
    else:
        # Extend the groundtruth image to 3 channels
        gt_img_3c = np.zeros((w, h, 3), dtype=np.uint8)
        gt_img8 = img_float_to_uint8(gt_img)
        gt_img_3c[:,:,0] = gt_img8
        gt_img_3c[:,:,1] = gt_img8
        gt_img_3c[:,:,2] = gt_img8
        img8 = img_float_to_uint8(img)
        # Concatenate images
        cimg = np.concatenate((img8, gt_img_3c), axis=1)
    return cimg


def label_to_img(imgwidth, imgheight, w, h, labels):
    '''
    Convert array of labels to an image
    input:  imgwidth - image width
            imgheight - image height
            w - patch width
            h - patch height
            labels - labels associated with the patches
    output: image with segmentation labels
    '''
    array_labels = np.zeros([imgwidth, imgheight])
    idx = 0
    for i in range(0, imgheight, h):
        for j in range(0, imgwidth, w):
            array_labels[j:j+w, i:i+h] = labels[idx]
            idx = idx + 1
    return array_labels


def img_crop_v2(im, w, h, sw, sh):
    '''
    Crop image into patches
    input:  im - initial image
            w - patch width
            h - patch height
            sw - patch window horizontal step
            sh - patch window vertical step
    output: a list of image patches
    '''
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    # Crop image
    for i in range(0,imgheight-h+1,sh):
        for j in range(0,imgwidth-w+1,sw):
            if is_2d:
                im_patch = im[j:j+w, i:i+h]
            else:
                im_patch = im[j:j+w, i:i+h, :]
            list_patches.append(im_patch)
    return list_patches


def pad_image(image, padding):
    '''
    Pad image
    input:  image - input image
            padding - padding size
    output: padded image
    '''
    if (len(image.shape) < 3): # gt image
        data = np.pad(image, ((padding, padding), (padding, padding)), 'reflect')
    else:
        data = np.pad(image, ((padding, padding), (padding, padding), (0, 0)), 'reflect')
    return data


def make_img_overlay(img, predicted_img):
    '''
    Overlay image with its predicted segmentation
    input:  img - input image
            predicted_img - segmentation predition
    output: image consisting of the overlay of the input images
    '''
    w = img.shape[0]
    h = img.shape[1]
    # Prepare overlay image
    color_mask = np.zeros((w, h, 3), dtype=np.uint8)
    color_mask[:, :, 0] = predicted_img * PIXEL_DEPTH

    img8 = img_float_to_uint8(img)
    # Overlay images
    background = Image.fromarray(img8, 'RGB').convert("RGBA")
    overlay = Image.fromarray(color_mask, 'RGB').convert("RGBA")
    new_img = Image.blend(background, overlay, 0.2)
    return new_img
