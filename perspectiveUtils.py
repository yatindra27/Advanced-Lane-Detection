# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 22:03:52 2018

@author: Yatindra Vaishnav
"""

import numpy as np
import cv2


# Edit this function to create your own pipeline.
def binarize_pipeline(img, s_thresh=(170, 255), sx_thresh=(20, 100), debug=False):
    img = np.copy(img)
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    if (debug==True):
        cv2.imshow('l_channel', l_channel)
        cv2.imshow('s_channel', s_channel)

    g_channel = img[:,:,1]
    r_channel = img[:,:,2]

    if (debug==True):
        cv2.imshow('r_channel', r_channel)
        cv2.imshow('g_channel', g_channel)

    # Sobel x
    sobelx = cv2.Sobel(r_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel_r = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    if (debug==True):
        cv2.imshow('scaled_sobel_r', scaled_sobel_r)

    sobelx = cv2.Sobel(g_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel_g = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    if (debug==True):
        cv2.imshow('scaled_sobel_g', scaled_sobel_g)

    sobelx = cv2.Sobel(s_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel_s = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    if (debug==True):
        cv2.imshow('scaled_sobel_s', scaled_sobel_s)

    sobely = cv2.Sobel(g_channel, cv2.CV_64F, 0, 1) # Take the derivative in x
    abs_sobely = np.absolute(sobely) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel_y_g = np.uint8(255*abs_sobely/np.max(abs_sobely))
    if (debug==True):
        cv2.imshow('scaled_sobel_y_g', scaled_sobel_y_g)

    sobely = cv2.Sobel(s_channel, cv2.CV_64F, 0, 1) # Take the derivative in x
    abs_sobely = np.absolute(sobely) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel_y_s = np.uint8(255*abs_sobely/np.max(abs_sobely))
    if (debug==True):
        cv2.imshow('scaled_sobel_y_s', scaled_sobel_y_s)

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel_s)
    sxbinary[(scaled_sobel_r >= sx_thresh[0]) & (scaled_sobel_r <= sx_thresh[1])] = 1
    sxbinary[(scaled_sobel_g >= sx_thresh[0]) & (scaled_sobel_g <= sx_thresh[1])] = 1
    sxbinary[(scaled_sobel_s >= sx_thresh[0]) & (scaled_sobel_s <= sx_thresh[1])] = 1

    # Threshold y gradient
    sxbinary[(scaled_sobel_y_g >= sx_thresh[0]) & (scaled_sobel_y_g <= sx_thresh[1])] = 1
    sxbinary[(scaled_sobel_y_s >= sx_thresh[0]) & (scaled_sobel_y_s <= sx_thresh[1])] = 1
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)             
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    s_binary[(g_channel >= s_thresh[0]) & (g_channel <= s_thresh[1])] = 1
    s_binary[(l_channel >= s_thresh[0]) & (l_channel <= s_thresh[1])] = 1

    # Stack each channel
    final_bin=np.zeros_like(s_channel)
    final_bin[(s_binary==1) | (sxbinary==1)] = 1
    
    final_bin = final_bin * 255
    return final_bin

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    img = np.uint8(img)
    if len(img.shape) is 2:
        img = np.dstack((img, np.zeros_like(img), np.zeros_like(img)))
    return cv2.addWeighted(initial_img, α, img, β, γ)



def getPerspective2BirdsView(image, debug = False):
    y, x = image.shape[:2]

#    mv = y/2
#    mh = x/2
#    src = np.float32([[mh-110, mv+90],
#                      [mh+110, mv+90],
#                      [100, y-50],
#                      [x-100, y-50]])
#    dst = np.float32([[0,0], [x, 0], [0,y], [x,y]])
    mid_ver = y/2
    mid_hor = x/2
    mid_ver_off = 90
    bott_ver_off = 50
    mid_hor_off = 110
    bott_hor_off = 100
    src = np.float32([[mid_hor-mid_hor_off, mid_ver+mid_ver_off],
                      [mid_hor+mid_hor_off, mid_ver+mid_ver_off],
                      [bott_hor_off, y-bott_ver_off],
                      [x-bott_hor_off, y-bott_ver_off]])
    dst = np.float32([[0,0], [x, 0], [0,y], [x,y]])

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    bvi = cv2.warpPerspective(image, M, (x, y))
    if (debug==True):
        print(M)
        print(Minv)        
        cv2.imshow('bvi', bvi)

    return  bvi, M, Minv

def perspective_pipeline(bin_img, debug=False):
        
    bvi,M, Minv=getPerspective2BirdsView(bin_img, debug)
    if (debug==True):
        cv2.imshow('bvi', bvi)
    
    return bvi, M, Minv    
