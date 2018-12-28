# -*- coding: utf-8 -*-
"""
Created on Sat Dec 22 14:29:28 2018

@author: Yatindra Vaishnav
"""
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2

def get_histogram_pixel_line_base(binary_warped, debug=False):
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    if debug == True:
        print(leftx_base)
        print(rightx_base)
    return leftx_base, rightx_base

def find_lane_pixels_sliding_window(binary_warped, leftx_base, rightx_base, margin, minpix, nwindows=9):
    window_height = np.int(binary_warped.shape[0]//nwindows)
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    leftx_current = leftx_base
    rightx_current = rightx_base

    left_lane_inds = []
    right_lane_inds = []

    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass    

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    return leftx, lefty, rightx, righty

def find_lane_pixels(binary_warped, nwindows = 10):
    leftx_base, rightx_base = get_histogram_pixel_line_base(binary_warped)
    margin = 100
    minpix = 50
    leftx, lefty, rightx, righty = find_lane_pixels_sliding_window(binary_warped, 
                                                                   leftx_base, 
                                                                   rightx_base, 
                                                                   margin, 
                                                                   minpix, 
                                                                   nwindows)

    return leftx, lefty, rightx, righty

def fit_polynomial_points(binary_warped, leftx, lefty, rightx, righty):
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    return left_fitx, right_fitx, ploty

def fit_polynomial_points_roc(binary_warped, leftx, lefty, rightx, righty, ym_per_pix, xm_per_pix):

    left_fit = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)

    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    try:
        left_fitx_rc = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx_rc = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        print('The function failed to fit a line!')
        left_fitx_rc = 1*ploty**2 + 1*ploty
        right_fitx_rc = 1*ploty**2 + 1*ploty

    return left_fitx_rc, right_fitx_rc, ploty

def measure_curvature_real(binary_warped, leftx, lefty, rightx, righty):
    '''
    Calculates the curvature of polynomial functions in meters.
    '''
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
    left_fit_cr, right_fit_cr, ploty = fit_polynomial_points_roc(binary_warped, leftx, lefty, rightx, righty, ym_per_pix, xm_per_pix)
    
    y_eval = np.max(ploty)
    
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    
    return left_curverad, right_curverad


def fit_polynomial_pipeline(binary_warped):
    leftx, lefty, rightx, righty = find_lane_pixels(binary_warped)
    left_fitx, right_fitx, ploty = fit_polynomial_points(binary_warped, leftx, lefty, rightx, righty)
    left_curverad, right_curverad = measure_curvature_real(binary_warped, leftx, lefty, rightx, righty)

    left_fit_pts = np.transpose(np.vstack((left_fitx, ploty)))
    right_fit_pts = np.transpose(np.vstack((right_fitx, ploty)))
    return left_fit_pts, right_fit_pts, left_curverad, right_curverad

def draw_curve_on_frame(image, left_fit_pts, right_fit_pts):
    for point in left_fit_pts:
            image[point[0], point[1]] = [0, 255, 255]
    
    for point in right_fit_pts:
            image[point[0], point[1]] = [0, 255, 255]
    return image