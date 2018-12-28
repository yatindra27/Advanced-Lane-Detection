# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 01:58:01 2018

@author: Yatindra Vaishnav
"""

from histLaneUtils import *
from perspectiveUtils import *
from cameraCalibration import *
import os
import matplotlib.pyplot as plt
import time
import glob
import cv2
import numpy as np
from Line import *
from moviepy.editor import *
from IPython.display import HTML

def getCameraCalibrationMats(cal_image_dir, filenames):
    fname = "calibration.p"
    mat = None
    dist_coef = None
    rvecs = None 
    tvecs = None
    if os.path.exists(cal_image_dir) == False:
        print("Invalid Path")
        return mat, dist_coef, rvecs, tvecs
        
    if (os.path.isfile(fname)):
        mat, dist_coef, rvecs, tvecs = loadCameraCalibrationMat(fname)
    else:
        mat, dist_coef, rvecs, tvecs = calibrateCamera(cal_image_dir, filenames)
        saveCameraCalibrationMats(mat, dist_coef, rvecs, tvecs, fname)
    return mat, dist_coef, rvecs, tvecs

def imageProcessingPipeline(frame, mat, dist_coef, debug=False, save_path=''):
    left_line = Line ()
    right_line = Line ()
    undist_img = undistortImage(frame, mat, dist_coef)
    if (debug == True):
        cv2.imshow('frame', frame)
        cv2.imshow('undist_img', undist_img)

    binz_img = binarize_pipeline(undist_img)
    if (debug == True):
        cv2.imshow('binz_img', binz_img)
    
    binary_warped, M, Minv = perspective_pipeline(binz_img)
    if (debug == True):
        cv2.imshow('binary_warped', binary_warped)
    
    left_fit_pts, right_fit_pts, left_curverad, right_curverad = fit_polynomial_pipeline(binary_warped)
#    print (left_curverad)
#    print (right_curverad)
    binz_dstacked = np.dstack((binary_warped, binary_warped, binary_warped))
    left_line.set_line_coordinates(left_fit_pts)
    right_line.set_line_coordinates(right_fit_pts)
    
    lane_img=np.zeros(binz_dstacked.shape)
    left_pts = np.array([left_fit_pts])
    right_pts = np.array([np.flipud(right_fit_pts)])
    pts = np.hstack((left_pts, right_pts))
    cv2.fillPoly(lane_img, np.int_([pts]), (255, 0, 0))
    cv2.polylines(lane_img, np.int32([left_pts]), isClosed=False, color=(255, 0, 0), thickness=15)
    cv2.polylines(lane_img, np.int32([right_pts]), isClosed=False, color=(255, 0, 0), thickness=15)
    if (debug == True):
        cv2.imshow("lane_img", lane_img)

    lane_img_inv = cv2.warpPerspective(lane_img, Minv, (lane_img.shape[1], lane_img.shape[0]))
    if (debug == True):
        plt.imshow(lane_img)
        cv2.imshow('lane_img', lane_img)
        cv2.imshow('binz_dstacked', binz_dstacked)
        cv2.imshow('lane_img_inv', lane_img_inv)
    
    lane_imposed_image = weighted_img(lane_img_inv, undist_img)
    if (debug == True):
        cv2.imshow("lane_imposed_image", lane_imposed_image)
        
    if save_path:
        i = 0
        while True:
            fname = 'frame' + str(i) + '.jpg'
            if not os.path.isfile(save_path + '\\' + fname):
                cv2.imwrite( save_path + '\\' + fname, frame)
                cv2.imwrite( save_path + '\\' + 'undist_img' + str(i) + '.jpg', undist_img)
                cv2.imwrite( save_path + '\\' + 'binz_img' + str(i) + '.jpg', binz_img)
                cv2.imwrite( save_path + '\\' + 'binary_warped' + str(i) + '.jpg', binary_warped)
                cv2.imwrite( save_path + '\\' + 'lane_img' + str(i) + '.jpg', lane_img)
                cv2.imwrite( save_path + '\\' + 'binz_dstacked' + str(i) + '.jpg', binz_dstacked)
                cv2.imwrite( save_path + '\\' + 'lane_img_inv' + str(i) + '.jpg', lane_img_inv)
                cv2.imwrite( save_path + '\\' + 'lane_imposed_image' + str(i) + '.jpg', lane_imposed_image)
                break
            i = i+1


    return lane_imposed_image

def getCameraCalibrationMat():
    cal_image_dir = 'E:\\Projects\\Udacity\\camera_cal\\'
    cal_images_names = "*.jpg"
    mat, dist_coef,_,_ = getCameraCalibrationMats(cal_image_dir, cal_images_names)
    if mat is None:
        print("Failed to get the distortion Coefficient")
    return mat, dist_coef

def processVideoFrame(frame):
    mat, dist_coef = getCameraCalibrationMat()
    ff = imageProcessingPipeline(frame, mat, dist_coef)
#    cv2.imshow("ff", ff)
    return ff


def processImages():
    image_dir = 'E:\\Projects\\Udacity\\test_images\\'
    filename = '*.jpg'
    mat, dist_coef = getCameraCalibrationMat()
    
    files = glob.glob(image_dir + filename)
    for file in files:
        image = cv2.imread(file)
        print(file)
        start = time.time() 
        final_image = imageProcessingPipeline(image, mat, dist_coef, False, "E:\\Projects\\Udacity\\result_image\\")
        end = time.time() 
        print(end - start)
        cv2.imshow("final_image", final_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def processVideoOpenCV():
    video_path = 'project_video.mp4'
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter("output_video.mp4", fourcc, 25, (1280, 720), 1)
    mat, dist_coef = getCameraCalibrationMat()
    
    while(cap.isOpened()):
        ret, frame = cap.read()    
        if ret == True:
            start = time.time() 
            final_frame = processVideoFrame(frame)
            end = time.time() 
            cv2.imshow(video_path, final_frame)
            out.write(final_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break 
        else:
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def processVideoMoviepy():
    video_path = 'project_video.mp4'
    output_video = 'project_video_output.mp4'
    video = VideoFileClip(video_path)
    final_video = video.fl_image(processVideoFrame)
    final_video.write_videofile(output_video)

processImages()
#processVideoOpenCV()
processVideoMoviepy()