# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 17:25:27 2018

@author: Yatindra Vaishnav
"""

import numpy as np
import cv2
import glob
import pickle

def calibrateCamera(cal_image_dir, filenames):
    files = glob.glob(cal_image_dir + filenames)

    objectpoints = []
    imgpoints = []
    
    objp = np.zeros((5*9, 3), np.float32)
    objp[:,:2] = np.mgrid[0:9, 0:5].T.reshape(-1, 2)
    
    for file in files:
        image = cv2.imread(file)
    
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
        ret, corners = cv2.findChessboardCorners(gray, (9,5), None)
        if ret == True:
            imgpoints.append(corners)
            objectpoints.append(objp)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objectpoints, imgpoints, gray.shape[::-1], None, None)
    
    return mtx, dist, rvecs, tvecs
    
def undistortImage(image, mtx, dist):
    h, w = image.shape[:2]
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    dst = cv2.undistort(image, mtx, dist, None, newcameramtx)
    x,y,w,h = roi
    dst = dst[y:y+h, x:x+w]
    return dst

def saveCameraCalibrationMats(mtx, dist, rvecs, tvecs, filename="calibration.p"):
    dist_pickle = {}
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"] = dist
    dist_pickle["rvecs"] = rvecs
    dist_pickle["tvecs"] = tvecs
    fileObj = open(filename, "wb")
    pickle.dump(dist_pickle, fileObj)
    fileObj.close()

def loadCameraCalibrationMat(full_file_path="calibration.p"):
    fileObj = open(full_file_path, "rb")
    dist_pickle = {}
    dist_pickle = pickle.load(fileObj)

    mtx = dist_pickle["mtx"]
    dist = dist_pickle["dist"]
    rvecs = dist_pickle["rvecs"]
    tvecs = dist_pickle["tvecs"]
    fileObj.close()
    return mtx, dist, rvecs, tvecs