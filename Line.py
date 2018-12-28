# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 13:57:35 2018

@author: Yatindra Vaishnav
"""
import numpy as np

class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = None
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x,y values for detected line pixels
        self.all_curve_pts = None
    
    def set_line_coordinates(self, fit_pts):
        self.all_curve_pts=fit_pts
        self.current_fit = fit_pts
        