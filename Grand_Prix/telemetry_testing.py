"""
MIT BWSI Autonomous RACECAR
MIT License

File Name: telemetry.py

Title: Telemetry Testing

Author: Angela Zhao

Purpose: Try out lidar visualizations

Expected Outcome: Print lidar visualizations
"""

########################################################################################
# Imports
########################################################################################

import sys
import cv2 as cv #type: ignore
import numpy as np #type: ignore
import skimage.measure as skim #type: ignore
import os 


# If this file is nested inside a folder in the labs folder, the relative path should
# be [1, ../../library] instead.
sys.path.insert(0, '../library')
import racecar_core #type: ignore

########################################################################################
# Global variables
########################################################################################

rc = racecar_core.create_racecar()

# Declare any global variables here


########################################################################################
# Functions
########################################################################################

#This function accepts a lidar scan list as well as a boolean that tells it whether this function is being used in the sim or IRL
#It returns an np array, where lidar points are seen as 255 and everything else are 0
def get_lidar_visualization(scan, sim):
    # shift scan based on yaw
    # yaw_shift = 0
    # angle_offset = int(round(yaw_shift))
    # shifted_scan = np.concatenate((scan[angle_offset*RES_PER_DEGREE:], scan[:angle_offset*RES_PER_DEGREE]))
    shifted_scan = scan

    if sim:
        RES_PER_DEGREE=2
    elif not sim:
        RES_PER_DEGREE=3
    
    # Convert polar to Cartesian
    angles_deg = np.arange(0, 360, 1/RES_PER_DEGREE)
    angles_rad = np.deg2rad(angles_deg)
    distances = np.array(shifted_scan)

    # Filter valid points
    valid = distances > 0
    distances = distances[valid]
    angles_rad = angles_rad[valid]

    x = distances * np.sin(angles_rad)
    y = distances * np.cos(angles_rad)

    # Create blank image
    #img = np.zeros((480, 640, 3), dtype=np.uint8)
    img = np.zeros((240, 240, 3), dtype=np.uint8)

    # Transform points to fit image coords (center at 320,400 and scale)
    scale = 0.2  # adjust for zoom
    # x_img = (x * scale + 320).astype(np.int32)
    # y_img = (320 - y * scale).astype(np.int32)
    x_img = (x * scale + 120).astype(np.int32)
    y_img = (120 - y * scale).astype(np.int32)

    # Clip to image bounds
    valid_idx = (x_img >= 0) & (x_img < 240) & (y_img >= 0) & (y_img < 240)
    x_img = x_img[valid_idx]
    y_img = y_img[valid_idx]
    x = x[valid_idx]
    y = y[valid_idx]

    # Draw lidar points
    for xi, yi in zip(x_img, y_img):
        cv.circle(img, (xi, yi), radius=1, color=(255, 255, 255), thickness=-1)  # white

    # Draw robot at center (red dot)
    #cv.circle(img, (120, 120), radius=4, color=(0, 0, 255), thickness=-1)  # red
    #img = rc_utils.crop(img, (0, 0), (160, 240))
    #print(img.shape) : (240, 240, 3)

    #Convert the image back to black and white for lidar visualization
    img=cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    #Use a max pooling algorithm to reduce the quality of the lidar image
    reduced=skim.block_reduce(img, (15, 15), np.max)
    return reduced

# [FUNCTION] The start function is run once every time the start button is pressed
def start():
    pass


# [FUNCTION] After start() is run, this function is run once every frame (ideally at
# 60 frames per second or slower depending on processing speed) until the back button
# is pressed  

def update():
    scan=rc.lidar.get_samples()
    lidar_img=get_lidar_visualization(scan)

    print(lidar_img)

    x, _=rc.controller.get_joystick(rc.controller.Joystick.RIGHT)
    _, y=rc.controller.get_joystick(rc.controller.Joystick.LEFT)

    rc.drive.set_speed_angle(y, x)



# [FUNCTION] update_slow() is similar to update() but is called once per second by
# default. It is especially useful for printing debug messages, since printing a 
# message every frame in update is computationally expensive and creates clutter
def update_slow():
    pass # Remove 'pass and write your source code for the update_slow() function here


########################################################################################
# DO NOT MODIFY: Register start and update and begin execution
########################################################################################

if __name__ == "__main__":
    rc.set_start_update(start, update, update_slow)
    rc.go()
