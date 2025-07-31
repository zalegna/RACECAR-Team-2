"""
MIT BWSI Autonomous RACECAR
MIT License

File Name: camera.py

Title: camera

Author: Angela Zhao

Purpose: Take a *beep* ton of photos

Expected Outcome: Takes a *beep* ton of a photos
"""

########################################################################################
# Imports
########################################################################################

import sys #type: ignore
import cv2 as cv #type: ignore
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
global counter

#NOTE: Adjust the directory based on where you want images to go!
directory="/home/racecar/jupyter_ws/training_images/racecar_imgs"

#NOTE: Rename your images/image type here. For labeling, keep it as a .jpg
imgName="racecar"
counter=0
jpg=".jpg"

########################################################################################
# Functions
########################################################################################
    
def save_photo():
    global counter
    frame=rc.camera.get_color_image()
    
    os.chdir(directory)
    filename=imgName+str(counter)+jpg
    counter+=1
    cv.imwrite(filename, frame)
    if cv.imwrite(filename, frame):
        print(f"Photo {filename} taken successfully")

# [FUNCTION] The start function is run once every time the start button is pressed
def start():
    print(f"Photos going to {directory}")

# [FUNCTION] After start() is run, this function is run once every frame (ideally at
# 60 frames per second or slower depending on processing speed) until the back button
# is pressed  
def update():
    if (rc.controller.was_released(rc.controller.Button.X)): #single photos
        save_photo()
    if (rc.controller.is_down(rc.controller.Button.B)): #burst photos
        save_photo()
    
    x, _ = rc.controller.get_joystick(rc.controller.Joystick.RIGHT)
    _, y2 = rc.controller.get_joystick(rc.controller.Joystick.LEFT)
    angle=x
    speed=y2

    rc.drive.set_speed_angle(speed, angle)
    
     # Remove 'pass' and write your source code for the update() function here

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
