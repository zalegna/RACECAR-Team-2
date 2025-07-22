"""
MIT BWSI Autonomous RACECAR
MIT License
racecar-neo-prereq-labs

File Name: camera.py

Title: camera

Author: Angela Zhao

Purpose: Take a *beep* ton of photos for model training.

Expected Outcome: When the X button is released, take one photo. When the B button is pressed, begin rapidly taking photos until it's pressed again.

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

#NOTE: Adjust the directory based on where you want the images to go!
directory="/home/racecar/jupyter_ws/(YOUR FOLDERS HERE)"

#Change the name of the file
imgName="go_Around"
#Change what number file (useful for keeping track of number of photos taken)
counter=0
#Change the datatype of the file
jpg=".jpg"

########################################################################################
# Functions
########################################################################################
    
def save_photo(frame):
    global counter

    #Change the directory 
    os.chdir(directory)
    #Name the file, and change the name
    filename=imgName+str(counter)+jpg
    counter+=1

    #Save image
    cv.imwrite(filename, frame)
    if cv.imwrite(filename, frame):
        print(f"Image number {counter} added successfully!")

# [FUNCTION] The start function is run once every time the start button is pressed
def start():
    print("Started!")

# [FUNCTION] After start() is run, this function is run once every frame (ideally at
# 60 frames per second or slower depending on processing speed) until the back button
# is pressed  

continuous=False
def update():
    global continuous 
    frame=rc.camera.get_color_image()
    rc.display.show_color_image(frame)
    
    if (rc.controller.was_released(rc.controller.Button.X)): #single photos
        save_photo(frame)
    if (rc.controller.was_pressed(rc.controller.Button.B)): #burst photos
        continuous=not continuous

    if continuous:
        save_photo(frame)
    
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
