"""
MIT BWSI Autonomous RACECAR
MIT License
racecar-neo-prereq-labs

File Name: lab_f.py

Title: Lab F - Line Follower

Author: [PLACEHOLDER] << [Write your name or team name here]

Purpose: Write a script to enable fully autonomous behavior from the RACECAR. The
RACECAR should automatically identify the color of a line it sees, then drive on the
center of the line throughout the obstacle course. The RACECAR should also identify
color changes, following colors with higher priority than others. Complete the lines 
of code under the #TODO indicators to complete the lab.

Expected Outcome: When the user runs the script, they are able to control the RACECAR
using the following keys:
- When the right trigger is pressed, the RACECAR moves forward at full speed
- When the left trigger is pressed, the RACECAR, moves backwards at full speed
- The angle of the RACECAR should only be controlled by the center of the line contour
- The RACECAR sees the color RED as the highest priority, then GREEN, then BLUE
"""

########################################################################################
# Imports
########################################################################################

import sys
import cv2 as cv #type: ignore
import numpy as np #type: ignore

# If this file is nested inside a folder in the labs folder, the relative path should
# be [1, ../../library] instead.
sys.path.insert(1, "../../library")
import racecar_core#type: ignore
import racecar_utils as rc_utils#type: ignore

########################################################################################
# Global variables
########################################################################################

rc = racecar_core.create_racecar()

# >> Constants
# The smallest contour we will recognize as a valid contour
MIN_CONTOUR_AREA = 30

# A crop window for the floor directly in front of the car
CROP_FLOOR = ((360, 0), (rc.camera.get_height(), rc.camera.get_width()))

# TODO Part 1: Determine the HSV color threshold pairs for GREEN and RED
# Colors, stored as a pair (hsv_min, hsv_max) Hint: Lab E!
GREEN = ((40, 36, 149), (84, 255, 255)) 
ORANGE = ((1, 100, 149), (20, 255, 255))


# Color priority: Red >> Green >> Blue
COLOR_PRIORITY = (GREEN, ORANGE)

# >> Variables
speed = 0.0  # The current speed of the car
angle = 0.0  # The current angle of the car's wheels
contour_center = None  # The (pixel row, pixel column) of contour
contour_area = 0  # The area of contour
prev_errors = []


########################################################################################
# Functions
########################################################################################

# [FUNCTION] Finds contours in the current color image and uses them to update 
# contour_center and contour_area
global minArea
minArea=30
def getLineColor(color):
    image=rc.camera.get_color_image()
    image=rc_utils.crop(image, (360, 0), (rc.camera.get_height(), rc.camera.get_width()))

    hsvLower=color[0]
    hsvUpper=color[1]

    hsv=cv.cvtColor(image, cv.COLOR_BGR2HSV)
    mask=cv.inRange(hsv, hsvLower,hsvUpper)

    contours, _=cv.findContours(mask, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    if (len(contours)>0):
        maxContour=[contours[0]]
        for contour in contours:
            if (cv.contourArea(contour)>cv.contourArea(maxContour[0])):
                maxContour[0]=contour
        if cv.contourArea(maxContour[0])>minArea:
            return [cv.contourArea(maxContour[0]), rc_utils.get_contour_center(maxContour[0]), maxContour[0]]
    return [-1]

# [FUNCTION] The start function is run once every time the start button is pressed
def start():
    global speed
    global angle
    global prev_error

    # Initialize variables
    speed = 0
    angle = 0

    prev_error=0
    # Set initial driving speed and angle
    rc.drive.set_speed_angle(speed, angle)

    # Set update_slow to refresh every half second
    rc.set_update_slow_time(0.5)

    # Print start message
    print(
        ">> Lab 2A - Color Image Line Following\n"
        "\n"
        "Controls:\n"
        "   Right trigger = accelerate forward\n"
        "   Left trigger = accelerate backward\n"
        "   A button = print current speed and angle\n"
        "   B button = print contour center and area"
    )

# [FUNCTION] After start() is run, this function is run once every frame (ideally at
# 60 frames per second or slower depending on processing speed) until the back button
# is pressed  
def update():
    """
    After start() is run, this function is run every frame until the back button
    is pressed
    """
    global speed
    global angle
    global contour_center
    global contour_area
    global error
    global prev_error

    # Search for contours in the current color image
    global lineColor
    lineColor=False
    for color in COLOR_PRIORITY:
        if getLineColor(color)[0]!=-1:
            lineColor=color
            break
    if lineColor:
        contour_area=getLineColor(lineColor)[0]
        contour_center=getLineColor(lineColor)[1]
        # image=rc.camera.get_color_image()
        # image=rc_utils.crop(image, (380, 0), (rc.camera.get_height(), rc.camera.get_width()))
        # cv.drawContours(image, getLineColor(lineColor)[2], -1, (0, 255, 0), 3)
        # rc.display.show_color_image(image)

    # TODO Part 3: Determine the angle that the RACECAR should receive based on the current 
    # position of the center of line contour on the screen. Hint: The RACECAR should drive in
    # a direction that moves the line back to the center of the screen.

    # Choose an angle based on contour_center
    # If we could not find a contour, keep the previous angle
    if contour_center is not None:
        #Find setpoint and current conoutr center
        setpoint=rc.camera.get_width()//2
        presentVal=contour_center[1]

        #Calculate angle
        kp = -0.004
        kd = -0.0008
        
        error=setpoint-presentVal
        change = (error - prev_error) / rc.get_delta_time()

        #setting angle
        unclamped= kp*error + kd*change
        # print(kp*error, "||", kd*change)

        #Clamp the angle
        angle=rc_utils.clamp(unclamped, -1, 1)

        prev_error=error
    # else:
    #     angle=0



    # Use the triggers to control the car's speed
    # rt = rc.controller.get_trigger(rc.controller.Trigger.RIGHT)
    # lt = rc.controller.get_trigger(rc.controller.Trigger.LEFT)
    # speed = rt - lt
    rc.drive.set_max_speed(1)

    if abs(error) > 275:
       speed = 0.8
    else:
        speed = 1

    
    
    # speed=1
    rc.drive.set_speed_angle(speed, angle)


    # Print the current speed and angle when the A button is held down
    if rc.controller.is_down(rc.controller.Button.A):
        print("Speed:", speed, "Angle:", angle)

    # Print the center and area of the largest contour when B is held down
    if rc.controller.is_down(rc.controller.Button.B):
        if contour_center is None:
            print("No contour found")
        else:
            print("Center:", contour_center, "Area:", contour_area)

# [FUNCTION] update_slow() is similar to update() but is called once per second by
# default. It is especially useful for printing debug messages, since printing a 
# message every frame in update is computationally expensive and creates clutter
def update_slow():
    """
    After start() is run, this function is run at a constant rate that is slower
    than update().  By default, update_slow() is run once per second
    """
    print(speed)
    # # Print a line of ascii text denoting the contour area and x-position
    # if rc.camera.get_color_image() is None:
    #     # If no image is found, print all X's and don't display an image
    #     print("X" * 10 + " (No image) " + "X" * 10)
    # else:
    #     # If an image is found but no contour is found, print all dashes
    #     if contour_center is None:
    #         print("-" * 32 + " : area = " + str(contour_area))

    #     # Otherwise, print a line of dashes with a | indicating the contour x-position
    #     else:
    #         s = ["-"] * 32
    #         s[int(contour_center[1] / 20)] = "|"
    #         print("".join(s) + " : area = " + str(contour_area))


########################################################################################
# DO NOT MODIFY: Register start and update and begin execution
########################################################################################

if __name__ == "__main__":
    rc.set_start_update(start, update, update_slow)
    rc.go()
