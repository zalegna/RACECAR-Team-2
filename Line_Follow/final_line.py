#Team 2 Line Follower Challenge Code
#Base code by Angela Zhao

#Version 5 (final version) 7/16/25
#Time ~ 19.58 sec

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
# Variables
########################################################################################

rc = racecar_core.create_racecar()

# HSV values for the two line colors
GREEN = ((40, 36, 149), (84, 255, 255)) 
ORANGE = ((1, 100, 149), (20, 255, 255))

#In this challenge, the priority of the colors does not matter. Name carried over from past challenge.
COLOR_PRIORITY = (GREEN, ORANGE)

# >> Variables
speed = 0.0  # The current speed of the car
angle = 0.0  # The current angle of the car's wheels
contour_center = None  # The (pixel row, pixel column) of contour
contour_area = 0  # The area of contour
minArea=30 #Minimum area required to be considered a contour
kp=0 #Initialize proportional gain
kd=0 #Initialize derivative gain
error=0 #Initialize error

########################################################################################
# Functions
########################################################################################

#getLineColor(color) takes an HSV color tuple and returns information about the largest contour of that color.
#If no contours of that color are detected, it returns [-1]
def getLineColor(color):
    #Take color image and crop it
    image=rc.camera.get_color_image()
    image=rc_utils.crop(image, (360, 0), (rc.camera.get_height(), rc.camera.get_width()))

    #Set HSV upper and lower values
    hsvLower=color[0]
    hsvUpper=color[1]

    #Use create a mask using the HSV values
    hsv=cv.cvtColor(image, cv.COLOR_BGR2HSV)
    mask=cv.inRange(hsv, hsvLower,hsvUpper)

    #Generate a list of contours from the mask
    contours, _=cv.findContours(mask, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    #If there exist contours, go through the list of contours and determine the largest one
    if (len(contours)>0):
        maxContour=[contours[0]]
        for contour in contours:
            if (cv.contourArea(contour)>cv.contourArea(maxContour[0])):
                maxContour[0]=contour
        #If the largest contour is bigger than the minimum area, return information about it
        if cv.contourArea(maxContour[0])>minArea:
            return [cv.contourArea(maxContour[0]), rc_utils.get_contour_center(maxContour[0]), maxContour[0]]
    #If there exist no sizeable contours, return [-1]
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
        "BWSI RACECAR Team 2\n"
        "Line Follower Challenge"
    )

# [FUNCTION] After start() is run, this function is run once every frame (ideally at
# 60 frames per second or slower depending on processing speed) until the back button
# is pressed  
def update():
    global speed
    global angle
    global contour_center
    global contour_area
    global error
    global prev_error
    global kp
    global kd
    global lineColor
    lineColor=False

    #Go through all colors in COLOR_PRIORITY
    #If a line is found, then break, and set the lineColor to that color
    for color in COLOR_PRIORITY:
        if getLineColor(color)[0]!=-1:
            lineColor=color
            break
    
    #If there exists a line, then set the contour area and contour center
    if lineColor:
        contour_area=getLineColor(lineColor)[0]
        contour_center=getLineColor(lineColor)[1]

        """Uncomment this block to view the RACECAR's camera"""
        # image=rc.camera.get_color_image()
        # image=rc_utils.crop(image, (360, 0), (rc.camera.get_height(), rc.camera.get_width()))
        # cv.drawContours(image, getLineColor(lineColor)[2], -1, (0, 255, 0), 3)
        # rc.display.show_color_image(image)

    #If there exists a contour center, then calculate angle
    if contour_center is not None:
        #Find setpoint and current contour center
        setpoint=rc.camera.get_width()//2
        presentVal=contour_center[1]

        #Calculate error and change in error
        error=setpoint-presentVal
        change = (error - prev_error) / rc.get_delta_time()

        #Calculate angle
        unclamped= kp*error + kd*change

        """Uncomment this line to print the gain contributions"""
        #print(kp*error, "||", kd*change)

        #Clamp the angle
        angle=rc_utils.clamp(unclamped, -1, 1)

        #Set the previous error
        prev_error=error

    #Set the speed to max
    rc.drive.set_max_speed(1)

    #ADAPTIVE SPEED AND GAINS 
    #If the error is high (on a turn), make the controller more reactive and slow down
    #Otherwise speed up
    if abs(error) > 230: \
        #Curve
        speed = 0.75
        kp = -0.003
        kd = -0.001
    else: 
        #Straight
        kp = -0.00075
        kd = -0.001
        speed = 1

    rc.drive.set_speed_angle(speed, angle)


# [FUNCTION] update_slow() is similar to update() but is called once per second by
# default. Remove pass to begin debugging.
def update_slow():
    """Uncomment this line to see speed"""
    # print(speed)
    
    """Uncomment this block to print a line of ascii text denoting the contour area and x-position"""
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
    pass


########################################################################################
# DO NOT MODIFY: Register start and update and begin execution
########################################################################################

if __name__ == "__main__":
    rc.set_start_update(start, update, update_slow)
    rc.go()
