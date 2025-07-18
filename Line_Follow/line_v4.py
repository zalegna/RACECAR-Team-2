#Version 4 7/15/25
#Currently diverging, tomorrow must tune gains for both straights and curves
#Also need to test dynamic switching

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
prev_error = 0
error = 0


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
    global angle, prev_errors

    # Initialize variables
    speed = 0
    angle = 0

    prev_errors = [0,0,0,0,0]
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
    global prev_errors, prev_error

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
        scan = rc.lidar.get_samples()
        _, front_dist = rc_utils.get_lidar_closest_point(scan, (-2, 2))
        #print(front_dist)

        # if front_dist < 120:
        #     kp = -0.002
        #     kd = -0.0002
        #     #print("close")
        # else:
        #     kp = -0.001
        #     kd = -0.0002
        #     print("straight")

        kp = -0.0014
        kd = -0.0003
            
        # kp = -0.004
        # kd = -0.0002
        error=setpoint-presentVal
        
        if len(prev_errors) >= 5:
            change = (-error + 8*prev_errors[0] - 8*(prev_errors[2]) + prev_errors[3])/12# / rc.get_delta_time()
        else:
            change = error - prev_errors[0] / rc.get_delta_time()
        print("change: ", change)
        #print("errors: ", error, prev_error)

        #setting angle
        unclamped= kp*error + kd*change
        print(kp*error, "||", kd*change)

        #Clamp the angle
        angle=rc_utils.clamp(unclamped, -1, 1)

        #store prev error
        prev_errors.insert(0, error)

        if len(prev_errors) > 5:
            prev_errors.pop(-1)

    # Use the triggers to control the car's speed
    # rt = rc.controller.get_trigger(rc.controller.Trigger.RIGHT)
    # lt = rc.controller.get_trigger(rc.controller.Trigger.LEFT)
    # speed = rt - lt
    rc.drive.set_max_speed(1)

    if abs(angle) > .7:
       speed = 0.8
    else:
        speed = 1

    #print("angle: ", angle)
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
    global error
    """
    After start() is run, this function is run at a constant rate that is slower
    than update().  By default, update_slow() is run once per second
    """
    print(error)
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
