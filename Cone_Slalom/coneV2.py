#Version 2 7/17/25
#Base code: Angela Zhao

#Max cones passed: 5
#Reliability: med

########################################################################################
# Imports
########################################################################################

import sys
import cv2 as cv # type: ignore
import numpy as np# type: ignore

# If this file is nested inside a folder in the labs folder, the relative path should
# be [1, ../../library] instead.
sys.path.insert(1, "../../library")
import racecar_core# type: ignore
import racecar_utils as rc_utils# type: ignore

########################################################################################
# Variables
########################################################################################

rc = racecar_core.create_racecar()

# >> Constants
# The smallest contour we will recognize as a valid contour
MIN_CONTOUR_AREA = 30
# TODO Part 1: Determine the HSV color threshold pairs for ORANGE
GREEN = ((36, 70, 90), (68, 255, 255), "green")  # The HSV range for the color blue
ORANGE = ((1, 170, 180), (15, 255, 255), "orange")  # The HSV range for the color orange

allColors=(GREEN, ORANGE)
# >> Variables
speed = 0.0  # The current speed of the car
angle = 0.0  # The current angle of the car's wheels
contour_center = None  # The (pixel row, pixel column) of contour
contour_area = 0  # The area of contour


########################################################################################
# Functions
########################################################################################

minConeArea=300
def getMaxContourCone(color):
    image=rc.camera.get_color_image()
    image=rc_utils.crop(image, (200, 0), (rc.camera.get_height(), rc.camera.get_width()))

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
        
        if cv.contourArea(maxContour[0])>minConeArea:
            return [cv.contourArea(maxContour[0]), rc_utils.get_contour_center(maxContour[0]), maxContour[0]]
        else:
            return [-1]
    else:
        return [-1]
def update_cone():
    global contour_center
    global contour_area
    global coneColor

    image = rc.camera.get_color_image()

    if image is None:
        contour_center = None
        contour_area = 0
    else:
        maxColor=allColors[0]
        for color in allColors:
            if getMaxContourCone(color)[0]>getMaxContourCone(maxColor)[0]:
                maxColor=color

        if (len(getMaxContourCone(maxColor))>1):
            contour_center=getMaxContourCone(maxColor)[1]
            contour_area=getMaxContourCone(maxColor)[0]
            coneColor=maxColor[2]
            # Draw it
            # image=rc.camera.get_color_image()
            # image=rc_utils.crop(image, (100, 0), (rc.camera.get_height(), rc.camera.get_width()))
            # cv.drawContours(image, getMaxContour(maxColor)[2], -1, (0, 255, 0), 3)
            # rc.display.show_color_image(image)
        else:
            contour_center=None
            contour_area=0
            coneColor="None"


# [FUNCTION] The start function is run once every time the start button is pressed
def start():
    global speed
    global angle

    # Initialize variables
    speed = 0
    angle = 0

    # Set initial driving speed and angle
    rc.drive.set_speed_angle(speed, angle)

    # Set update_slow to refresh every half second
    rc.set_update_slow_time(0.5)

    # Print start message
    print(
        ">> Lab H - Cone Slalom\n"
        "\n"
        "Controls:\n"
        "   A button = print current speed and angle\n"
        "   B button = print contour center and area"
    )


# [FUNCTION] After start() is run, this function is run once every frame (ideally at
# 60 frames per second or slower depending on processing speed) until the back button
# is pressed  

lastColor="none"
def update():
    global speed
    global angle
    global coneColor
    global lastColor

    # Search for contours in the current color image
    update_cone()

    scan = rc.lidar.get_samples()
    numSamples=rc.lidar.get_num_samples()
    
    crashLeft=False
    for i in range (round(numSamples*0.83), round(numSamples*0.99)):
        if scan[i]>0 and scan[i]<100:
            crashLeft=True
    crashRight=False
    
    for i in range (round(numSamples*0.01), round(numSamples*0.17)):
        if scan[i]>0 and scan[i]<100:
            crashRight=True
    
    rc.drive.set_speed_angle(speed, angle)
    if contour_center is not None and contour_area>0:
        if coneColor=="orange" and not crashRight:
            print("Curr cone: orange")
            print(f"Area:{contour_area}")
            #Find setpoint and current contour center
            setpoint=rc.camera.get_width()*.05
            presentVal=contour_center[1]

            #Calculate angle
            kp=-0.003125
            errorA=setpoint-presentVal
            unclamped=kp*errorA
            lastColor="orange"

            #Clamp the angle
            angle=rc_utils.clamp(unclamped, -1, 1)
        if coneColor=="green" and not crashLeft:
            print("Curr cone: green")
            print(f"Area:{contour_area}")
            #Find setpoint and current contour center
            setpoint=rc.camera.get_width()*.95
            presentVal=contour_center[1]

            #Calculate angle
            kp=-0.003125
            errorA=setpoint-presentVal
            unclamped=kp*errorA
            lastColor="green"

            #Clamp the angle
            angle=rc_utils.clamp(unclamped, -1, 1)
    else:
        if lastColor=="orange" and not crashLeft:
            print("Last cone: orange")
            angle=-1
        elif lastColor=="green" and not crashRight:
            print("Last cone: green")
            angle=1
    
    speed=0.8


# [FUNCTION] update_slow() is similar to update() but is called once per second by
# default. It is especially useful for printing debug messages, since printing a 
# message every frame in update is computationally expensive and creates clutter
def update_slow():
    pass


########################################################################################
# DO NOT MODIFY: Register start and update and begin execution
########################################################################################

if __name__ == "__main__":
    rc.set_start_update(start, update, update_slow)
    rc.go()

