#Version 4 (final) 7/18/25
#Base code: Roshik with Angela's cone approach

#Cones completed in trial: max
#Max cones: max
#Reliability: high 

#NOTE: Use tape to mark out optimal starting position (facing AWAY from the first cone)

########################################################################################
# Imports
########################################################################################

import sys #type: ignore
import cv2 #type: ignore

# If this file is nested inside a folder in the labs folder, the relative path should
# be [1, ../../library] instead.
sys.path.insert(1, '../../library')
import racecar_core #type: ignore
import racecar_utils as rc_utils #type: ignore

########################################################################################
# Variables
########################################################################################
ORANGE = ((1, 170, 180), (15, 255, 255), "ORANGE")
GREEN = ((36, 40, 89), (68, 255, 255), "GREEN")

cone_distance = 0
left_cone_distance = 0
right_cone_distance = 0

#Initialize speed, angle, and state machine
speed = 0
angle = 0
state = "I_See_Orange_I_Follow"


rc = racecar_core.create_racecar()

########################################################################################
# Functions
########################################################################################

#get_lidar() obtains the cloest lidar point from certain areas.
def get_lidar():
    global cone_distance, left_cone_distance, right_cone_distance
    
    lidar_scan = rc.lidar.get_samples()
    scan_angle, unreal_cone_distance = rc_utils.get_lidar_closest_point(lidar_scan, (-20, 20))
    scan_angle, left_cone_distance = rc_utils.get_lidar_closest_point(lidar_scan, (-135, -45)) #45
    scan_angle, right_cone_distance = rc_utils.get_lidar_closest_point(lidar_scan, (45, 135)) #45

    cone_distance = unreal_cone_distance - 13.5

#follow_orange() turns the car to the side of the nearest orange cone
def follow_orange():
    global speed, angle, state

    #Obtain the largest orange contour 
    image = rc.camera.get_color_image()
    
    if image is None:
        return False
    cropped = rc_utils.crop(image, (100, 0), (rc.camera.get_height(), rc.camera.get_width()))
    hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, ORANGE[0], ORANGE[1])
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return False
    
    #If there exists a largest contour, update the contour center and maximum area
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 500:
        return False
    center = rc_utils.get_contour_center(largest)
    if center is None:
        return False
    
    #Approach the side of the cone, following the center of the contour from the edge of the screen
    setpoint = rc.camera.get_width() * 0.2
    error = setpoint - center[1]
    kp = -0.00315
    angle = rc_utils.clamp(kp * error, -1, 1)
    print("orange")

#follow_green() turns the car to the side of the nearest green cone
def follow_green():
    global speed, angle, state

    #Obtain the largest orange contour 
    image = rc.camera.get_color_image()
    if image is None:
        return False
    
    cropped = rc_utils.crop(image, (100, 0), (rc.camera.get_height(), rc.camera.get_width()))
    hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, GREEN[0], GREEN[1])
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return False
    
    #If there exists a largest contour, update the contour center and maximum area
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 500:
        return False
    center = rc_utils.get_contour_center(largest)
    if center is None:
        return False
    
    #Approach the side of the cone, following the center of the contour from the edge of the screen
    setpoint = rc.camera.get_width() * 0.8
    error = setpoint - center[1]
    kp = -0.00315
    angle = rc_utils.clamp(kp * error, -1, 1)
    print("green")

def start():
    global state
    state = "I_See_Orange_I_Follow"
    rc.drive.set_max_speed(0.35)

def update():
    global cone_distance, left_cone_distance, right_cone_distance
    global speed, angle, state

    #Based on the current state, follow a different color of cone. When the distance to the side is below a certain
    #threshold, begin turning the other direction.
    if state == "I_See_Orange_I_Follow":
        follow_orange()
        get_lidar()
        if left_cone_distance < 60:
            print("Turning back")
            angle = -0.8
            state = "I_See_Green_I_Follow"
    elif state == "I_See_Green_I_Follow":
        follow_green()
        get_lidar()
        if right_cone_distance < 60:
            print("Turning back")
            angle = 0.8
            state = "I_See_Orange_I_Follow"

    speed=0.7
    rc.drive.set_speed_angle(speed, angle)

def update_slow():
    pass


########################################################################################
# DO NOT MODIFY: Register start and update and begin execution
########################################################################################

if __name__ == "__main__":
    rc.set_start_update(start, update, update_slow)
    rc.go()
