#Team 2 Cone Slalom Challenge Code
#Base code by Roshik Patibandla

#Version 3 7/17/25
#Max cones passed: 6
#Reliability: med

########################################################################################
# Imports
########################################################################################
import sys 
import cv2
import math

sys.path.insert(1, '../../library')
import racecar_core
import racecar_utils as utils


########################################################################################
# Variables
########################################################################################
RED = ((1, 170, 180), (15, 255, 255), "RED")
BLUE = ((36, 40, 89), (68, 255, 255), "BLUE")
slalom_state = "search_red"
passed_cone = False
LEFT_LIDAR = (-45, -15)
RIGHT_LIDAR = (15, 45)
FRONT_LIDAR = (0, 0)

speed = 0
angle = 0
state = 2
largest = 0

rc = racecar_core.create_racecar()

def cone_slalom():
    global speed, angle, slalom_state, largest


    rc.drive.set_max_speed(0.25)

    image = rc.camera.get_color_image()
    if image is None:
        return False

    cropped = utils.crop(image, (100, 0), (rc.camera.get_height(), rc.camera.get_width()))
    hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)

    lidar_scan = rc.lidar.get_samples()
    scan_angle, unreal_cone_distance = utils.get_lidar_closest_point(lidar_scan, (-20, 20))
    scan_angle, left_cone_distance = utils.get_lidar_closest_point(lidar_scan, (-135, -45)) #45
    scan_angle, right_cone_distance = utils.get_lidar_closest_point(lidar_scan, (45, 135)) #45

    cone_distance = unreal_cone_distance - 13.5

    if slalom_state == "red":
        mask = cv2.inRange(hsv, RED[0], RED[1])
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return False
        largest = max(contours, key=cv2.contourArea)
        print(cv2.contourArea(largest))
        if cv2.contourArea(largest) < 1000:
            return False
        center = utils.get_contour_center(largest)
        if center is None:
            return False
        setpoint = rc.camera.get_width() // 2
        error = setpoint - center[1]
        kp = -0.005
        angle = utils.clamp(kp * error, -1, 1)
        speed  = 0.7
        if cone_distance < 75:
            speed  = 0.7
            angle = 1
            slalom_state = "search_blue"
    
    if slalom_state == "search_blue":
        if left_cone_distance < 50:
            speed  = 0.7
            angle = -1
            slalom_state = "blue"
    
    if slalom_state == "blue":
        mask = cv2.inRange(hsv, BLUE[0], BLUE[1])
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return False
        largest = max(contours, key=cv2.contourArea)
        print(cv2.contourArea(largest))
        if cv2.contourArea(largest) < 1000:
            return False
        center = utils.get_contour_center(largest)
        if center is None:
            return False
        setpoint = rc.camera.get_width() // 2
        error = setpoint - center[1]
        kp = -0.005
        angle = utils.clamp(kp * error, -1, 1)
        speed  = 0.7
        if cone_distance < 75:
            speed  = 0.7
            angle = -1
            slalom_state = "search_red"
            
    if slalom_state == "search_red":
        if right_cone_distance < 50:
            speed  = 0.7
            angle = 1
            slalom_state = "red"

    print(f"State = {slalom_state} | Contour: {cv2.contourArea(largest)}| Left: {left_cone_distance}")

def update():
    global state, speed, angle

    image = rc.camera.get_color_image()
    if image is None:
        return
                              

def start():
   global state, slalom_state
   print("Started!")
   state = 2
   rc
   slalom_state = "red"
 
def update():
    global speed, state, slalom_state

    image = rc.camera.get_color_image()


    if state == 2:
        cone_slalom()
        rc.drive.set_speed_angle(speed, angle)

    if rc.controller.was_pressed(rc.controller.Button.A):
        slalom_state = "red"
        state = 2

def update_slow():
    pass

if __name__ == "__main__":
    rc.set_start_update(start, update, update_slow)
    rc.go()
