#Version 1 7/16/25
#Base code: Roshik Patibandla

#Max cones passed: 3
#Reliability: low

import sys 
import cv2
import math

sys.path.insert(1, '../../library')
import racecar_core
import racecar_utils as utils

RED = ((160, 132, 50), (10, 255, 255), "RED")
BLUE = ((88, 116, 50), (120, 255, 255), "BLUE")
slalom_state = "search_red"
passed_cone = False
LEFT_LIDAR = (-45, -15)
RIGHT_LIDAR = (15, 45)
FRONT_LIDAR = (0, 0)

speed = 0
angle = 0
state = 0

rc = racecar_core.create_racecar()

def cone_slalom():
    global speed, angle, slalom_state

    rc.drive.set_max_speed(0.5)

    image = rc.camera.get_color_image()
    if image is None:
        return False

    cropped = utils.crop(image, (200, 0), (rc.camera.get_height(), rc.camera.get_width()))
    hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)

    lidar_scan = rc.lidar.get_samples()
    scan_angle, cone_distance = utils.get_lidar_closest_point(lidar_scan, (-10, 10))
    scan_angle, left_cone_distance = utils.get_lidar_closest_point(lidar_scan, (-135, -45)) #45
    scan_angle, right_cone_distance = utils.get_lidar_closest_point(lidar_scan, (45, 135)) #45

    if slalom_state == "red":
        mask = cv2.inRange(hsv, RED[0], RED[1])
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return False
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < 500:
            return False
        center = utils.get_contour_center(largest)
        if center is None:
            return False
        setpoint = rc.camera.get_width() // 2
        setpoint =  240#NEW
        error = setpoint - center[1]
        kp = -0.0315
        angle = utils.clamp(kp * error, -1, 1)
        speed  = 0.7
        cv.drawContours(image, contours, -1, (0, 255, 0), 3)
        rc.display.show_color_image(image)
        if cone_distance < 125:
            speed  = 0.7
            angle = 1
            slalom_state = "search_blue"
    
    if slalom_state == "search_blue":
        if left_cone_distance < 100:
            speed  = 0.7
            angle = -1
            slalom_state = "blue"
    
    if slalom_state == "blue":
        mask = cv2.inRange(hsv, BLUE[0], BLUE[1])
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return False
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < 500:
            return False
        center = utils.get_contour_center(largest)
        if center is None:
            return False
        #setpoint = rc.camera.get_width() // 2
        setpoint =  400#NEW 240
        error = setpoint - center[1]
        kp = -0.0315
        angle = utils.clamp(kp * error, -1, 1)
        speed  = 0.7
        if cone_distance < 125:
            speed  = 0.7
            angle = -1
            slalom_state = "search_red"
            
    if slalom_state == "search_red":
        if right_cone_distance < 100:
            speed  = 0.7
            angle = 1
            slalom_state = "red"

def update():
    global state, speed, angle

    image = rc.camera.get_color_image()
    if image is None:
        return

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    red_mask1 = cv2.inRange(hsv, RED[0], RED[1])
    red_mask2 = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))  # Add backup red range
    combined_red = cv2.bitwise_or(red_mask1, red_mask2)
    blue_mask = cv2.inRange(hsv, BLUE[0], BLUE[1])

    red_contours, _ = cv2.findContours(combined_red, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    blue_contours, _ = cv2.findContours(blue_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    best_red = None
    best_blue = None
    red_area = 0
    blue_area = 0

    for r in red_contours:
        area = cv2.contourArea(r)
        if area > 500:
            best_red = r
            red_area = area
            break

    for b in blue_contours:
        area = cv2.contourArea(b)
        if area > 500:
            best_blue = b
            blue_area = area
            break

    if best_red is not None and best_blue is not None:
        if abs(red_area - blue_area) < 100 and blue_area > 750:
            red_center = utils.get_contour_center(best_red)
            blue_center = utils.get_contour_center(best_blue)

            if red_center is not None and blue_center is not None:
                state = 3
                speed = 0
                angle = 0
                print("Gate detected, stopping...")
                              

def start():
   global state, slalom_state
   state = 0
   slalom_state = "nothing"
 
def update():
    global speed, state, slalom_state

    image = rc.camera.get_color_image()

    if image is not None and state < 2:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        red_mask1 = cv2.inRange(hsv, RED[0], RED[1])
        red_mask2 = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255)) 
        blue_mask = cv2.inRange(hsv, BLUE[0], BLUE[1])
        combined_red = cv2.bitwise_or(red_mask1, red_mask2)
        red_contours, _ = cv2.findContours(combined_red, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        blue_contours, _ = cv2.findContours(blue_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        red_center = None
        blue_center = None

        for r in red_contours:
            if cv2.contourArea(r) > 500:
                red_center = utils.get_contour_center(r)
                break
        for b in blue_contours:
            if cv2.contourArea(b) > 500:
                blue_center = utils.get_contour_center(b)
                break

        if red_center is not None:
            state = 2
            slalom_state = "red"

        if blue_center is not None:
            state = 2
            slalom_state = "blue"

    if state == 2:
        cone_slalom()
        rc.drive.set_speed_angle(speed, angle)

    print(slalom_state)

def update_slow():
    pass

if __name__ == "__main__":
    rc.set_start_update(start, update, update_slow)
    rc.go()
