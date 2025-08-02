# in roshik's folder

import sys
import math
import numpy as np
import cv2

sys.path.insert(1, '../../library')
import racecar_core
import racecar_utils as rc_utils

rc = racecar_core.create_racecar()
speed = 0.0
angle = 0.0
state = 2

segment = 0
marker = 0
orientation = 0

contour_center = None
contour_area = 0
minArea=30

GREEN = ((36, 40, 180), (68, 255, 255)) 
ORANGE = ((1, 100, 149), (20, 255, 255))
COLOR_PRIORITY = (GREEN, ORANGE)

cone_state = "I_See_Orange_I_Follow"
cone_distance = 0
left_cone_distance = 0
right_cone_distance = 0
find_cone = False


def marker_detect(image):
    global segment, marker_id, orientation
    
    markers = rc_utils.get_ar_markers(image)

    for marker in markers:

        marker_id = marker.get_id()
        orientation = marker.get_orientation()
        corners = marker.get_corners()
        area = abs((corners[2][1] - corners[0][1]) * (corners[2][0] - corners[0][0]))

        return marker_id, orientation

    return None, None

def fix_angle(deg):
    return deg + 360 if deg < 0 else deg

def getLineColor(color):
    #Take color image and crop it
    image=rc.camera.get_color_image()
    image=rc_utils.crop(image, (200, 0), (rc.camera.get_height(), rc.camera.get_width()))

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


def wall():
    global speed, angle

    scan_data = rc.lidar.get_samples()
    fov_span = 120
    fan_width = 30
    scan_limit = 100
    min_gap = 120
    step = 2

    chosen_heading = 0
    best_opening = 0

    for heading in range(-75, 75, step):
        start = heading - fan_width // 2
        end = heading + fan_width // 2

        samples = []
        for ang in range(start, end + 1):
            adjusted = fix_angle(ang)
            dist = rc_utils.get_lidar_average_distance(scan_data, adjusted)
            if dist is not None and dist > scan_limit:
                samples.append(dist)

        if not samples or min(samples) < min_gap:
            continue

        candidate_clearance = min(samples)
        if candidate_clearance > best_opening:
            chosen_heading = heading
            best_opening = candidate_clearance

    special_light = 60
    sample_window = 2
    kp = 0.003

    r_angle, r_dist = rc_utils.get_lidar_closest_point(scan_data, (0, 180))
    l_angle, l_dist = rc_utils.get_lidar_closest_point(scan_data, (180, 360))
    r_shift = rc_utils.get_lidar_average_distance(scan_data, special_light, sample_window)
    l_shift = rc_utils.get_lidar_average_distance(scan_data, 360 - special_light, sample_window)

    r_component = math.sqrt(max(0, r_shift ** 2 - r_dist ** 2))
    l_component = math.sqrt(max(0, l_shift ** 2 - l_dist ** 2))

    error = r_component - l_component
    wall_adjust = rc_utils.clamp(error * kp, -1, 1)

    merged_angle = rc_utils.clamp((chosen_heading / 70.0 + wall_adjust) / 2.0, -1.0, 1.0)
    speed = 1.0 if best_opening > 120 else 0.6
    angle = merged_angle



def line():
    global speed, angle
    global contour_center, contour_area
    global error, prev_error
    global kp, kd
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
        image=rc.camera.get_color_image()
        image=rc_utils.crop(image, (200, 0), (rc.camera.get_height(), rc.camera.get_width()))
        cv.drawContours(image, getLineColor(lineColor)[2], -1, (0, 255, 0), 3)
        rc.display.show_color_image(image)

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
    scan_data = rc.lidar.get_samples()
    front = rc_utils.get_lidar_average_distance(scan_data, 0, 4)
    if front < 175: 
        print("curve")
        #Curve
        speed = 0.85
        kp = -0.002
        kd = -0
    else: 
        print("straight")
        #Straight
        kp = -0.00075
        kd = -0.001
        speed = 1

    rc.drive.set_speed_angle(speed, angle)

    
def get_lidar_cone():
    global cone_distance, left_cone_distance, right_cone_distance
    
    lidar_scan = rc.lidar.get_samples()
    scan_angle, unreal_cone_distance = rc_utils.get_lidar_closest_point(lidar_scan, (-20, 20))
    scan_angle, left_cone_distance = rc_utils.get_lidar_closest_point(lidar_scan, (-135, -45)) #45
    scan_angle, right_cone_distance = rc_utils.get_lidar_closest_point(lidar_scan, (45, 135)) #45

    cone_distance = unreal_cone_distance - 13.5

def follow_orange():
    global speed, angle, state
    image = rc.camera.get_color_image()
    if image is None:
        return False
    
    cropped = rc_utils.crop(image, (100, 0), (rc.camera.get_height(), rc.camera.get_width()))
    hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, ORANGE[0], ORANGE[1])
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return False
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 500:
        return False
    center = rc_utils.get_contour_center(largest)
    if center is None:
        return False
    setpoint = rc.camera.get_width() * 0.2
    error = setpoint - center[1]
    kp = -0.00315
    angle = rc_utils.clamp(kp * error, -1, 1)
    print("orange")

def follow_green():
    global speed, angle, state
    image = rc.camera.get_color_image()
    if image is None:
        return False
    
    cropped = rc_utils.crop(image, (100, 0), (rc.camera.get_height(), rc.camera.get_width()))
    hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, GREEN[0], GREEN[1])
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return False
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 500:
        return False
    center = rc_utils.get_contour_center(largest)
    if center is None:
        return False
    setpoint = rc.camera.get_width() * 0.8
    error = setpoint - center[1]
    kp = -0.00315
    angle = rc_utils.clamp(kp * error, -1, 1)
    print("green")

def cone():
    global cone_distance, left_cone_distance, right_cone_distance
    global speed, angle, cone_state

    if cone_state == "I_See_Orange_I_Follow":
        follow_orange()
        get_lidar_cone()
        if left_cone_distance < 60:
            print("Turning back")
            angle = -0.8
            state = "I_See_Green_I_Follow"
    elif cone_state == "I_See_Green_I_Follow":
        follow_green()
        get_lidar_cone()
        if right_cone_distance < 70:
            print("Turning back")
            angle = 0.8
            state = "I_See_Orange_I_Follow"

    speed=0.7
    rc.drive.set_speed_angle(speed, angle)

def find_cone_color():
    global cone_state, find_cone

    image = rc.camera.get_color_image()
    if image is None:
        return

    cropped = rc_utils.crop(image, (100, 0), (rc.camera.get_height(), rc.camera.get_width()))
    hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)

    # Masks
    orange_mask = cv2.inRange(hsv, ORANGE[0], ORANGE[1])
    green_mask = cv2.inRange(hsv, GREEN[0], GREEN[1])

    # Contours
    orange_contours, _ = cv2.findContours(orange_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    green_contours, _ = cv2.findContours(green_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    orange_area = 0
    green_area = 0

    if orange_contours:
        largest_orange = max(orange_contours, key=cv2.contourArea)
        if cv2.contourArea(largest_orange) > 400:
            center = rc_utils.get_contour_center(largest_orange)
            if center and 150 < center[0] < 470:
                orange_area = cv2.contourArea(largest_orange)

    if green_contours:
        largest_green = max(green_contours, key=cv2.contourArea)
        if cv2.contourArea(largest_green) > 400:
            center = rc_utils.get_contour_center(largest_green)
            if center and 150 < center[0] < 470:
                green_area = cv2.contourArea(largest_green)

    # Update state
    if orange_area == 0 and green_area == 0:
        find_cone = False
        return  # nothing found
    elif orange_area > green_area:
        cone_state = "I_See_Orange_I_Follow"
        find_cone = True
    else:
        state = "I_See_Green_I_Follow"
        find_cone = True
    

def start():
    global speed, angle
    global prev_error
    global kp, kd
    rc.drive.set_max_speed(1.0)
    speed = 1.0
    angle = 0.0

    prev_error = 0
    rc.set_update_slow_time(0.5)

    kp = 0
    kd = 0

    # Print start message
    print(
        "BWSI RACECAR Team 2\n"
        "Grand Prix"
    )

def update():
    global speed, angle, state
    global marker, orientation
    global find_cone

    # print(state)

    image = rc.camera.get_color_image()
    if image is not None:
        marker, orientation = marker_detect(image)


    if state == 1:
        wall()
        if marker == 2:
            state = 2

    if state == 3:
        line()

    if state == 2:
        cone()

    rc.drive.set_speed_angle(speed, angle)

def update_slow():
    pass

if __name__ == "__main__":
    rc.set_start_update(start, update, update_slow)
    rc.go()
