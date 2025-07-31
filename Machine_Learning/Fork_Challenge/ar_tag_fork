"""
7/31/25
Code by Roshik Patibandla, Angela Zhao, and Dinesh Babu
AR Tag is 6x6 ID=2

Successful in trial!
"""

import sys
import math
import numpy as np
import cv2

sys.path.insert(0, '../library')
import racecar_core
import racecar_utils as rc_utils

rc = racecar_core.create_racecar()
speed = 0.0
angle = 0.0
state = 2

segment = 0
marker = 0
orientation = 0
area = 0

def fix_angle(deg):
    return deg + 360 if deg < 0 else deg

def marker_detect(image):
    global segment, marker_id, orientation, area
    
    markers = rc_utils.get_ar_markers(image)

    for marker in markers:

        marker_id = marker.get_id()
        orientation = marker.get_orientation()
        corners = marker.get_corners()
        area = abs((corners[2][1] - corners[0][1]) * (corners[2][0] - corners[0][0]))

        return marker_id, orientation

    return None, None
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

def start():
    
    rc.drive.set_max_speed(1)
    rc.set_update_slow_time(0.5)

left = False
right = False
doing_turn = None
l = 0
r = 0

def update():
    global speed, angle, state
    global marker, orientation, area
    global find_cone
    global left, right, doing_turn

    # print(state)

    image = rc.camera.get_color_image()
    if image is not None:
        marker, orientation = marker_detect(image)

    
    #Down = 2, Left = 1, Right = 3, UP = 0
    # speed = 0.65
    scan_data = rc.lidar.get_samples()
    if orientation == rc_utils.Orientation(3):
        left = True
        state = 3
        speed = 0
        print("unlocking left turn")
        if area > 400 and right == True:
            speed = 0.7
            angle = 1
            doing_turn = "right"
            print("turning right")
        
    elif orientation == rc_utils.Orientation(1):
        right = True
        state = 3
        speed = 0
        print("unlocking right turn")
        if area > 400 and left == True:
            speed = 0.7
            angle = -1
            doing_turn = "left"
            print("turning left")
    else:
        angle = 0
        print("YAY")
        state == 4

    if state == 4:
        wall()


    rc.drive.set_speed_angle(speed, angle)

def update_slow():
    pass

if __name__ == "__main__":
    rc.set_start_update(start, update, update_slow)
    rc.go()
