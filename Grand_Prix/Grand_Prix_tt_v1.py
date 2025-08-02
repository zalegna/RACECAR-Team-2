"""
MIT BWSI Autonomous RACECAR
MIT License
racecar-neo-prereq-labs

File Name: grand_prix_irl.py

Title: Grand Prix!!!!!!!!

Author: team MONOPOLY (2)

Purpose: to win the grand prix lesgoo

Expected Outcome: win
"""

########################################################################################
# Imports
########################################################################################

import sys
import cv2 as cv
import numpy as np
import skimage.measure as skim

# If this file is nested inside a folder in the labs folder, the relative path should
# be [1, ../../library] instead.
sys.path.insert(1, '../../library')
import racecar_core
import racecar_utils as rc_utils

## do more tpu bs here

########################################################################################
# Global variables
########################################################################################

rc = racecar_core.create_racecar()

# sensor vals
image = None
scan = None

# Movement variables
speed = 0
angle = 0

# wall stuff
front_distance = 0
left_distance = 0
left_angle = 0
right_distance = 0
right_angle = 0
turn_error = 0
kp_wall = .003
kd_wall = 0

# Lidar constants      
LEFT_LIDAR = (-45, -15)
RIGHT_LIDAR = (15, 45)
FRONT_LIDAR = (0, 0)
SPEED_LIMITER = 500
TURN_LIMITER = 27

# Line follower shit
line_min_area = 30 #Minimum area required to be considered a contour
kp_line = 0 #Initialize proportional gain
kd_line = 0 #Initialize derivative gain
error_line = 0
prev_error_line = 0

# AR Marker detection
arucoDict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_6X6_50)
arucoParams = cv.aruco.DetectorParameters()
detector = cv.aruco.ArucoDetector(arucoDict, arucoParams)

# HSV color thresholds
BLUE = ((90, 115, 115), (120, 255, 255), "BLUE")
#GREEN = ((40, 115, 115), (80, 255, 255), "GREEN")
RED = ((160, 50, 20), (179, 255, 255), "RED")
RED2 = ((10, 100, 100), (20, 255, 255))
#ORANGE = ((10, 100, 100), (20, 255, 255), "ORANGE")
PURPLE = ((130, 100, 100), (160, 255, 255), "PURPLE")
ORANGE = ((1, 170, 180), (15, 255, 255), "ORANGE")
GREEN = ((36, 40, 89), (68, 255, 255), "GREEN")

COLORS = [BLUE, RED, GREEN]
COLOR_PRIORITY = [BLUE, GREEN]

# Storage for last detected marker info
last_marker_id = None
last_marker_color = None
last_marker_area = 0
state = -1
error_integral = 0

# slalom stuff
cone_distance = 0
left_cone_distance = 0
right_cone_distance = 0
slalom_state = "I_See_Orange_I_Follow"

#List of states
states=["Wall Follow", "Cone Slalom", "Waiting State"]

# brake relative to velocity (use for safety stop)
def p_brake():
    velocity = rc.physics.get_velocity()
    if np.abs(velocity) > 0.1:
        speed = -rc.physics.get_velocity()*5
        rc_utils.clamp(speed, -1, 1)

########################################################################################
# AR Marker Class
########################################################################################

class ARMarker:
    def __init__(self, marker_id, marker_corners, orientation, area):
        self.id = marker_id
        self.corners = marker_corners
        self.orientation = orientation
        self.area = area
        self.color = ""
        self.color_area = 0

    def find_color_border(self, image):
        crop_points = self.find_crop_points(image)
        image = image[crop_points[0][0]:crop_points[0][1], crop_points[1][0]:crop_points[1][1]]
        self.color, self.color_area = self.find_colors(image)

    def find_crop_points(self, image):
        ORIENT = {"UP": 0, "LEFT": 1, "DOWN": 2, "RIGHT": 3}
        o = ORIENT[self.orientation]
        left, top = self.corners[o]
        right, bottom = self.corners[(o + 2) % 4]

        half_len = (right - left) // 2
        half_wid = (bottom - top) // 2

        top = max(0, top - half_wid)
        left = max(0, left - half_len)
        bottom = min(image.shape[0], bottom + half_wid) + 1
        right = min(image.shape[1], right + half_len) + 1

        return ((int(top), int(bottom)), (int(left), int(right)))

    def find_colors(self, image):
        color_name = "None"
        color_area = 0
        for (hsv_lower, hsv_upper, color) in COLORS:
            contours = rc_utils.find_contours(image, hsv_lower, hsv_upper)
            largest = rc_utils.get_largest_contour(contours)
            if largest is not None:
                area = rc_utils.get_contour_area(largest)
                if area > color_area:
                    color_area = area
                    color_name = color
        return color_name, color_area

########################################################################################
# Cone slalom D:
########################################################################################

def get_lidar_slalom():
    global cone_distance, left_cone_distance, right_cone_distance, scan
    
    lidar_scan = scan
    _, unreal_cone_distance = rc_utils.get_lidar_closest_point(lidar_scan, (-20, 20)) # first output is scan angle btw
    _, left_cone_distance = rc_utils.get_lidar_closest_point(lidar_scan, (-135, -45)) #45
    _, right_cone_distance = rc_utils.get_lidar_closest_point(lidar_scan, (45, 135)) #45

    cone_distance = unreal_cone_distance - 13.5

def follow_orange():
    global speed, angle, slalom_state, image

    #Obtain the largest orange contour
    if image is None:
        return False
    cropped = rc_utils.crop(image, (100, 0), (rc.camera.get_height(), rc.camera.get_width()))
    hsv = cv.cvtColor(cropped, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, ORANGE[0], ORANGE[1])
    contours, _ = cv.findContours(mask, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return False
    
    #If there exists a largest contour, update the contour center and maximum area
    largest = max(contours, key=cv.contourArea)
    if cv.contourArea(largest) < 500:
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
    global speed, angle, slalom_state, image

    #Obtain the largest orange contour 
    if image is None:
        return False
    cropped = rc_utils.crop(image, (100, 0), (rc.camera.get_height(), rc.camera.get_width()))
    hsv = cv.cvtColor(cropped, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, GREEN[0], GREEN[1])
    contours, _ = cv.findContours(mask, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return False
    
    #If there exists a largest contour, update the contour center and maximum area
    largest = max(contours, key=cv.contourArea)
    if cv.contourArea(largest) < 500:
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

def cone_slalom():
    global cone_distance, left_cone_distance, right_cone_distance
    global speed, angle, slalom_state

    #Based on the current state, follow a different color of cone. When the distance to the side is below a certain
    #threshold, begin turning the other direction.
    if slalom_state == "I_See_Orange_I_Follow":
        follow_orange()
        get_lidar_slalom()
        if left_cone_distance < 60:
            print("Turning back")
            angle = -0.8
            slalom_state = "I_See_Green_I_Follow"
    elif slalom_state == "I_See_Green_I_Follow":
        follow_green()
        get_lidar_slalom()
        if right_cone_distance < 60:
            print("Turning back")
            angle = 0.8
            slalom_state = "I_See_Orange_I_Follow"

    speed=0.7
     
########################################################################################
# Line Following
########################################################################################

def getLineColor(color):
    global image
    
    #Take color image and crop it
    image = rc_utils.crop(image, (360, 0), (rc.camera.get_height(), rc.camera.get_width()))

    #Set HSV upper and lower values
    hsvLower = color[0]
    hsvUpper = color[1]

    #Use create a mask using the HSV values
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, hsvLower,hsvUpper)

    #Generate a list of contours from the mask
    contours, _=cv.findContours(mask, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    #If there exist contours, go through the list of contours and determine the largest one
    if (len(contours)>0):
        maxContour=[contours[0]]
        for contour in contours:
            if (cv.contourArea(contour)>cv.contourArea(maxContour[0])):
                maxContour[0]=contour
        #If the largest contour is bigger than the minimum area, return information about it
        if cv.contourArea(maxContour[0])>line_min_area:
            return [cv.contourArea(maxContour[0]), rc_utils.get_contour_center(maxContour[0]), maxContour[0]]
    #If there exist no sizeable contours, return [-1]
    return [-1]


def line_follow():
    global speed, angle, contour_center, contour_area, prev_error_line, error_line, kp_line, kd_line, lineColor
    lineColor = False

    #Go through all colors in COLOR_PRIORITY
    #If a line is found, then break, and set the lineColor to that color
    for color in COLOR_PRIORITY:
        if getLineColor(color)[0] != -1:
            lineColor = color
            break

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
        setpoint = rc.camera.get_width()//2
        presentVal = contour_center[1]

        #Calculate error and change in error
        error_line=setpoint-presentVal
        change = (error_line - prev_error_line) / rc.get_delta_time()

        #Calculate angle
        unclamped = kp_line*error_line + kd_line*change

        #print(kp*error, "||", kd*change)

        #Clamp the angle
        angle = rc_utils.clamp(unclamped, -1, 1)

        #Set the previous error
        prev_error_line = error_line

    #ADAPTIVE SPEED AND GAINS 
    #If the error is high (on a turn), make the controller more reactive and slow down
    #Otherwise speed up
    if abs(error_line) > 230:
        #Curve
        speed = 0.75
        kp_line = -0.003
        kd_line = -0.001
    else: 
        #Straight
        kp_line = -0.00075
        kd_line = -0.001
        speed = 1

    rc.drive.set_speed_angle(speed, angle)

########################################################################################
# Wall following
########################################################################################

def fix_angle(deg):
    return deg + 360 if deg < 0 else deg

# commenting maybe?
def wall_follow():
    global speed, angle, scan, kp_wall

    # parameters
    fov_span = 120
    fan_width = 30
    scan_limit = 100
    min_gap = 120
    step = 2

    # a few initializations
    chosen_heading = 0
    best_opening = 0

    for heading in range(-75, 75, step):
        start = heading - fan_width // 2
        end = heading + fan_width // 2

        samples = []
        for ang in range(start, end + 1):
            adjusted = fix_angle(ang)
            dist = rc_utils.get_lidar_average_distance(scan, adjusted)
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
    kp_wall = 0.003

    r_angle, r_dist = rc_utils.get_lidar_closest_point(scan, (0, 180))
    l_angle, l_dist = rc_utils.get_lidar_closest_point(scan, (180, 360))
    r_shift = rc_utils.get_lidar_average_distance(scan, special_light, sample_window)
    l_shift = rc_utils.get_lidar_average_distance(scan, 360 - special_light, sample_window)

    r_component = np.sqrt(max(0, r_shift ** 2 - r_dist ** 2))
    l_component = np.sqrt(max(0, l_shift ** 2 - l_dist ** 2))

    error_wall = r_component - l_component
    wall_adjust = rc_utils.clamp(error_wall * kp_wall, -1, 1)

    merged_angle = rc_utils.clamp((chosen_heading / 70.0 + wall_adjust) / 2, -1.0, 1.0)
    speed = 1.0 if best_opening > 220 else 0.6

    #print(chosen_heading, wall_adjust)
    angle = merged_angle
    print(angle)
    #display_angle = int(merged_angle * 100)
    #rc.display.show_text(f".{display_angle}")

########################################################################################
# Lidar visualization
########################################################################################

def get_lidar_visualization(scan, sim):
    # shift scan based on yaw
    # yaw_shift = 0
    # angle_offset = int(round(yaw_shift))
    # shifted_scan = np.concatenate((scan[angle_offset*RES_PER_DEGREE:], scan[:angle_offset*RES_PER_DEGREE]))
    shifted_scan = scan

    if sim:
        RES_PER_DEGREE=2
    elif not sim:
        RES_PER_DEGREE=3
    
    # Convert polar to Cartesian
    angles_deg = np.arange(0, 360, 1/RES_PER_DEGREE)
    angles_rad = np.deg2rad(angles_deg)
    distances = np.array(shifted_scan)

    # Filter valid points
    valid = distances > 0
    distances = distances[valid]
    angles_rad = angles_rad[valid]

    x = distances * np.sin(angles_rad)
    y = distances * np.cos(angles_rad)

    # Create blank image
    #img = np.zeros((480, 640, 3), dtype=np.uint8)
    img = np.zeros((240, 240, 3), dtype=np.uint8)

    # Transform points to fit image coords (center at 320,400 and scale)
    scale = 0.2  # adjust for zoom
    # x_img = (x * scale + 320).astype(np.int32)
    # y_img = (320 - y * scale).astype(np.int32)
    x_img = (x * scale + 120).astype(np.int32)
    y_img = (120 - y * scale).astype(np.int32)

    # Clip to image bounds
    valid_idx = (x_img >= 0) & (x_img < 240) & (y_img >= 0) & (y_img < 240)
    x_img = x_img[valid_idx]
    y_img = y_img[valid_idx]
    x = x[valid_idx]
    y = y[valid_idx]

    # Draw lidar points
    for xi, yi in zip(x_img, y_img):
        cv.circle(img, (xi, yi), radius=1, color=(255, 255, 255), thickness=-1)  # white

    # Draw robot at center (red dot)
    #cv.circle(img, (120, 120), radius=4, color=(0, 0, 255), thickness=-1)  # red
    #img = rc_utils.crop(img, (0, 0), (160, 240))
    #print(img.shape) : (240, 240, 3)

    #Convert the image back to black and white for lidar visualization
    img=cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    #Use a max pooling algorithm to reduce the quality of the lidar image
    reduced=skim.block_reduce(img, (15, 15), np.max)
    return reduced

########################################################################################
# Elevator
########################################################################################

def elevator_wait():
    global speed
    print(scan[0])
    if scan[0] > 50 or scan[0] == 0: # if scan > 50; lidar is above
        speed = 1
    else:
        speed = 0

def elevator_align():
    pass
    
########################################################################################
# start/update/update_slow
########################################################################################

def start():
    global speed, angle, state
    speed = 1
    rc.drive.set_max_speed(.5)
    state = 0

def update():
    global speed, angle, image, scan
    global left_angle, left_distance, right_angle, right_distance, front_distance, turn_error
    global last_marker_id, last_marker_color, last_marker_area, state

    image = rc.camera.get_color_image()
    scan = rc.lidar.get_samples()

    #GET UPDATE FUNCTION FPS
    fps = 1/rc.get_delta_time()
    #print(f"FPS: {fps}")

    #Print telemetry info
    #lidar_img=get_lidar_visualization(scan, False)
    # print(lidar_img)
    # print(f"Current State: {states[state]}")

    #state = 0
    # waiting state
    if state == -1:
        # wait
        pass

    # larger overarching state machine for the obstacles
    if state == 0:
        wall_follow()
    elif state == 2:
        cone_slalom()
    if state == 3:
        elevator_wait()
    
    # --- AR MARKER DETECTION ---
    corners, ids, _ = detector.detectMarkers(image)
    #print(ids)
    if corners:
        current_corners = corners[0][0]
        AR_area = abs((current_corners[2][1] - current_corners[0][1]) * (current_corners[2][0] - current_corners[0][0]))
        #print(AR_area)
        if AR_area > 3300:
            last_marker_id = ids[0]
    
    state = last_marker_id
        
    # Safety stop
    # if scan[0] != 0 and scan[0] < 20: # if we are is too close (but not reading blanks)
    #     p_brake()

    # pretty sure this gets run anyway in all of them, but just in case lol. Clamp to avoid out of bounds error
    speed = rc_utils.clamp(speed, -1, 1)
    angle = rc_utils.clamp(angle, -1, 1)

    # send that speed and anglee
    rc.drive.set_speed_angle(speed, angle)

def update_slow():
    global last_marker_id, last_marker_color, last_marker_area
    if last_marker_id is not None:
        print("====== AR Marker Info ======")
        print(f"ID: {last_marker_id} | Color: {last_marker_color} | Color Area: {last_marker_area}")
        print("============================\n")

########################################################################################
# DO NOT MODIFY: Register start and update and begin execution
########################################################################################

if __name__ == "__main__":
    rc.set_start_update(start, update, update_slow)
    rc.go()
