"""
MIT BWSI Autonomous RACECAR

Title: Grand Prix!!!!!!!!

Author: team MONOPOLY (2)

Purpose: to win the grand prix lesgoo
"""

########################################################################################
# Imports
########################################################################################

import sys
import math
import numpy as np
import cv2
import skimage.measure as skim

sys.path.insert(0, '../library')
import racecar_core
import racecar_utils as rc_utils
from enum import IntEnum

from pycoral.adapters.common import input_size
from pycoral.adapters.detect import get_objects
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.edgetpu import run_inference

# Define paths to model and label directories
default_path = 'elevator' # location of model weights and labels
model_name = 'team2_elevatorsign_edgetpu.tflite'
label_name = 'elevator.txt'

model_path = default_path + "/" + model_name
label_path = default_path + "/" + label_name

########################################################################################
# Global variables
########################################################################################

rc = racecar_core.create_racecar()
interpreter = None
labels = None
inference_size = None

SCORE_THRESH = 0.1
NUM_CLASSES = 4

speed = 0.0
angle = 0.0
state = 0

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

# for ml
def get_obj_and_type(cv2_im, inference_size, objs):
    global x0, x1, y0, y1
    height, width, _ = cv2_im.shape
    max_score = 0
    correct_obj = None
    scale_x, scale_y = width / inference_size[0], height / inference_size[1]
    for obj in objs:
        if obj.score > max_score:
            max_score = obj.score
            correct_obj = obj
    if (correct_obj is not None):
        bbox = correct_obj.bbox.scale(scale_x, scale_y)
        x0, y0 = int(bbox.xmin), int(bbox.ymin)
        x1, y1 = int(bbox.xmax), int(bbox.ymax)
        center = ((x0+x1)/2, (y0+y1)/2)
        id = correct_obj.id
    return center, id, correct_obj

# brake relative to velocity (use for safety stop)
def p_brake():
    velocity = rc.physics.get_velocity()
    if np.abs(velocity) > 0.1:
        speed = -rc.physics.get_velocity()*5
        rc_utils.clamp(speed, -1, 1)

########################################################################################
# AR Markers
########################################################################################

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
    
########################################################################################
# Line Following
########################################################################################

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
    
def line():
    global speed, angle
    global contour_center, contour_area
    global error, prev_error
    global kpL, kd
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
        unclamped= kpL*error + kd*change

        """Uncomment this line to print the gain contributions"""
        #print(kpL*error, "||", kd*change)

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
        kpL = -0.002
        kd = -0
    else: 
        print("straight")
        #Straight
        kpL = -0.00075
        kd = -0.001
        speed = 1

########################################################################################
# Wall following
########################################################################################

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

    r_angle, r_dist = rc_utils.get_lidar_closest_point(scan_data, (0, 180))
    l_angle, l_dist = rc_utils.get_lidar_closest_point(scan_data, (180, 360))
    r_shift = rc_utils.get_lidar_average_distance(scan_data, special_light, sample_window)
    l_shift = rc_utils.get_lidar_average_distance(scan_data, 360 - special_light, sample_window)

    r_component = math.sqrt(max(0, r_shift ** 2 - r_dist ** 2))
    l_component = math.sqrt(max(0, l_shift ** 2 - l_dist ** 2))

    kpW = 0.003

    error = r_component - l_component
    wall_adjust = rc_utils.clamp(error * kpW, -1, 1)

    merged_angle = rc_utils.clamp((chosen_heading / 65 + wall_adjust) / 1.5, -1.0, 1.0)
    speed = 1.0
    angle = merged_angle

    display_angle = int(merged_angle * 100)
    rc.display.show_text(f".{display_angle}")

########################################################################################
# Cone slalom D:
########################################################################################

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
    angle = rc_utils.clamp(kpC * error, -1, 1)
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
    angle = rc_utils.clamp(kpC * error, -1, 1)
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

    speed=0.5

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

########################################################################################
# telemetry
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
# start/update/update_slow
########################################################################################

def start():
    global speed, angle
    global prev_error
    global kpL, kd, kpC
    global interpreter, labels, inference_size

    kpC = -0.00315
    kpL = -0.002
    kd = -0
    
    rc.drive.set_max_speed(1.0)
    speed = 1.0
    angle = 0.0

    prev_error = 0
    rc.set_update_slow_time(0.5)

    # Print start message
    print(
        "BWSI RACECAR Team 2\n"
        "Grand Prix"
    )
    print("                                                                                                  ")
    print("                                                                                                  ")
    print("                                                     ...                                          ")
    print("                                                   .+=-=-..       ..::..                          ")
    print("                 .......-.....                     :=    :=:.  .-==-. :=-.                        ")
    print("                        #.:::.                    :-      .--.-=:       +.                        ")
    print("                        #                         ::        .-=.        +.                        ")
    print("                        #                         ::         ..        .=.                        ")
    print("                        #                         .+.                 :-.                         ")
    print("                        #                          :-.              .=:.                          ")
    print("                        #                           .==-:.        :=-.                            ")
    print("                 .-----=#=========:                   ..:-----:..:-.                              ")
    print("                                                             .:=+-.                               ")
    print("                                                             :-::.                                ")
    print("                                                                                                  ")
    print("                                                                                                  ")
    print("                                                                                                  ")
    print("                                                                                                  ")
    print("                                                                                                  ")
    print("                                            :--. ..                                               ")
    print("                                      .:====++*++===-.:.                                          ")
    print("                                  .:===*##%%%##%%%%##*==:                                         ")
    print("                                .:-=+*#################*==-:                                      ")
    print("                                :+=*##*+++==+*#+=+#######+=-.                                     ")
    print("                              ..-=+%%#**###***##*+=+#####*==-.                                    ")
    print("                              .:+=*%#####+==##+=*##+=+####+==.                                    ")
    print("                              .:+=+#######*==*%*+=+*==+%%#==-.                                    ")
    print("                               .===*########+==*%%####%%#+=*=.                                    ")
    print("                               .+*+=+##################*===+=.                                    ")
    print("                               .:++===+*#############*==+#*-                                      ")
    print("                                 .=##+=====++++++======+**=.                                      ")
    print("                                  .:-+*###+==+**===####+.                                         ")
    print("                                     .+#**#########=---.                                          ")
    print("                                       .. .-=-..:-:                                               ")
    print("                                                                                                  ")
    print("                                                                                                  ")
    print("                                                                                                  ")
    print("                                                                                                  ")
    print("                                                                                                  ")


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
    
def elevator():
    return

########################################################################################
# start/update/update_slow
########################################################################################

def update():
    global speed, angle, state
    global marker, orientation
    global find_cone

    image = rc.camera.get_color_image()
    if image is not None:
        marker, orientation = marker_detect(image)
        if marker == 1:
            state = 1
        if marker == 2:
            state = 2
        if marker == 3:
            state = 3
        if marker == 4:
            state = 4
        if marker == 5:
            state = 5


    if state == 0:
        wall()

    if state == 1:
        elevator()

    if state == 2:
        cone()

    if state == 4:
        speed = 0

    if state == 1 or state == 5:
        wall()

    scan=rc.lidar.get_samples()
    lidar_img=get_lidar_visualization(scan, False)

    print(lidar_img)
    rc.drive.set_speed_angle(speed, angle)

def update_slow():
    pass

if __name__ == "__main__":
    rc.set_start_update(start, update, update_slow)
    rc.go()
