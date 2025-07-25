# combines Roshik's better.py wall follower with Dinesh's sign detection
# 7/25/25

########################################################################################
# Imports
########################################################################################

from random import seed
import sys
import cv2 as cv2
import numpy as np
import os, time
import math


sys.path.insert(0, "../library")
import racecar_core
import racecar_utils as rc_utils
from enum import IntEnum

from pycoral.adapters.common import input_size
from pycoral.adapters.detect import get_objects
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.edgetpu import run_inference

# Define paths to model and label directories
default_path = 'models' # location of model weights and labels
model_name = 'signModelWithFakes2_edgetpu.tflite'
label_name = 'newsigns2.txt'

model_path = default_path + "/" + model_name
label_path = default_path + "/" + label_name

# Define thresholds and number of classes to output
SCORE_THRESH = 0.1
NUM_CLASSES = 3

rc = racecar_core.create_racecar()

speed = 0.0
angle = 0.0

interpreter = None
labels = None
inference_size = None

counter_s = 0
counter_y = 0
counter_g = 0

def append_objs_to_img(cv2_im, inference_size, objs, labels):
    height, width, channels = cv2_im.shape
    scale_x, scale_y = width / inference_size[0], height / inference_size[1]
    for obj in objs:
        if obj.score > 0.5:
            bbox = obj.bbox.scale(scale_x, scale_y)
            x0, y0 = int(bbox.xmin), int(bbox.ymin)
            x1, y1 = int(bbox.xmax), int(bbox.ymax)
    
            percent = int(100 * obj.score)
            label = '{}% {}'.format(percent, labels.get(obj.id, obj.id))
    
            cv2_im = cv2.rectangle(cv2_im, (x0, y0), (x1, y1), (0, 255, 0), 2)
            cv2_im = cv2.putText(cv2_im, label, (x0, y0+30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
    return cv2_im

def fix_angle(deg):
    return deg + 360 if deg < 0 else deg

def start():
    global speed, angle
    global interpreter, labels, inference_size
    rc.drive.set_max_speed(1.0)
    speed = 1.0
    angle = 0.0
    rc.set_update_slow_time(0.5)

    #for signs
    interpreter = make_interpreter(model_path)
    interpreter.allocate_tensors()
    labels = read_label_file(label_path)
    inference_size = input_size(interpreter)

def update():
    global speed, angle
    global interpreter, labels, inference_size
    global counter_s, counter_y, counter_g

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
    speed = 1.0 if best_opening > 220 else 0.6

    #sign code here
    frame = rc.camera.get_color_image()

    if frame is not None:
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_image = cv2.resize(rgb_image, inference_size)
    
        # STEP 5: Let the model do the work
        run_inference(interpreter, rgb_image.tobytes())
    
        # STEP 6: Get objects detected from the model
        objs = get_objects(interpreter, SCORE_THRESH)[:NUM_CLASSES]
        # order: stop (0), yield (1), go (2), fakeStop (3), fakeYield (4)
        
        
        # stop sign
        if len(objs) > 0:
            sign = objs[0].id
            score = objs[0].score

            #stop
            if sign == 0 and score > 0.7 or counter_s > 0:
                print("STOP")
                speed = 0
                angle = 0
                counter_s += 1
                if counter_s > 50:
                    counter_s = 0
                
            #yield
            elif sign == 1 and score > 0.8 or counter_y > 0:
                print("YIELD")
                speed = 0.5

                counter_y += 1
                if counter_y > 50:
                    counter_y = 0
                
            
            #go around
            elif sign == 2 and score > 0.8 or counter_g > 0:
                print("GO AROUND")
                angle = 0.3
                rc.drive.set_speed_angle(speed, angle) # override wall follow angle
                
                counter_g += 1
                if counter_g > 20:
                    counter_g = 0
                
            else:
                # feed in speed angle from wall follower otherwise
                speed = 1
                angle = 0

        
    rc.drive.set_speed_angle(speed, merged_angle)

def update_slow():
    pass

if __name__ == "__main__":
    rc.set_start_update(start, update, update_slow)
    rc.go()
