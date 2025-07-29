"""
MIT BWSI Autonomous RACECAR
MIT License
racecar-neo-prereq-labs

File Name: template.py << [Modify with your own file name!]

Title: [PLACEHOLDER] << [Modify with your own title]

Author: [PLACEHOLDER] << [Write your name or team name here]

Purpose: [PLACEHOLDER] << [Write the purpose of the script here]

Expected Outcome: [PLACEHOLDER] << [Write what you expect will happen when you run
the script.]
"""

########################################################################################
# Imports
########################################################################################

import sys

# If this file is nested inside a folder in the labs folder, the relative path should
# be [1, ../../library] instead.
sys.path.insert(0, '../../library')
import racecar_core
import racecar_utils as rc_utils
import numpy as np
import cv2 as cv
import math

########################################################################################
# Global variables
########################################################################################

rc = racecar_core.create_racecar()

# Declare any global variables here
lk_params = None
feature_params = None
prev_lidar_map = None
prev_lidar_corners = None
mean_disp = None
yaw_fixed = 0.0

RES_PER_DEGREE = 2
# my ettiquite is dog sorry lol

class OneDKalman:
    def __init__(self, process_var=0.01, sensor_var=1.0):
        # Q = process noise variance, R = measurement noise variance
        self.Q = process_var
        self.R = sensor_var
        # state estimate and covariance
        self.x = 0.0
        self.P = 1.0

    def update(self, z: float) -> float:
        # 1) compute Kalman gain
        K = self.P / (self.P + self.R)
        # 2) update estimate
        self.x = self.x + K * (z - self.x)
        # 3) update covariance
        self.P = (1 - K) * self.P + self.Q
        return self.x

########################################################################################
# Functions
########################################################################################

# [FUNCTION] The start function is run once every time the start button is pressed
def start():
    global lk_params, feature_params, prev_lidar_corners, prev_lidar_map, mean_disp, alpha, roll
    global pitch, yaw, prev_accel_x, prev_accel_y, prev_accel_z, mag, x_velocity, y_velocity, z_velocity
    global initial_yaw, yaw_fixed, kf_ax, kf_ay, kf_az

    alpha = .7
    initial_yaw = 0

    # intialize gyro stuff
    roll = 0.0
    pitch = 0.0
    yaw = 0.0

    prev_accel_x = 0.0
    prev_accel_y = 0.0
    prev_accel_z = 0.0

    mag = None

    # set up velocity
    x_velocity = 0.0
    y_velocity = 0.0
    z_velocity = 0.0

    # lidar stuff
    # params for ShiTomasi corner detection
    feature_params = dict( maxCorners = 30,
                        qualityLevel = 0.7,
                        minDistance = 7,
                        blockSize = 7 )
    
    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (30, 30),
                    maxLevel = 2,
                    criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
    
    prev_lidar_map = cv.cvtColor(np.zeros((480, 640, 3), dtype=np.uint8), cv.COLOR_BGR2GRAY) # blank image

    prev_lidar_corners = cv.goodFeaturesToTrack(prev_lidar_map, mask=None, **feature_params)
    mean_disp = 0

    ### Kalman Filter
    kf_ax = OneDKalman(process_var=3.0, sensor_var=0.01)
    kf_ay = OneDKalman(process_var=3.0, sensor_var=0.02)
    kf_az = OneDKalman(process_var=1.0, sensor_var=0.01)


def get_lidar_image(scan, yaw_shift):
    # shift scan based on yaw
    yaw_shift = 0
    angle_offset = int(round(yaw_shift))
    shifted_scan = np.concatenate((scan[angle_offset*RES_PER_DEGREE:], scan[:angle_offset*RES_PER_DEGREE]))
    #shifted_scan = scan
    
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
    img = rc_utils.crop(img, (0, 0), (160, 240))
    #print(img.shape) : (240, 240, 3)
    return img
    
def angle_fix(angle_rad):
    if angle_rad < 0:
        return angle_rad + 2*math.pi
        # return angle_rad
    else:
        return angle_rad
    
# [FUNCTION] After start() is run, this function is run once every frame (ideally at
# 60 frames per second or slower depending on processing speed) until the back button
# is pressed  
def update():
    global lk_params, feature_params, prev_lidar_corners, prev_lidar_map, mean_disp, alpha, roll
    global pitch, yaw, prev_accel_x, prev_accel_y, prev_accel_z, mag, x_velocity, y_velocity, z_velocity
    global initial_yaw, kf_ax, kf_ay, kf_az
    accel = rc.physics.get_linear_acceleration()
    gyro = rc.physics.get_angular_velocity()
    mag = rc.physics.get_magnetic_field()

    # Calculate time delta
   #now = 
    dt = rc.get_delta_time() # Time delta
    #prev_time = now # refresh checkpoint

    # TODO: Derive tilt angles from accelerometer
    accel_roll = np.arctan2(accel[1], np.sqrt(accel[0]**2 + accel[2]**2)) # theta_x
    accel_pitch = np.arctan2(-accel[0], np.sqrt(accel[1]**2 + accel[2]**2)) # theta_y - seems correct

    # TODO: Integrate gyroscope to get attitude angles
    gyro_roll = (roll + gyro[0]*dt) #% 2*np.pi # theta_xt
    gyro_pitch = (pitch + gyro[1]*dt) #% 2*np.pi # theta_yt
    gyro_yaw = (yaw + gyro[2]*dt) #% 2*np.pi # theta_zt

    # TODO: Compute yaw angle from magnetometer
    if mag is not None:
        mx, my, mz = mag
        #print(f"Mag norm (~50 uT): {math.sqrt(mx**2 + my**2 + mz**2) * 1e6}") # used for checking magnetic disturbances/offsets
        by = -mz*np.sin(pitch) + my*np.cos(pitch)
        bx = mz*np.sin(roll)*np.cos(pitch) + my*np.sin(roll)*np.sin(pitch) + mx*np.cos(pitch)
        mag_accel_yaw = np.arctan2(-by, bx)
    else:
        mag_accel_yaw = yaw
    
    # TODO: Fuse gyro, mag, and accel derivations in complemtnary filter
    roll = alpha*gyro_roll + (1-alpha)*accel_roll
    pitch = alpha*gyro_pitch + (1-alpha)*accel_pitch
    yaw = (alpha*gyro_yaw + (1-alpha)*mag_accel_yaw)

    yaw_fixed = angle_fix(yaw) * 180/math.pi
    #print("yaw in degrees:", yaw_fixed)

    ## START OF VELOCITY EXCLUSIVE PROCESSING ##
    
    # calculate gravity vector based on attitude, then remove it from linear acceleration
    g = -9.81
    gravity = np.array([
        g * np.sin(pitch),
        -g * np.sin(roll),
        g * np.cos(pitch) * np.cos(roll)
    ])
    
     # acceleration from vector3 to  np array to do operations, also a bias
    no_gravity = accel - gravity

    # put accel thru kalman
    ax_f=kf_ax.update(float(no_gravity[0]))
    ay_f=kf_ay.update(float(no_gravity[1]))
    az_f=kf_az.update(float(no_gravity[2]))

    if abs(ax_f) < .05:
        ay_f = 0
    if abs(ay_f) < .05:
        ay_f = 0
    if abs(az_f) < .05:
        az_f = 0

    accel_array = np.array([accel[0], accel[1], accel[1]]) # acceleration from vector3 to  np array to do operations
    no_gravity = accel_array - gravity

    # print(f"Accel")
    # print(f"x: {no_gravity[0]}")
    # print(f"y: {no_gravity[1]}")
    # print(f"z: {no_gravity[2]}")
    # calc new xyz velocity by integrating (trapezoid sum not rectangles i forget the name)
    # the accel_ is there because these are the points calculated using the accelerometer
    # however, these are velocity values

    accel_velocity_x = x_velocity + 0.5 * (no_gravity[0] + prev_accel_x) * dt
    accel_velocity_y = y_velocity + 0.5 * (no_gravity[1] + prev_accel_y) * dt
    accel_velocity_z = z_velocity + 0.5 * (no_gravity[2] + prev_accel_z) * dt

    scan = rc.lidar.get_samples()

    if scan is not None:
        lidar_map = get_lidar_image(scan, yaw_fixed)
        lidar_gray = cv.cvtColor(lidar_map, cv.COLOR_BGR2GRAY)
        
        # If this is the first frame, or we lost our corners, just detect and wait
        if  prev_lidar_corners is None or len(prev_lidar_corners) < 4 or prev_lidar_map is None:
            prev_lidar_corners = cv.goodFeaturesToTrack(lidar_gray, mask=None, **feature_params)
            prev_lidar_map = lidar_gray
            
        # Track existing corners
        current_lidar_corners, st, err = cv.calcOpticalFlowPyrLK(
            prev_lidar_map,
            lidar_gray,
            prev_lidar_corners,
            None,
            **lk_params
        )

        if current_lidar_corners is not None and st is not None:
            good_new = current_lidar_corners[st == 1]
            good_old = prev_lidar_corners[st == 1]

            if len(good_new) >= 1:
                displacements = good_new - good_old
                mean_disp = np.mean(displacements, axis=0)
                if abs(np.sum(mean_disp)) > .001:
                    print(f"Mean optical flow dx, dy: {mean_disp.flatten()}")
                    pass
                    
                # Update for next round
                prev_lidar_map = lidar_gray
                prev_lidar_corners = good_new.reshape(-1, 1, 2)
            else:
                # If all tracking lost, reinitialize
                prev_lidar_corners = cv.goodFeaturesToTrack(lidar_gray, mask=None, **feature_params)
                prev_lidar_map = lidar_gray
        else:
            # If flow failed, reinitialize
            prev_lidar_corners = cv.goodFeaturesToTrack(lidar_gray, mask=None, **feature_params)
            prev_lidar_map = lidar_gray

    # draw the tracks
    mask = np.zeros_like(lidar_map)  # 3-channel for color drawing
    frame = lidar_map.copy()

    color = np.random.randint(0, 255, (100, 3))
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
        frame = cv.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
    img = cv.add(frame, mask)

    # movement
    angle = 0
    rt = rc.controller.get_trigger(rc.controller.Trigger.RIGHT)
    lt = rc.controller.get_trigger(rc.controller.Trigger.LEFT)
    if rc.controller.is_down(rc.controller.Button.RB):
        angle = 1
    elif rc.controller.is_down(rc.controller.Button.LB):
        angle = -1
    speed = rt - lt

    flow_magnitude = np.sqrt(mean_disp[0]**2 + mean_disp[1]**2)
    scale = 1
    lidar_velocity_x = mean_disp[0]*scale
    lidar_velocity_y = mean_disp[1]*scale

    print(round(lidar_velocity_x, 2))
    print(round(lidar_velocity_y, 2))
    print("\n")

    #print(round(x_velocity, 2), round(y_velocity, 2))
            # Print results for sanity checking
    # print(f"====== Complementary Filter Results ======")
    # if dt != 0:
    #     print(f"Speed || Freq = {round(1/dt,0)} || dt (ms) = {round(dt*1e3, 2)}")
    # else:
    #     print(f"dt = 0!!!!")
    # print(f"Accel + Mag Derivation")
    # print(f"Roll (deg): {accel_roll * 180/math.pi}")
    # print(f"Pitch (deg): {accel_pitch * 180/math.pi}")
    # print(f"Yaw (deg): {mag_accel_yaw * 180/math.pi}")
    # print()
    # print(f"Gyro Derivation")
    # print(f"Roll (deg): {gyro_roll * 180/math.pi}")
    # print(f"Pitch (deg): {gyro_pitch * 180/math.pi}")
    # print(f"Yaw (deg): {gyro_yaw * 180/math.pi}")
    # print()
    # print(f"Fused Results")
    # print(f"Roll (deg): {roll * 180/math.pi}")
    # print(f"Pitch (deg): {pitch * 180/math.pi}")
    # print(f"Yaw (deg): {yaw * 180/math.pi}")
    # print("\n")

    rc.display.show_color_image(img)
    rc.drive.set_speed_angle(speed, angle)

# [FUNCTION] update_slow() is similar to update() but is called once per second by
# default. It is especially useful for printing debug messages, since printing a 
# message every frame in update is computationally expensive and creates clutter
def update_slow():
    pass # Remove 'pass and write your source code for the update_slow() function here


########################################################################################
# DO NOT MODIFY: Register start and update and begin execution
########################################################################################

if __name__ == "__main__":
    rc.set_start_update(start, update, update_slow)
    rc.go()
