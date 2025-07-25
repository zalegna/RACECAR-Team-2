"""
MIT BWSI Autonomous RACECAR
MIT License
racecar-neo-summer-labs

File Name: velocity_kalman

Title: velocity calculation node

Author: Annabeth Pan

Purpose: ROS2 node that takes IMU linear acceleration data and integrates it and fuses it with velocity
data derived from measuring optical flow in an image from the LIDAR. I might add a Kalman filter after
this at some point but idk 

Expected Outcome: Subscribe to the /imu and /mag topics, and publish to the /velocity
topic with accurate velocity estimations.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, MagneticField, LaserScan
from geometry_msgs.msg import Vector3
import numpy as np
import math
import cv2 as cv

class VelocityNode(Node):
    def __init__(self):
        super().__init__('velocity_node')

        # Set up subscriber and publisher nodes
        self.subscription_imu = self.create_subscription(Imu, '/imu', self.imu_callback, 10)
        self.publisher_velocity = self.create_publisher(Vector3, '/velocity', 10) # output as [roll, pitch, yaw] angles
        self.subscription_mag = self.create_subscription(MagneticField, '/mag', self.mag_callback, 10)
        self.subscription_lidar = self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)

        self.prev_time = self.get_clock().now().nanoseconds / 10**9 # initialize time checkpoint

        self.alpha = .95

        # intialize gyro stuff
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        
        self.roll_fixed = 0.0
        self.pitch_fixed = 0.0
        self.yaw_fixed = 0.0

        self.prev_accel_x = 0.0
        self.prev_accel_y = 0.0
        self.prev_accel_z = 0.0

        self.mag = None

        # set up velocity
        self.x_velocity = 0.0
        self.y_velocity = 0.0
        self.z_velocity = 0.0

        # lidar stuff
        # params for ShiTomasi corner detection
        self.feature_params = dict( maxCorners = 100,
                            qualityLevel = 0.3,
                            minDistance = 7,
                            blockSize = 7 )
        
        # Parameters for lucas kanade optical flow
        self.lk_params = dict( winSize  = (30, 30),
                        maxLevel = 2,
                        criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
        
        self.prev_lidar_map = cv.cvtColor(np.zeros((480, 640, 3), dtype=np.uint8), cv.COLOR_BGR2GRAY) # blank image

        self.prev_lidar_corners = cv.goodFeaturesToTrack(self.prev_lidar_map, mask=None, **self.feature_params)
        self.mean_disp = 0

        # inputs neg/pos angle, outputs necessarily pos angle (all in rad)
    def angle_fix(self, angle_rad):
        if angle_rad < 0:
            return angle_rad + 2*math.pi
            # return angle_rad
        else:
            return angle_rad
            
    # [FUNCTION] Called when new IMU data is received, attidude calc completed here as well
    def imu_callback(self, data):
        
        ## THE FIRST BIT IS GETTING ATTITUDE, NECESSARY FOR GRAVITY

        # Grab linear acceleration and gyroscope values from subscribed data points
        accel = data.linear_acceleration
        gyro = data.angular_velocity

        # Calculate time delta
        now = self.get_clock().now().nanoseconds / 10**9 # Current ROS time
        dt = now - self.prev_time # Time delta
        self.prev_time = now # refresh checkpoint
    
        # Derive tilt angles from accelerometer
        accel_roll = np.arctan2(accel.y, np.sqrt(accel.x**2 + accel.z**2)) # theta_x
        accel_pitch = np.arctan2(-accel.x, np.sqrt(accel.y**2 + accel.z**2)) # theta_y - seems correct

        # Integrate gyroscope to get attitude angles
        gyro_roll = (self.roll + gyro.x*dt) #% 2*np.pi # theta_xt
        gyro_pitch = (self.pitch + gyro.y*dt) #% 2*np.pi # theta_yt
        gyro_yaw = (self.yaw + gyro.z*dt) #% 2*np.pi # theta_zt

        # Compute yaw angle from magnetometer
        if self.mag:
            mx, my, mz = self.mag
            #print(f"Mag norm (~50 uT): {math.sqrt(mx**2 + my**2 + mz**2) * 1e6}") # used for checking magnetic disturbances/offsets
            # now for the magic:
            by = -mz*np.sin(self.pitch) + my*np.cos(self.pitch)
            bx = mz*np.sin(self.roll)*np.cos(self.pitch) + my*np.sin(self.roll)*np.sin(self.pitch) + mx*np.cos(self.pitch)
            mag_accel_yaw = np.arctan2(-by, bx)
        else:
            mag_accel_yaw = self.yaw
        
        # Fuse gyro, mag, and accel derivations in complemtnary filter
        self.roll = self.alpha*gyro_roll + (1-self.alpha)*accel_roll
        self.pitch = self.alpha*gyro_pitch + (1-self.alpha)*accel_pitch
        self.yaw = self.alpha*gyro_yaw + (1-self.alpha)*mag_accel_yaw

        # self.roll_fixed = self.angle_fix(self.roll)
        # self.pitch_fixed = self.angle_fix(self.pitch)
        self.yaw_fixed = self.angle_fix(self.yaw * 180/math.pi) # eliminate negative angles

        ## START OF VELOCITY EXCLUSIVE PROCESSING ##
        
        # calculate gravity vector based on attitude, then remove it from linear acceleration
        g = 9.81
        gravity = np.array([ 
            g * np.sin(self.pitch),                                    # X
            -g * np.sin(self.roll),                                    # Y
            -g * np.cos(self.roll) * np.cos(self.pitch)                # Z
        ])

        accel_array = np.array([accel.x, accel.y, accel.z]) # acceleration from vector3 to  np array to do operations
        no_gravity = accel_array - gravity

        # calc new xyz velocity by integrating (trapezoid sum not rectangles i forget the name)
        # the accel_ is there because these are the points calculated using the accelerometer
        # however, these are velocity values
        accel_velocity_x = self.x_velocity + 0.5 * (no_gravity[0] + self.prev_accel_x) * dt
        accel_velocity_y = self.y_velocity + 0.5 * (no_gravity[1] + self.prev_accel_y) * dt
        accel_velocity_z = self.z_velocity + 0.5 * (no_gravity[2] + self.prev_accel_z) * dt

        print(f"velocityx: {accel_velocity_x}")
        print(f"velocityy: {accel_velocity_y}")
        print(f"velocityz: {accel_velocity_z}")

        # lidar optical flow velocity finding method starts hereee

         # if scan data is ok, collect map and do everyting else
        if hasattr(self, 'scan_data') and self.scan_data is not None:
            lidar_map = self.get_lidar_image(self.scan_data, self.yaw_fixed) # centers around yaw = 0, in degrees
            lidar_gray = cv.cvtColor(lidar_map, cv.COLOR_BGR2GRAY)

            # If this is the first frame, or we lost our corners, detect again
            if self.prev_lidar_corners is None or len(self.prev_lidar_corners) < 10 or self.prev_lidar_map is None:
                self.prev_lidar_corners = cv.goodFeaturesToTrack(lidar_gray, mask=None, **self.feature_params)
                self.prev_lidar_map = lidar_gray
            
            # Track existing corners
            current_lidar_corners, st, err = cv.calcOpticalFlowPyrLK(
                self.prev_lidar_map,
                lidar_gray,
                self.prev_lidar_corners,
                None,
                **self.lk_params
            )

            if current_lidar_corners is not None and st is not None:
                good_new = current_lidar_corners[st == 1]
                good_old = self.prev_lidar_corners[st == 1]

                if len(good_new) >= 1:
                    displacements = good_new - good_old
                    self.mean_disp = np.mean(displacements, axis=0)
                    if abs(np.sum(self.mean_disp)) > .001:
                        print(f"Mean optical flow dx, dy: {self.mean_disp.flatten()}")
                        
                    # Update for next round
                    self.prev_lidar_map = lidar_gray
                    self.prev_lidar_corners = good_new.reshape(-1, 1, 2)
                else:
                    # If all tracking lost, reinitialize
                    self.prev_lidar_corners = cv.goodFeaturesToTrack(lidar_gray, mask=None, **self.feature_params)
                    self.prev_lidar_map = lidar_gray
            else:
                # If flow failed, reinitialize
                self.prev_lidar_corners = cv.goodFeaturesToTrack(lidar_gray, mask=None, **self.feature_params)
                self.prev_lidar_map = lidar_gray

            # draw the tracks
            color = np.random.randint(0, 255, (100, 3))
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
                frame = cv.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
            img = cv.add(frame, mask)

        ## calculating velocities from optical flow stuff
        lidar_velocity_x = self.mean_disp
        lidar_velocity_y = self.mean_disp

        if np.sum(self.mean_disp) != 0:
            velocity = Vector3()
            velocity.x = self.x_velocity
            velocity.y = self.y_velocity
            velocity.z = self.z_velocity
            self.publisher_velocity.publish(velocity)

    # [FUNCTION] Called when magnetometer topic receives an update
    def mag_callback(self, data):
        # Assign self.mag to the magnetometer data points
        self.mag = (data.magnetic_field.x, data.magnetic_field.y, data.magnetic_field.z,)

    def lidar_callback(self, data):
        scan_data_orig = np.flip(np.multiply(np.array(data.ranges), 100))
        self.scan_data = np.array([0 if str(x) == "inf" else x for x in scan_data_orig])

    def get_lidar_image(self, scan, yaw_shift):
        # shift scan based on yaw
        angle_offset = int(yaw_shift % 1) # make int
        shifted_scan = np.concatenate(scan[angle_offset:], scan[:angle_offset])

        # Convert polar to Cartesian
        angles_deg = np.arange(0, 360, .5)
        angles_rad = np.deg2rad(angles_deg)
        distances = np.array(shifted_scan)

        # Filter valid points
        valid = distances > 0
        distances = distances[valid]
        angles_rad = angles_rad[valid]

        x = distances * np.sin(angles_rad)
        y = distances * np.cos(angles_rad)

        # Create blank image
        img = np.zeros((240, 240, 3), dtype=np.uint8)

        # Transform points to fit image coords (center at 320,400 and scale)
        scale = 0.2  # adjust for zoom
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
        return img
        
def main():
    rclpy.init(args=None)
    node = VelocityNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
