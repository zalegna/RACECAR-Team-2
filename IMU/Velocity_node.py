"""
MIT BWSI Autonomous RACECAR
MIT License
racecar-neo-summer-labs

File Name: velocity_kalman

Title: velocity calculation node

Author: Annabeth Pan

Purpose: ROS2 node that takes IMU linear acceleration data and integrates it and uses a kalman filter to calculate somewhat
accurate velocity values.

Expected Outcome: Subscribe to the /imu and /mag topics, and publish to the /velocity
topic with accurate velocity estimations.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, MagneticField
from geometry_msgs.msg import Vector3
import numpy as np
import math
import time
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

class VelocityNode(Node):
    def __init__(self):
        super().__init__('kalman_velocity_node')

        # Set up subscriber and publisher nodes
        self.subscription_imu = self.create_subscription(Imu, '/imu', self.imu_callback, 10)
        self.publisher_velocity = self.create_publisher(Vector3, '/velocity', 10) # output as [roll, pitch, yaw] angles

        self.prev_time = self.get_clock().now().nanoseconds / 10**9 # initialize time checkpoint

        self.alpha = .95 # TODO: Determine an alpha value that works with the complementary filter

        # set up velocity params
        self.x_velocity = 0.0
        self.y_velocity = 0.0
        self.z_velocity = 0.0

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

        self.new_x = self.prev_x

        # Print results for sanity checking
        # print(f"====== Results ======")
        # if dt != 0:
        #     print(f"Speed || Freq = {round(1/dt,0)} || dt (ms) = {round(dt*1e3, 2)}")
        # else:
        #     print(f"dt = 0!!!!")
        # print("\n")
    
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
            print(f"Mag norm (~50 uT): {math.sqrt(mx**2 + my**2 + mz**2) * 1e6}") # used for checking magnetic disturbances/offsets
            # now for the magic:
            by = -mz*np.sin(self.pitch) + my*np.cos(self.pitch)
            bx = mz*np.sin(self.roll)*np.cos(self.pitch) + my*np.sin(self.roll)*np.sin(self.pitch) + mx*np.cos(self.pitch)
            mag_accel_yaw = np.arctan2(-by, bx)
        else:
            mag_accel_yaw = self.yaw
        
        # Fuse gyro, mag, and accel derivations in complemtnary filter
        self.roll = self.alpha*gyro_roll + (1-self.alpha)*accel_roll
        self.pitch = self.alpha*gyro_pitch + (1-self.alpha)*accel_pitch
        self.yaw = (self.alpha*gyro_yaw + (1-self.alpha)*mag_accel_yaw)

        ## START OF VELOCITY EXCLUSIVE PROCESSING ##
        
        # calculate gravity vector based on attitude, then remove it from linear acceleration
        g = 9.81
        gravity = np.array([ 
            -g * np.sin(self.roll),
            g * np.sin(self.roll) * np.cos(self.pitch),
            g * np.cos(self.roll) * np.cos(self.pitch)
        ])
        no_gravity = accel - gravity

        # calc new xyz velocity by integrating (trapezoid sum not rectangles i forget the name)
        # the accel_ is there because these are the points calculated using the accelerometer
        # however, these are velocity values
        accel_velocity_x = self.x_velocity + 0.5 * (no_gravity.x + self.prev_accel_x) * dt
        accel_velocity_y = self.y_velocity + 0.5 * (no_gravity.y + self.prev_accel_y) * dt
        accel_velocity_z = self.z_velocity + 0.5 * (no_gravity.z + self.prev_accel_z) * dt

        
    
    # [FUNCTION] Called when magnetometer topic receives an update
    def mag_callback(self, data):
        # TODO: Assign self.mag to the magnetometer data points
        self.mag = (data.magnetic_field.x, data.magnetic_field.y, data.magnetic_field.z,)

def main():
    rclpy.init(args=None)
    node = VelocityNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
