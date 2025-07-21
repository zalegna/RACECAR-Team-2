"""
MIT BWSI Autonomous RACECAR
MIT License
racecar-neo-summer-labs

File Name: PoseNode.py

Title: pose calculation node

Author: Annabeth Pan

Purpose: ROS2 node that takes IMU linear acceleration data and integrates it and uses a kalman filter to calculate somewhat
accurate velocity values. then does integration and kalman to get velocity to position.
Also does roll-pitch-yaw calcs for gravity vector elimination and to get yaw (theta) for pose

Expected Outcome: Subscribe to the /imu and /mag topics, and publish to the /pose
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
        self.publisher_pose = self.create_publisher(Vector3, '/pose', 10) # output as [roll, pitch, yaw] angles
        self.subscription_mag = self.create_subscription(MagneticField, '/mag', self.mag_callback, 10)

        self.prev_time = self.get_clock().now().nanoseconds / 10**9 # initialize time checkpoint

        self.alpha = .95

        # intialize gyro things
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0

        # initialize previous acceleratoins
        self.prev_accel_x = 0.0
        self.prev_accel_y = 0.0
        self.prev_accel_z = 0.0

        self.mag = None

        # set up velocity params
        self.x_velocity = 0.0
        self.y_velocity = 0.0
        self.z_velocity = 0.0

        self.prev_vel_x = 0.0
        self.prev_vel_y = 0.0
        self.prev_vel_z = 0.0

        # pose shit
        self.x_pos = 0.0
        self.y_pos = 0.0

        # initialize kalman stuff
        self.xvf = self.create_kalman_filter()
        self.yvf = self.create_kalman_filter()
        self.zvf = self.create_kalman_filter()

        self.xpf = self.create_kalman_filter()
        self.ypf = self.create_kalman_filter()

    def create_kalman_filter(self, dt=0.1, var=0.13):
        kf = KalmanFilter(dim_x=2, dim_z=1)
        kf.x = np.array([0., 0.])
        kf.F = np.array([[1., dt], [0., 1.]])
        kf.H = np.array([[0., 1.]])
        kf.P = np.eye(2) * 1000.
        kf.R = 1
        kf.Q = Q_discrete_white_noise(dim=2, dt=dt, var=var)
        return kf

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
            g * np.sin(self.pitch),                                    # X
            -g * np.sin(self.roll) * np.cos(self.pitch),               # Y
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

        # i give up KALMAN TIME
        xz = accel_velocity_x
        self.xvf.predict()
        self.xvf.update([[xz]])

        yz = accel_velocity_y
        self.yvf.predict()
        self.yvf.update([[yz]])

        zz = accel_velocity_z
        self.zvf.predict()
        self.zvf.update([[zz]])

        # ADD ANOTHER COMP FILTER HERE? SOMEHOW?? MAYBE???

        # make current values into the previous values for the next iteration :D
        self.prev_accel_x = no_gravity[0] # pure linear acceleration
        self.prev_accel_y = no_gravity[1]
        self.prev_accel_z = no_gravity[2]

        self.prev_vel_x = self.xvf.x[1]
        self.prev_vel_x = self.yvf.x[1]
        self.prev_vel_x = self.zvf.x[1]
        
        # push it to values
        self.x_velocity = self.xvf.x[1] # velocity after kalmans
        self.y_velocity = self.yvf.x[1]
        self.z_velocity = self.zvf.x[1]

        ## POSE ESTIMATION: POSITION + YAW ##

        # integration (discrete)
        xz_pos = self.x_pos + 0.5 * (self.x_velocity + self.prev_vel_x) * dt
        yz_pos = self.y_pos + 0.5 * (self.y_velocity + self.prev_vel_y) * dt

        # kalman time
        self.xpf.predict()
        self.xpf.update([[xz_pos]])

        self.ypf.predict()
        self.ypf.update([[yz_pos]])

        # make current values into the previous values for the next iteration :D
        self.prev_vel_x = self.x_velocity
        self.prev_vel_y = self.y_velocity
        self.prev_vel_z = self.z_velocity
        
        # assign
        self.x_pos = self.xvf.x[1] # position after kalmans
        self.y_pos = self.yvf.x[1]

        # publishing time
        pose = Vector3()
        pose.x = self.x_pos
        pose.y = self.y_pos
        pose.z = self.yaw # ok chat i KNOW this isnt a proper vector3 but im not thinking ab 4d to do this
        self.publisher_pose.publish(pose)

    # [FUNCTION] Called when magnetometer topic receives an update
    def mag_callback(self, data):
        # Assign self.mag to the magnetometer data points
        self.mag = (data.magnetic_field.x, data.magnetic_field.y, data.magnetic_field.z,)
        

def main():
    rclpy.init(args=None)
    node = VelocityNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
