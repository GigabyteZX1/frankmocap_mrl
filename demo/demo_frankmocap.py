#!/usr/bin/env python3

""" 
This script defines a ROS node that connects to a socket server to receive transformation matrices, 
converts them to ROS `TransformStamped` messages, and publishes them to a specified topic.
The node continuously reads data from the socket, processes the received JSON-encoded transformation matrices,
and publishes the corresponding transformations.

Author: Emaad Ahmed
Date: March 2025
"""

import rospy
import socket
import json
import numpy as np
import threading
import sys
import errno
from scipy.spatial.transform import Rotation as R
from Cobot import Cobot
from cobo_msgs.msg import Trajectory, JointState
from TrajectoryPlanners import LinearTrajectoryPlanner, CubicTrajectoryPlanner

from collections import deque
# import matplotlib
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
# matplotlib.use('TkAgg')
# import pyqtgraph as pg
# from pyqtgraph.Qt import QtCore, QtWidgets

#Plotter class
class RealTimePlotter:
    def __init__(self, telekinesis_instance):
        self.telekinesis = telekinesis_instance

        self.data_keys = ['x', 'y', 'z', 'phi', 'psi']
        self.colors = ['r', 'g', 'b', 'm', 'c']
             
        self.fig, self.ax = plt.subplots(figsize=(8, 6))

        self.pose_lines = {}
        
        for key, color in zip(self.data_keys, self.colors):
            line, = self.ax.plot([], [], label=key, color=color)
            self.pose_lines[key] = line

        self.ax.set_xlim(0, 100)
        self.ax.set_ylim(-1.5, 1.5)
        self.ax.set_title("Real-time EE Pose")
        self.ax.set_xlabel("Time steps")
        self.ax.set_ylabel("Value")
        self.ax.legend()        

        self.ani = FuncAnimation(self.fig, self.update_plot, interval=200)
        plt.ion()
        plt.tight_layout()
        plt.show()

    def update_plot(self, frame):
        for key in self.data_keys:
            data = self.telekinesis.ee_pose_history[key][-100:]  # Only keep latest 100
            x_vals = list(range(len(data)))
            self.pose_lines[key].set_data(x_vals, data)
        
        self.ax.relim()
        self.ax.autoscale_view()
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


class KalmanFilter:
    def __init__(self, process_variance=1e-4, measurement_variance=1e-1):
        self.A = np.eye(5)  # State transition matrix
        self.H = np.eye(5)  # Measurement matrix
        self.Q = np.eye(5) * process_variance  # Process noise covariance
        self.R = np.eye(5) * measurement_variance  # Measurement noise covariance
        self.P = np.eye(5)  # Covariance matrix
        self.x = np.zeros(5)  # State estimate

    def apply(self, measurement):
        """
        Apply Kalman filtering to smooth the input measurement.
        """
        # Prediction step
        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q

        # Measurement update step
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman Gain
        y = measurement - (self.H @ self.x)  # Innovation
        self.x = self.x + K @ y
        self.P = (np.eye(5) - K @ self.H) @ self.P  # Update covariance

        return self.x

class CobotTelekinesis:
    def __init__(self, host='localhost', port=60001):
        self.host = host
        self.port = port
        self.sock = None
        self.buffer = b""
        self.connected = False
        self.lock = threading.Lock()
        self.cobot = Cobot()

        # ROS publisher for TransformStamped messages.
        self.pub = rospy.Publisher('/cobo/joint_command', JointState, queue_size=1)
        self.joint_state_act_sub = rospy.Subscriber('/cobo/joint_state_act', JointState, self.joint_state_callback)

        # Initialize the current joint states.
        self.current_joints = np.zeros(6)
        self.joint_velocities = np.zeros(6)
        self.home_joints = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        # Motion parameters
        self.default_duration = 3.0  # seconds
        self.default_dt = 0.002  # seconds
        self.max_velocity = np.array([5, 5, 5, 1, 1, 1])  # rad/s

        # Trajectory execution
        self.current_trajectory = None
        self.trajectory_index = 0
        self.trajectory_active = False
        self.control_rate = 500  # Hz
        self.current_ee_pose = None

        self.first_point_flag = False

        # Planner
        self.planner_type = "cubic"  # or "linear"

        # Plotter parameters
        self.filter_window = 5
        self.ee_pose_buffers = {
           'x': deque(maxlen=self.filter_window),
           'y': deque(maxlen=self.filter_window),
           'z': deque(maxlen=self.filter_window),
           'phi': deque(maxlen=self.filter_window),
           'psi': deque(maxlen=self.filter_window)}

        self.ee_pose_history = {
         'x': [],
        'y': [],
        'z': [],
        'phi': [],
        'psi': []}
        
        # Kalman Filter
        self.kalman_filter = KalmanFilter(process_variance=1e-3)

    def connect_socket(self, max_retries=5, retry_delay=2):
        """
        Connect to the server using UDP protocol.
        """
        retries = 0
        while retries < max_retries and not rospy.is_shutdown():
            try:
                # Create UDP socket
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                
                self.sock.connect((self.host, self.port))
                
                # Send an initial packet to establish communication
                self.sock.send(b"client_hello\n")
                
                self.connected = True
                rospy.loginfo(f"UDP client ready, sending to {self.host}:{self.port}")
                return True
                
            except socket.error as e:
                rospy.logwarn(f"Connection attempt {retries+1}/{max_retries} failed: {e}")
                retries += 1
                rospy.sleep(retry_delay)
                
        rospy.logerr("Failed to establish UDP communication with server")
        return False


    def joint_state_callback(self, msg):
        """Callback to update current joint states."""
        self.current_joints = np.array(msg.position)

    def move_to_home(self, joint_pos = None):
        """Move the robot to the home position."""

        if not np.any(joint_pos):
            rospy.loginfo("Retargetting completed. Moving to home position.")
            joint_pos = self.home_joints

        # Create trajectory to home position
        if self.planner_type == "cubic":
            planner = CubicTrajectoryPlanner(
                self.current_joints, 
                joint_pos, 
                self.default_duration,
                self.default_dt,
                self.max_velocity
            )
        else:
            planner = LinearTrajectoryPlanner(
                self.current_joints, 
                joint_pos, 
                self.default_duration,
                self.default_dt,
                max_velocity=self.max_velocity,
                use_trapezoidal=False 
            )
        
        # Generate and execute trajectory
        self.execute_trajectory(planner.generate_trajectory())

        if not self.trajectory_active and not rospy.is_shutdown():
            rospy.loginfo("Reached Home")

    def execute_trajectory(self, trajectory):
        """Prepare trajectory for execution."""
        self.current_trajectory = trajectory
        self.trajectory_index = 0
        self.trajectory_active = True

        if self.current_trajectory is None:
            return

        rospy.loginfo("Executing trajectory...")
        # Get current trajectory point
        while self.trajectory_index < len(self.current_trajectory.time_points):
            joint_state_ref = JointState()
            joint_state_ref.mode = 1
            joint_state_ref.position = self.current_trajectory.positions[:, self.trajectory_index].tolist()
            joint_state_ref.velocity = self.current_trajectory.velocities[:, self.trajectory_index].tolist()
            joint_state_ref.effort = np.zeros_like(self.home_joints).tolist()
            
            self.pub.publish(joint_state_ref)
            self.trajectory_index += 1
            rospy.sleep(1.0 / self.control_rate)

        # Trajectory complete
        self.trajectory_active = False
        self.current_trajectory = None

    def read_loop(self):
        """
        Continuously read data from the UDP socket. Messages are assumed to be newline-delimited.
        """
        while not rospy.is_shutdown() and self.connected:
            try:
                # Set a timeout to allow checking for shutdown
                self.sock.settimeout(1.0)
                
                try:
                    # Receive data with sender address
                    data, server_address = self.sock.recvfrom(1024)
                    
                    # Store server address if needed
                    if not hasattr(self, 'server_address') or not self.server_address:
                        self.server_address = server_address
                        rospy.loginfo(f"Connected to server at {server_address}")
                    
                    if not data:
                        continue
                    
                    # Check if this is a disconnect command
                    try:
                        # Parse the JSON message
                        message = json.loads(data.decode('utf-8').strip())
                        if isinstance(message, dict) and message.get("command") == "CLOSE_CONNECTION":
                            rospy.loginfo("Received disconnect command from server")
                            self.connected = False
                            self.move_to_home()
                            break
                    except json.JSONDecodeError:
                        pass
                    
                    # Process as regular data
                    with self.lock:
                        self.buffer += data
                    
                    # Process all complete messages in the buffer
                    while b'\n' in self.buffer:
                        with self.lock:
                            line, self.buffer = self.buffer.split(b'\n', 1)
                        if line:
                            self.process_message(line)
                
                except socket.timeout:
                    # Timeout is used just to allow checking for shutdown
                    continue
                    
            except Exception as e:
                import errno
                rospy.logerr(f"Error in UDP socket read loop: {e}")
                # For other errors, log but continue
                rospy.logwarn(f"Continuing after error: {e}")
        
        self.sock.close()
        rospy.loginfo("Disconnected from server, socket closed")
            
    def process_message(self, message):
        """
        Deserialize the JSON message
        """
        try:
            # Decode the message from bytes to string and then load it as JSON.
            transform_list = json.loads(message.decode('utf-8'))
            relative_transform = np.array(transform_list)

            # Extract Gripper State
            gripper_state = relative_transform[3,3]
            relative_transform[3, 3] = 1 # Reverting to original value

            if relative_transform.shape != (4, 4):
                rospy.logwarn("Received matrix does not have shape (4, 4). Ignoring message.")
                return
            
            print("\n=== [DEBUG] New Relative Transform Received ===")
            print("Relative Transform:\n", relative_transform)
            print("Current Joint States:", self.current_joints)
            self.current_ee_pose = self.cobot.forward_kinematics(self.current_joints[:5])
            print("Current Robot EE Pose:", self.current_ee_pose)

            # Clamping factor as per the max reach of cobot
            robot_max_reach = 0.811

            T_robot_torso = np.eye(4)
            T_robot_torso[2,3] = -0.1 # Adjusting Robot Torso Height
            T_robot_torso[1,3] = 0.15 # Adjusting Robot Torso Position
            T_target = T_robot_torso @ relative_transform

            ee_target_pos = self.extract_target_pose(T_target)
            print("Target EE Pose:", ee_target_pos)

            # Applying Kalman Filter to smooth the target pose
            ee_target_pos = self.kalman_filter.apply(ee_target_pos)

            xyz = ee_target_pos[:3]

            clamped_xyz = np.clip(xyz, -robot_max_reach, robot_max_reach)
            ee_target_pos[:3] = clamped_xyz
            if not np.allclose(xyz, clamped_xyz):
                rospy.logwarn(f"[Clamped] Target EE Pose exceeded robot max reach on one or more axes. Clamped to {clamped_xyz}")

            x, y, z, phi, psi = ee_target_pos
            self.ee_pose_history['x'].append(x)
            self.ee_pose_history['y'].append(y)
            self.ee_pose_history['z'].append(z)
            self.ee_pose_history['phi'].append(phi)
            self.ee_pose_history['psi'].append(psi)

            #Solve IK
            ik_result = self.cobot.inverse_kinematics(ee_target_pos)
            if ik_result is None:
               rospy.logwarn("IK failed for pose:", ee_target_pos)
               return
            
            ik_result[4] = np.pi/2
            ik_result = np.append(ik_result, 0.0 if gripper_state == 1 else 1.0) # Adding gripper angle
            print("IK Result:", np.hstack(ik_result))

            if not self.first_point_flag:
                rospy.loginfo("Moving to First point..")
                self.move_to_home(np.vstack(ik_result))
                self.first_point_flag = True
                # rospy.sleep(1)

            target_msg = JointState()
            target_msg.mode = 1
            target_msg.position = np.vstack(ik_result)
            target_msg.velocity = np.array([0.1]*5)
            target_msg.effort = np.zeros(5)

            self.pub.publish(target_msg)
            print(target_msg)
            print("Published JointState to robot.")
            
        except Exception as e:
            rospy.logerr("Failed to process message: {}".format(e))
    
    def extract_target_pose(self, T_target_robot):
        """
        Extract target x, y, z, phi, psi from the 4x4 homogeneous transformation matrix.
        """
        # Extract Position
        x = T_target_robot[0, 3]
        y = T_target_robot[1, 3]
        z = T_target_robot[2, 3]
        
        # Extract Rotation Matrix
        R_target = T_target_robot[:3, :3]

        # Reversed
        psi = np.arctan2(R_target[2, 1], R_target[2, 2])
        phi = np.arctan2(-R_target[2, 0], np.sqrt(R_target[2, 1]**2 + R_target[2, 2]**2))

        # Convert Radians to Degrees
        # phi = np.degrees(phi)
        psi = np.degrees(psi)
        print("raw_phi: {:.2f}, raw_psi: {:.2f}".format(phi, psi))
        
        # Temporary fix for phi and psi
        # phi = 0
        psi = 0
        
        return np.array([x, y, z, phi, psi])

    def run(self):
        """
        Connect to the socket server, start the read loop in a separate thread,
        and keep the node running.
        """
        self.connect_socket()
        if not self.connected:
            rospy.logerr("Could not establish socket connection. Exiting node.")
            return

        # Start reading in a separate thread to avoid blocking the main ROS loop.
        read_thread = threading.Thread(target=self.read_loop)
        read_thread.daemon = True
        read_thread.start()

        rospy.loginfo("SocketTransformPublisher is running. Spinning rospy ...")
        rospy.spin()

        # On shutdown, close the socket.
        try:
            self.sock.close()
        except Exception:
            pass
    
    
if __name__ == "__main__":
    rospy.init_node('cobot_telekinesis_node', anonymous=True)
    node = CobotTelekinesis()
    # plotter=RealTimePlotter(node)
    # threading.Thread(target=node.run, daemon=True).start()
    # while not rospy.is_shutdown():
    #     plt.pause(0.1)
    node.run()
