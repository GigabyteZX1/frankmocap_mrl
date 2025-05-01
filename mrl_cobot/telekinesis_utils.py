# Author: Emaad Ahmed
# Date: 2025-02-25

import cv2
import numpy as np
import socket
import time
import json
from scipy.spatial.transform import Rotation as R
import time
from renderer import glViewer
import matplotlib.pyplot as plt

class CobotTelekinesis:
    def __init__(self):
        self.connected = False
        self.client_socket = None
        self.server_socket = None
        self.start_time = None
        self.stop_time = None

        # Test
        self.roll = []
        self.pitch = []
        self.yaw = []

    # TCP/IP communication functions
    def connect_to_cobot(self, ip_address, max_retries=5, retry_delay=2, timeout=60):
        """Set up UDP server and wait for initial client message
        Parameters:
            ip_address: IP address and port tuple (ip, port) for the server to bind to.
            max_retries: Maximum number of connection retry attempts.
            retry_delay: Delay between retry attempts in seconds.
            timeout: Timeout in seconds to wait for initial client message.
        """
        retries = 0
        while retries < max_retries:
            print(f"Connection attempt: {retries+1}/{max_retries}")
            try:
                # Set up UDP socket
                self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                print("Initializing UDP Server...")
                print(f"Binding to: {ip_address}")
                self.server_socket.bind(ip_address)
                print("Initialized UDP Server.")
                
                # Set timeout for receiving initial message
                self.server_socket.settimeout(timeout)
                
                # Wait for first message from client to establish "connection"
                print(f"Waiting for client connection at {ip_address}...")
                try:
                    data, self.client_address = self.server_socket.recvfrom(1024)
                    print(f"Received initial message from client at {self.client_address}")
                    print(f"Client message: {data.decode('utf-8').strip()}")
                    
                    # Send acknowledgment to client
                    ack_message = "Connection established with server"
                    self.server_socket.sendto(ack_message.encode('utf-8'), self.client_address)
                    
                    self.connected = True
                    print(f"Connection established with client at {self.client_address}")
                    
                    # Switch to non-blocking mode for future communication
                    self.server_socket.settimeout(None)
                    break
                    
                except socket.timeout:
                    print(f"Timeout waiting for client connection (waited {timeout} seconds)")
                    self.server_socket.close()
                    retries += 1
                    continue
                    
            except socket.error as e:
                print(f"Error setting up UDP socket: {e}")
                retries += 1
                time.sleep(retry_delay)

        if not self.connected:
            print("Failed to establish connection with client.")
            
    def send_data(self, msg):
        """Send data to the cobot client using UDP protocol.
        Parameters:
            msg: Data to be sent to the client.
        """
        if not self.connected:
            print("Not connected to any client.")
            return

        try:
            data = json.dumps(msg.tolist())
            cmd_msg_bytes = data.encode('utf-8') + b'\n'
            try:
                self.server_socket.sendto(cmd_msg_bytes, self.client_address)
                print(f"Sent {msg} command to client via UDP!")
            except Exception as e:
                print(f"Error while sending data to client: {e}")
                self.connected = False
        except Exception as e:
            print(f"Error while preparing data: {e}")

    def disconnect_from_cobot(self):
        """Disconnect from the cobot server.
        Parameters:
            None
        """
        
        print(f"Connected: {self.connected}")
        try:
            # Test if socket is still connected
            if self.connected:
                disconnect_msg = json.dumps({"command": "CLOSE_CONNECTION"})
                self.server_socket.sendto(disconnect_msg.encode('utf-8') + b'\n', self.client_address)
                # self.server_socket.getpeername()  # Will raise error if not connected
                self.server_socket.close()
                print("Socket Disconnected")
        except socket.error:
            print("Socket Already Disconnected")
        

    # Utility functions
    def axis_angle_to_rotation_matrix(self, axis_angle):
        """Convert axis-angle to rotation matrix"""
        angle = np.linalg.norm(axis_angle)
        axis = axis_angle / angle if angle != 0 else np.array([1, 0, 0])
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        R = cos_angle * np.eye(3) + sin_angle * np.cross(np.eye(3), axis) + (1 - cos_angle) * np.outer(axis, axis)
        return R

    def show_image(self, image):
        glViewer.setWindowSize(image.shape[1], image.shape[0])
        glViewer.setBackgroundTexture(image)
        glViewer.SetOrthoCamera(True)
        glViewer.show(1)

    def resize_image(self, image, target_height):
        """
        Resize an image to the target height, while maintaining the aspect ratio.
        
        Args:
            image: The image to be resized.
            target_height: The target height of the resized image.
            scaling_factor: The scaling factor to be applied to the image.
        
        Returns:
            The resized image.
        """
        height, width = image.shape[:2]
        scaling_factor = target_height / height
        target_width = int(width * scaling_factor)
        return cv2.resize(image, (target_width, target_height))
    
    def draw_axes(self, image, position, orientation, scale=50, rotation_matrix_flag = False):
        """Draw 3D axes on the image"""

        if not rotation_matrix_flag:
            rotation_matrix = self.axis_angle_to_rotation_matrix(orientation)
        else:
            rotation_matrix = orientation
            
        axes = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, scale]])
        
        rotated_axes = rotation_matrix @ axes.T

        # Draw axes
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # Red, Green, Blue for X, Y, Z
        for i, color in enumerate(colors):
            start_point = (int(position[0]), int(position[1]))
            end_point = (int(position[0] + rotated_axes[0, i]),
                        int(position[1] + rotated_axes[1, i]))
            cv2.arrowedLine(image, start_point, end_point, color, 2)
        
        return image

    def extract_joint_data(self, pred_output, joint_indices, pose_type_flag = 0):
        """
        Extracts joint rotations for specified indices.
        Parameters:
            pred_output: Dictionary containing SMPL-X body/hand pose model outputs.
            joint_indices: List of joint indices to extract.
            pose_type_flag: Flag to determine if orientation or position is to be extracted.
        Returns:
            rotations: List of 3x3 rotation matrices for joints. [OR]
            positions: List of 3D positions for joints.
        """

        data = None
        data = [pred_output[idx] for idx in joint_indices]
        return data

        
    def isGripperClosed(self, joint_positions, threshold = 15):
        """
        Determine if the gripper is closed.
        Parameters:
            joint_positions: List of 3D positions for joints.
        Returns:
            is_closed: True if gripper is closed, False otherwise.
        """
        dist = np.linalg.norm(joint_positions[0] - joint_positions[1])
        print("Dist: ", dist)
        return dist < threshold
    
    def compute_relative_orientation(self, joint_rotations):
        """
        Compute cumulative rotation for given joint_rotations chain.
        Parameters:
            joint_rotations: List of 3x3 rotation matrices for joints along the chain.
        Returns:
            relative_orientation: 3x3 rotation matrix.
        """
        relative_orientation = np.eye(3)  # Start with identity matrix
        for rotation in joint_rotations:
            relative_orientation = np.dot(relative_orientation, rotation)
        return relative_orientation

    def convert_fm_to_ros_pos(self, T_mat):
        """
        Convert transformation matrix from image space to ROS coordinate system
        """

        R_convert = np.array([[0, 0, 1, 0],
                              [1, 0, 0, 0],
                              [0, 1, 0, 0],
                              [0, 0, 0, 1]])
        
        return R_convert @ T_mat @ R_convert.T

    def convert_fm_to_ros_orn(self, rotation_mat):
        """
        Convert Rotations from SMPL space to ROS coordinate system
        Parameters:
            rotation_matrix
        """
        
        # R_align_torso = np.array([[0, 0, -1],
        #                           [-1, 0, 0],
        #                           [0, 1, 0]])
        R_align_torso = np.eye(3)
        
        return R_align_torso@rotation_mat@R_align_torso.T
    
    def convert_to_euler(self, R_wrist, window_size = 5):

        # Convert rotation matrix to Euler angles in ROS space
        euler_angles = R.from_matrix(R_wrist).as_euler('xyz', degrees=True)

        print("Euler Angles in ROS Space (degrees):")
        print("Roll  (X-axis):", euler_angles[0])
        print("Pitch (Y-axis):", euler_angles[1])
        print("Yaw   (Z-axis):", euler_angles[2])

        # Initialize buffer lists if they don't exist
        if not hasattr(self, 'roll_buffer'):
            self.roll_buffer = []
        if not hasattr(self, 'pitch_buffer'):
            self.pitch_buffer = []
        if not hasattr(self, 'yaw_buffer'):
            self.yaw_buffer = []
        
        # Add current values to buffers
        self.roll_buffer.append(euler_angles[0])
        self.pitch_buffer.append(euler_angles[1])
        self.yaw_buffer.append(euler_angles[2])
        
        # Keep only the most recent window_size values
        self.roll_buffer = self.roll_buffer[-window_size:]
        self.pitch_buffer = self.pitch_buffer[-window_size:]
        self.yaw_buffer = self.yaw_buffer[-window_size:]
        
        # Calculate moving averages
        if len(self.roll_buffer) > 0:  # Ensure we have data
            roll_avg = sum(self.roll_buffer) / len(self.roll_buffer)
            pitch_avg = sum(self.pitch_buffer) / len(self.pitch_buffer)
            yaw_avg = sum(self.yaw_buffer) / len(self.yaw_buffer)
            
            # Append the filtered values to the main lists
            self.roll.append(roll_avg)
            self.pitch.append(pitch_avg)
            self.yaw.append(yaw_avg)
        else:
            # If no buffer data yet, use raw values
            self.roll.append(euler_angles[0])
            self.pitch.append(euler_angles[1])
            self.yaw.append(euler_angles[2])

    def invert_T(self, T):
        """
        Invert Transformation Matrix
        """
        T_inv = np.eye(4)
        T_inv[:3, :3] = T[:3, :3].T
        T_inv[:3, 3] = -T[:3, :3].T @ T[:3, 3]
        return T_inv

    def compute_relative_transformation(self, joint_positions, joint_rotations, scale_factor):
        """
        Form Relative Tranformation Matrix"""

        # relative_orn = self.compute_relative_orientation(joint_rotations)

        T_torso = np.eye(4)
        T_torso[:3, :3] = joint_rotations[0]
        T_torso[:3, 3] = joint_positions[0]

        T_torso = self.convert_fm_to_ros_pos(T_torso)

        T_wrist = np.eye(4)
        T_wrist[:3, :3] = joint_rotations[-1]
        T_wrist[:3, 3] = joint_positions[1]

        T_wrist = self.convert_fm_to_ros_pos(T_wrist)

        print("torso_pos: ", T_torso[:3, 3])
        print("wrist_pos: ", T_wrist[:3, 3])
        print("Scale factor: ", scale_factor)
        T_relative = np.dot(self.invert_T(T_torso), T_wrist)

        # self.convert_to_euler(T_relative[:3, :3])

        T_relative[:3, 3] *= scale_factor
        
        return T_relative, T_torso, T_wrist

    def verify_relative_transform(self, T_torso, T_wrist, T_relative):
        """
        Check if the computed T_relative correctly reconstructs the wrist pose.
        """
        # Reconstruct wrist pose using: T_wrist_check = T_torso * T_relative
        T_wrist_check = T_torso @ T_relative

        # Compute error between actual and reconstructed wrist pose
        position_error = np.linalg.norm(T_wrist_check[:3, 3] - T_wrist[:3, 3])
        rotation_error = np.linalg.norm(T_wrist_check[:3, :3] - T_wrist[:3, :3])

        print(f"Position Error: {position_error}")
        print(f"Rotation Error: {rotation_error}")

        # Threshold for correctness
        if position_error < 1e-3 and rotation_error < 1e-3:
            print("Relative transformation is correct!")
        else:
            print("Relative transformation has an error!")

        return T_wrist_check

    def proc_time(self, mode = 0):
        if mode == 0:
            self.start_time = time.time()
        elif mode == 1:
            self.end_time = time.time()
            print(f"Processing time: {self.end_time - self.start_time}")

    def plot_and_save_orientation(self, save_path="orientation_plots.png"):
        """
        Plot roll, pitch, and yaw arrays and save the plots.
        
        Parameters:
        -----------
        roll : array-like
            Array containing roll values in degrees or radians
        pitch : array-like
            Array containing pitch values in degrees or radians
        yaw : array-like
            Array containing yaw values in degrees or radians
        save_path : str, optional
            Path to save the plot image (default: "orientation_plots.png")
        
        Returns:
        --------
        bool
            True if successful, False otherwise
        """
        # Check that all arrays have the same length
        if not (len(self.roll) == len(self.pitch) == len(self.yaw)):
            print("Error: Arrays must have the same length")
            return False
        
        try:
            # Convert to numpy arrays if they aren't already
            roll_array = np.array(self.roll)
            pitch_array = np.array(self.pitch)
            yaw_array = np.array(self.yaw)
            
            # Create time/sample index
            time = np.arange(len(roll_array))
            
            # Create subplot figure
            fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
            fig.suptitle('Orientation Data', fontsize=16)
            
            # Plot roll data
            axes[0].plot(time, roll_array, 'r-', linewidth=2)
            axes[0].set_ylabel('Roll')
            axes[0].grid(True)
            
            # Plot pitch data
            axes[1].plot(time, pitch_array, 'g-', linewidth=2)
            axes[1].set_ylabel('Pitch')
            axes[1].grid(True)
            
            # Plot yaw data
            axes[2].plot(time, yaw_array, 'b-', linewidth=2)
            axes[2].set_xlabel('Sample')
            axes[2].set_ylabel('Yaw')
            axes[2].grid(True)
            
            # Adjust layout
            plt.tight_layout()
            
            # Save the figure
            plt.savefig(save_path, dpi=300)
            print(f"Successfully saved orientation plots to {save_path}")
            
            # Display the plot (optional - comment out if not needed)
            plt.show()
            
            return True
            
        except Exception as e:
            print(f"Error plotting orientation data: {str(e)}")
            return False