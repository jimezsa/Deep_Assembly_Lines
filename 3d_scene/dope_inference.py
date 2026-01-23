"""
DOPE (Deep Object Pose Estimation) inference module.

This module provides 6D pose estimation for objects using the DOPE framework.
It wraps the DOPE detector and provides a clean interface for detecting objects
and drawing their 3D bounding boxes.
"""

import os
import sys
import cv2
import yaml
import time
import numpy as np
from scipy.spatial.transform import Rotation

# Add DOPE framework to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "frameworks", "dope"))
from cuboid import Cuboid3d
from cuboid_pnp_solver import CuboidPNPSolver
from detector import ModelData, ObjectDetector


class DOPEDetector:
    """DOPE-based 6D pose estimator for object detection (optimized).
    
    This class wraps the DOPE framework to provide 6D pose estimation
    (position + orientation) for trained object classes.
    
    Attributes:
        class_name: Name of the object class being detected
        draw_color: RGB color tuple for drawing detections
        dimension: Object dimensions in cm (x, y, z)
    """
    
    def __init__(self, config_path, camera_info_path, weight_path, class_name="tool"):
        """Initialize the DOPE detector.
        
        Args:
            config_path: Path to DOPE config YAML file
            camera_info_path: Path to camera info YAML file
            weight_path: Path to trained weights (.pth file)
            class_name: Name of the object class to detect
        """
        self.class_name = class_name
        self.weight_path = weight_path
        
        # Load configurations
        with open(config_path) as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        with open(camera_info_path) as f:
            self.camera_info = yaml.load(f, Loader=yaml.FullLoader)
        
        self.input_is_rectified = self.config["input_is_rectified"]
        self.downscale_height = self.config["downscale_height"]
        
        # Detection configuration (use object instead of lambda for speed)
        class ConfigDetect:
            pass
        self.config_detect = ConfigDetect()
        self.config_detect.mask_edges = 1
        self.config_detect.mask_faces = 1
        self.config_detect.vertex = 1
        self.config_detect.threshold = 0.5
        self.config_detect.softmax = 1000
        self.config_detect.thresh_angle = self.config["thresh_angle"]
        self.config_detect.thresh_map = self.config["thresh_map"]
        self.config_detect.sigma = self.config["sigma"]
        self.config_detect.thresh_points = self.config["thresh_points"]
        
        # Load neural network model
        # Set parallel=True to handle DDP-trained weights with 'module.' prefix
        self.model = ModelData(
            name=class_name,
            net_path=weight_path,
            parallel=True
        )
        self.model.load_net_model()
        print(f"[DOPE] Model loaded for class: {class_name}")
        
        # Get draw color (BGR for OpenCV)
        try:
            rgb_color = tuple(self.config["draw_colors"][class_name])
            self.draw_color = (rgb_color[2], rgb_color[1], rgb_color[0])  # BGR
            self.draw_color_rgb = rgb_color
        except:
            self.draw_color = (0, 255, 0)  # BGR
            self.draw_color_rgb = (0, 255, 0)
        
        # Get object dimensions (in cm)
        self.dimension = tuple(self.config["dimensions"][class_name])
        self.class_id = self.config["class_ids"][class_name]
        
        # Create PNP solver
        self.pnp_solver = CuboidPNPSolver(
            class_name, 
            cuboid3d=Cuboid3d(self.config["dimensions"][class_name])
        )
        
        # Setup camera matrices
        self._setup_camera_matrices()
        
        # Pre-compute scaled camera matrix and target size
        self._cached_scaling_factor = None
        self._cached_camera_matrix = None
        self._cached_target_size = None
        
        # Pre-allocate zero vectors for projectPoints
        self._zero_vec = np.zeros(3)
        
        print(f"[DOPE] Initialized - Object dimensions (cm): {self.dimension}")
    
    def _setup_camera_matrices(self):
        """Setup camera intrinsic matrices from camera info."""
        if self.input_is_rectified:
            P = np.matrix(
                self.camera_info["projection_matrix"]["data"], dtype="float64"
            ).copy()
            P.resize((3, 4))
            self.camera_matrix = P[:, :3]
            self.dist_coeffs = np.zeros((4, 1))
        else:
            self.camera_matrix = np.matrix(
                self.camera_info["camera_matrix"]["data"], dtype="float64"
            )
            self.camera_matrix.resize((3, 3))
            self.dist_coeffs = np.matrix(
                self.camera_info["distortion_coefficients"]["data"], dtype="float64"
            )
            self.dist_coeffs.resize((5, 1))
    
    def _get_scaled_params(self, height, width):
        """Get cached scaled camera matrix and target size.
        
        Args:
            height: Frame height
            width: Frame width
            
        Returns:
            tuple: (camera_matrix, target_size, scaling_factor)
        """
        scaling_factor = float(self.downscale_height) / height
        
        # Check if we can use cached values
        if self._cached_scaling_factor == scaling_factor:
            return self._cached_camera_matrix, self._cached_target_size, scaling_factor
        
        # Compute and cache
        camera_matrix = self.camera_matrix.copy()
        if scaling_factor < 1.0:
            camera_matrix[:2] *= scaling_factor
            target_size = (int(scaling_factor * width), int(scaling_factor * height))
        else:
            target_size = None
        
        self._cached_scaling_factor = scaling_factor
        self._cached_camera_matrix = camera_matrix
        self._cached_target_size = target_size
        
        return camera_matrix, target_size, scaling_factor
    
    def detect(self, frame):
        """Run 6D pose detection on a frame (optimized).
        
        Args:
            frame: BGR image (numpy array from OpenCV)
            
        Returns:
            dict with detection results containing:
                - detected: bool, True if object was detected
                - location: [x, y, z] in meters
                - quaternion: [x, y, z, w] orientation
                - projected_points: 2D bounding box corners
                - timestamp: detection timestamp
            Returns None if no detection
        """
        height, width = frame.shape[:2]
        
        # Get cached scaled parameters
        camera_matrix, target_size, scaling_factor = self._get_scaled_params(height, width)
        
        # Convert BGR to RGB (faster than [..., ::-1].copy())
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #frame_rgb = frame
        
        # Resize if needed
        if target_size is not None:
            frame_rgb = cv2.resize(frame_rgb, target_size, interpolation=cv2.INTER_LINEAR)
        
        # Update PNP solver camera parameters
        self.pnp_solver.set_camera_intrinsic_matrix(camera_matrix)
        self.pnp_solver.set_dist_coeffs(self.dist_coeffs)
        
        # Run object detection
        results, _ = ObjectDetector.detect_object_in_image(
            self.model.net, 
            self.pnp_solver, 
            frame_rgb, 
            self.config_detect,
            grid_belief_debug=False
        )
        
        if not results:
            return None
        
        # Get the best detection (first result with valid location)
        for result in results:
            loc = result["location"]
            if loc is not None:
                return {
                    "detected": True,
                    "location": [loc[0] * 0.01, loc[1] * 0.01, loc[2] * 0.01],  # cm to m
                    "quaternion": list(result["quaternion"]),
                    "projected_points": result["projected_points"],
                    "timestamp": time.time()
                }
        
        return None
    
    def draw_detection(self, frame, detection_result):
        """Draw 3D bounding box and coordinate axes on frame (optimized with OpenCV).
        
        Args:
            frame: BGR image to draw on (modified in-place)
            detection_result: Detection result dict from detect()
            
        Returns:
            Frame with drawn annotations (BGR format)
        """
        if detection_result is None or not detection_result["detected"]:
            return frame
        
        height, width = frame.shape[:2]
        camera_matrix, target_size, scaling_factor = self._get_scaled_params(height, width)
        
        # Draw the 3D bounding box using OpenCV (no PIL conversion needed)
        projected_points = detection_result["projected_points"]
        if projected_points is not None and len(projected_points) >= 8:
            try:
                # Convert to integer points
                pts = np.array(projected_points, dtype=np.int32)
                
                # Scale points back to original frame size if needed
                if scaling_factor < 1.0:
                    pts = (pts / scaling_factor).astype(np.int32)
                
                # Draw cube edges (8 corners: 0-3 front face, 4-7 back face)
                color = self.draw_color
                thickness = 2
                
                # Front face
                cv2.line(frame, tuple(pts[0]), tuple(pts[1]), color, thickness)
                cv2.line(frame, tuple(pts[1]), tuple(pts[2]), color, thickness)
                cv2.line(frame, tuple(pts[2]), tuple(pts[3]), color, thickness)
                cv2.line(frame, tuple(pts[3]), tuple(pts[0]), color, thickness)
                
                # Back face
                cv2.line(frame, tuple(pts[4]), tuple(pts[5]), color, thickness)
                cv2.line(frame, tuple(pts[5]), tuple(pts[6]), color, thickness)
                cv2.line(frame, tuple(pts[6]), tuple(pts[7]), color, thickness)
                cv2.line(frame, tuple(pts[7]), tuple(pts[4]), color, thickness)
                
                # Connecting edges
                cv2.line(frame, tuple(pts[0]), tuple(pts[4]), color, thickness)
                cv2.line(frame, tuple(pts[1]), tuple(pts[5]), color, thickness)
                cv2.line(frame, tuple(pts[2]), tuple(pts[6]), color, thickness)
                cv2.line(frame, tuple(pts[3]), tuple(pts[7]), color, thickness)
                
            except (TypeError, ValueError, IndexError):
                pass  # Skip drawing if points are invalid
        
        # Draw coordinate axes at object centroid (use cached camera matrix)
        location = detection_result["location"]
        quaternion = detection_result["quaternion"]
        location_cm = [loc * 100.0 for loc in location]
        
        # Scale camera matrix back for original frame if needed
        if scaling_factor < 1.0:
            cam_mat_orig = self.camera_matrix
        else:
            cam_mat_orig = camera_matrix
        
        self._draw_coordinate_system_cv(frame, cam_mat_orig, self.dist_coeffs, 
                                         location_cm, quaternion, axis_length=10)
        
        return frame
    
    def _draw_coordinate_system_cv(self, frame, camera_matrix, dist_coeffs, location, quaternion, axis_length=10):
        """Draw 3D coordinate axes using OpenCV (faster than PIL).
        
        Args:
            frame: BGR image to draw on (modified in-place)
            camera_matrix: Camera intrinsic matrix
            dist_coeffs: Distortion coefficients
            location: Object location in cm [x, y, z]
            quaternion: Object orientation [x, y, z, w]
            axis_length: Length of coordinate axes in cm
        """
        # Convert quaternion to rotation matrix
        rot = Rotation.from_quat(quaternion)
        rotation_matrix = rot.as_matrix()
        
        # Define axis endpoints in object frame
        axes_3d = np.array([
            [axis_length, 0, 0],  # X axis (red)
            [0, axis_length, 0],  # Y axis (green)
            [0, 0, axis_length],  # Z axis (blue)
        ], dtype=np.float64)
        
        # Transform axes to world frame
        location_arr = np.array(location, dtype=np.float64)
        axes_world = (rotation_matrix @ axes_3d.T).T + location_arr
        
        # Project all points at once (centroid + 3 axis endpoints)
        all_points = np.vstack([location_arr.reshape(1, 3), axes_world])
        points_2d, _ = cv2.projectPoints(all_points, self._zero_vec, self._zero_vec,
                                          camera_matrix, dist_coeffs)
        
        # Extract 2D coordinates
        centroid_2d = tuple(points_2d[0][0].astype(int))
        x_axis_2d = tuple(points_2d[1][0].astype(int))
        y_axis_2d = tuple(points_2d[2][0].astype(int))
        z_axis_2d = tuple(points_2d[3][0].astype(int))
        
        # Draw axes (BGR colors)
        cv2.line(frame, centroid_2d, x_axis_2d, (0, 0, 255), 3)  # X: Red
        cv2.line(frame, centroid_2d, y_axis_2d, (0, 255, 0), 3)  # Y: Green
        cv2.line(frame, centroid_2d, z_axis_2d, (255, 0, 0), 3)  # Z: Blue
    
def load_dope_detector(weights_path, config_path, camera_info_path, class_name="tool"):
    """Load and initialize a DOPE detector.
    
    Args:
        weights_path: Path to trained weights (.pth file)
        config_path: Path to DOPE config YAML file
        camera_info_path: Path to camera info YAML file
        class_name: Name of the object class to detect
        
    Returns:
        DOPEDetector instance or None if loading fails
    """
    try:
        if not os.path.exists(weights_path):
            print(f"[DOPE] Weights not found: {weights_path}")
            return None
        if not os.path.exists(config_path):
            print(f"[DOPE] Config not found: {config_path}")
            return None
        if not os.path.exists(camera_info_path):
            print(f"[DOPE] Camera info not found: {camera_info_path}")
            return None
            
        print(f"[DOPE] Loading model: {weights_path}")
        detector = DOPEDetector(
            config_path=config_path,
            camera_info_path=camera_info_path,
            weight_path=weights_path,
            class_name=class_name
        )
        print(f"[DOPE] Model ready")
        return detector
        
    except Exception as e:
        print(f"[DOPE] Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_empty_pose():
    """Create an empty/default pose dictionary.
    
    Returns:
        dict with default pose values
    """
    return {
        "detected": False,
        "fresh": False,
        "location": [0, 0, 0],
        "quaternion": [0, 0, 0, 1],
        "projected_points": [],
        "timestamp": 0
    }
