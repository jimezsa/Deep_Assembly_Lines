"""
3D Human Pose Estimation using multi-view triangulation.

This module implements a two-stage pipeline:
1. Stage 1: 2D pose estimation using YOLOv11-Pose on each camera view
2. Stage 2: 3D triangulation using DLT with RANSAC for robust reconstruction

The pipeline uses 8 calibrated cameras to detect 2D keypoints and triangulate
them into 3D world coordinates using known camera intrinsics and extrinsics.
"""

import os
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import threading

# =============================================================================
# Configuration
# =============================================================================

# YOLO-Pose keypoint names (COCO format - 17 keypoints)
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

# Skeleton connections for visualization (pairs of keypoint indices)
SKELETON_CONNECTIONS = [
    # Face
    (0, 1), (0, 2), (1, 3), (2, 4),
    # Arms
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    # Torso
    (5, 11), (6, 12), (11, 12),
    # Legs
    (11, 13), (13, 15), (12, 14), (14, 16)
]

# Colors for skeleton visualization (BGR format)
SKELETON_COLORS = {
    'face': (255, 200, 100),      # Light blue
    'left_arm': (100, 255, 100),  # Green
    'right_arm': (100, 100, 255), # Red
    'torso': (255, 255, 100),     # Cyan
    'left_leg': (100, 255, 255),  # Yellow
    'right_leg': (255, 100, 255)  # Magenta
}

# Keypoint color assignments for bones
BONE_COLORS = {
    (0, 1): SKELETON_COLORS['face'],
    (0, 2): SKELETON_COLORS['face'],
    (1, 3): SKELETON_COLORS['face'],
    (2, 4): SKELETON_COLORS['face'],
    (5, 6): SKELETON_COLORS['torso'],
    (5, 7): SKELETON_COLORS['left_arm'],
    (7, 9): SKELETON_COLORS['left_arm'],
    (6, 8): SKELETON_COLORS['right_arm'],
    (8, 10): SKELETON_COLORS['right_arm'],
    (5, 11): SKELETON_COLORS['torso'],
    (6, 12): SKELETON_COLORS['torso'],
    (11, 12): SKELETON_COLORS['torso'],
    (11, 13): SKELETON_COLORS['left_leg'],
    (13, 15): SKELETON_COLORS['left_leg'],
    (12, 14): SKELETON_COLORS['right_leg'],
    (14, 16): SKELETON_COLORS['right_leg']
}

# Minimum confidence threshold for keypoint detection
DEFAULT_KEYPOINT_CONF = 0.5

# Minimum number of views required for triangulation
MIN_VIEWS_FOR_TRIANGULATION = 2

# Default YOLO-Pose model
DEFAULT_POSE_MODEL = "yolo11n-pose.pt"


# =============================================================================
# Utility Functions
# =============================================================================

def get_best_device():
    """Detect the best available device for inference.
    
    Priority: MPS (Apple Silicon) > CUDA > CPU
    
    Returns:
        str: Device identifier ('mps', 'cuda', or 'cpu')
    """
    try:
        import torch
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            print("[Pose] MPS (Apple Silicon GPU) available")
            return "mps"
        elif torch.cuda.is_available():
            print(f"[Pose] CUDA available: {torch.cuda.get_device_name(0)}")
            return "cuda"
        else:
            print("[Pose] Using CPU")
            return "cpu"
    except ImportError:
        print("[Pose] PyTorch not available, using CPU")
        return "cpu"
    except Exception as e:
        print(f"[Pose] Error detecting device: {e}, using CPU")
        return "cpu"


# =============================================================================
# Data Classes
# =============================================================================

class Pose2D:
    """2D pose detection result from a single camera view."""
    
    def __init__(self, camera_id: str, keypoints: np.ndarray, 
                 confidences: np.ndarray, person_id: int = 0,
                 bbox: Optional[np.ndarray] = None):
        """
        Args:
            camera_id: Camera identifier
            keypoints: (17, 2) array of (u, v) pixel coordinates
            confidences: (17,) array of confidence scores
            person_id: ID for tracking multiple persons
            bbox: Optional bounding box [x1, y1, x2, y2]
        """
        self.camera_id = camera_id
        self.keypoints = keypoints  # Shape: (17, 2)
        self.confidences = confidences  # Shape: (17,)
        self.person_id = person_id
        self.bbox = bbox
    
    def get_valid_keypoints(self, min_conf: float = DEFAULT_KEYPOINT_CONF) -> Dict[int, Tuple[float, float, float]]:
        """Get keypoints with confidence above threshold.
        
        Returns:
            Dict mapping keypoint index -> (u, v, confidence)
        """
        valid = {}
        for i in range(len(self.keypoints)):
            if self.confidences[i] >= min_conf:
                valid[i] = (self.keypoints[i, 0], self.keypoints[i, 1], self.confidences[i])
        return valid
    
    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dictionary."""
        return {
            'camera_id': self.camera_id,
            'keypoints': self.keypoints.tolist(),
            'confidences': self.confidences.tolist(),
            'person_id': self.person_id,
            'bbox': self.bbox.tolist() if self.bbox is not None else None
        }


class Pose3D:
    """3D pose reconstruction from multi-view triangulation."""
    
    def __init__(self, keypoints_3d: np.ndarray, 
                 confidences: np.ndarray,
                 num_views_per_keypoint: np.ndarray,
                 person_id: int = 0):
        """
        Args:
            keypoints_3d: (17, 3) array of (X, Y, Z) world coordinates
            confidences: (17,) array of aggregated confidence scores
            num_views_per_keypoint: (17,) array of view counts used for each keypoint
            person_id: ID for tracking multiple persons
        """
        self.keypoints_3d = keypoints_3d  # Shape: (17, 3)
        self.confidences = confidences  # Shape: (17,)
        self.num_views = num_views_per_keypoint  # Shape: (17,)
        self.person_id = person_id
    
    def get_valid_keypoints(self, min_conf: float = 0.3) -> Dict[int, Tuple[float, float, float, float]]:
        """Get 3D keypoints with confidence above threshold.
        
        Returns:
            Dict mapping keypoint index -> (X, Y, Z, confidence)
        """
        valid = {}
        for i in range(len(self.keypoints_3d)):
            if self.confidences[i] >= min_conf and self.num_views[i] >= MIN_VIEWS_FOR_TRIANGULATION:
                valid[i] = (
                    self.keypoints_3d[i, 0],
                    self.keypoints_3d[i, 1],
                    self.keypoints_3d[i, 2],
                    self.confidences[i]
                )
        return valid
    
    def get_skeleton_lines(self, min_conf: float = 0.3) -> List[Tuple[np.ndarray, np.ndarray, Tuple[int, int, int]]]:
        """Get skeleton line segments for visualization.
        
        Returns:
            List of (point1, point2, color) tuples
        """
        lines = []
        valid_kpts = self.get_valid_keypoints(min_conf)
        
        for (i, j) in SKELETON_CONNECTIONS:
            if i in valid_kpts and j in valid_kpts:
                p1 = np.array([valid_kpts[i][0], valid_kpts[i][1], valid_kpts[i][2]])
                p2 = np.array([valid_kpts[j][0], valid_kpts[j][1], valid_kpts[j][2]])
                color = BONE_COLORS.get((i, j), (255, 255, 255))
                lines.append((p1, p2, color))
        
        return lines
    
    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dictionary."""
        return {
            'keypoints_3d': self.keypoints_3d.tolist(),
            'confidences': self.confidences.tolist(),
            'num_views': self.num_views.tolist(),
            'person_id': self.person_id,
            'skeleton_connections': SKELETON_CONNECTIONS
        }


# =============================================================================
# YOLO-Pose Detector
# =============================================================================

class PoseDetector:
    """YOLO-Pose based 2D human pose detector.
    
    Uses YOLOv11-Pose (or YOLOv8-Pose) for fast and accurate 2D keypoint detection.
    """
    
    def __init__(self, model, device: str = "cpu"):
        """Initialize the pose detector.
        
        Args:
            model: Loaded YOLO-Pose model from ultralytics
            device: Device for inference ('mps', 'cuda', or 'cpu')
        """
        self.model = model
        self.device = device
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.3
        self.font_thickness = 1
        
        # Enable CUDA optimizations
        try:
            import torch
            if torch.cuda.is_available():
                torch.backends.cudnn.benchmark = True
        except ImportError:
            pass
    
    def warmup(self, imgsz: int = 640):
        """Warmup the model with a dummy inference."""
        try:
            import torch
            dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
            _ = self.model(dummy, verbose=False, device=self.device)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"[Pose] Model warmed up on {self.device}")
        except Exception as e:
            print(f"[Pose] Warmup warning: {e}")
    
    def detect(self, frame: np.ndarray, camera_id: str,
               conf: float = 0.5, iou: float = 0.45, 
               imgsz: int = 640) -> List[Pose2D]:
        """Run pose detection on a frame.
        
        Args:
            frame: BGR image (numpy array from OpenCV)
            camera_id: Camera identifier for this frame
            conf: Confidence threshold for detection
            iou: IoU threshold for NMS
            imgsz: Input image size for the model
            
        Returns:
            List of Pose2D objects for detected persons
        """
        poses = []
        
        try:
            results = self.model(
                frame,
                verbose=False,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                device=self.device
            )
            
            if results and len(results) > 0:
                result = results[0]
                
                # Check if keypoints were detected
                if result.keypoints is not None and result.keypoints.data is not None:
                    keypoints_data = result.keypoints.data.cpu().numpy()  # Shape: (N, 17, 3) - x, y, conf
                    
                    # Get bounding boxes if available
                    boxes = None
                    if result.boxes is not None:
                        boxes = result.boxes.xyxy.cpu().numpy()
                    
                    for person_idx, kpts in enumerate(keypoints_data):
                        # kpts shape: (17, 3) - [x, y, confidence]
                        keypoints = kpts[:, :2]  # (17, 2)
                        confidences = kpts[:, 2]  # (17,)
                        
                        bbox = boxes[person_idx] if boxes is not None and person_idx < len(boxes) else None
                        
                        pose = Pose2D(
                            camera_id=camera_id,
                            keypoints=keypoints,
                            confidences=confidences,
                            person_id=person_idx,
                            bbox=bbox
                        )
                        poses.append(pose)
        
        except Exception as e:
            print(f"[Pose] Detection error on camera {camera_id}: {e}")
        
        return poses
    
    def draw_pose(self, frame: np.ndarray, pose: Pose2D, 
                  min_conf: float = DEFAULT_KEYPOINT_CONF,
                  draw_skeleton: bool = True,
                  draw_keypoints: bool = True,
                  scale_x: float = 1.0,
                  scale_y: float = 1.0) -> np.ndarray:
        """Draw 2D pose on frame.
        
        Args:
            frame: BGR image to draw on (modified in place)
            pose: Pose2D object with detected keypoints
            min_conf: Minimum confidence threshold for drawing
            draw_skeleton: Whether to draw skeleton lines
            draw_keypoints: Whether to draw keypoint circles
            scale_x: Scale factor for x coordinates (for drawing on resized frame)
            scale_y: Scale factor for y coordinates (for drawing on resized frame)
            
        Returns:
            Frame with drawn pose
        """
        if pose is None:
            return frame
        
        h, w = frame.shape[:2]
        
        # Draw skeleton lines first (so they're behind keypoints)
        if draw_skeleton:
            for (i, j) in SKELETON_CONNECTIONS:
                if pose.confidences[i] >= min_conf and pose.confidences[j] >= min_conf:
                    pt1 = (int(pose.keypoints[i, 0] * scale_x), int(pose.keypoints[i, 1] * scale_y))
                    pt2 = (int(pose.keypoints[j, 0] * scale_x), int(pose.keypoints[j, 1] * scale_y))
                    
                    # Check if points are within frame bounds
                    if (0 <= pt1[0] < w and 0 <= pt1[1] < h and
                        0 <= pt2[0] < w and 0 <= pt2[1] < h):
                        color = BONE_COLORS.get((i, j), (255, 255, 255))
                        cv2.line(frame, pt1, pt2, color, 2)
        
        # Draw keypoints
        if draw_keypoints:
            for i in range(len(pose.keypoints)):
                if pose.confidences[i] >= min_conf:
                    pt = (int(pose.keypoints[i, 0] * scale_x), int(pose.keypoints[i, 1] * scale_y))
                    if 0 <= pt[0] < w and 0 <= pt[1] < h:
                        # Draw filled circle
                        cv2.circle(frame, pt, 3, (0, 255, 255), -1)
                        cv2.circle(frame, pt, 3, (0, 0, 0), 1)
        
        # Draw bounding box if available
        if pose.bbox is not None:
            x1 = int(pose.bbox[0] * scale_x)
            y1 = int(pose.bbox[1] * scale_y)
            x2 = int(pose.bbox[2] * scale_x)
            y2 = int(pose.bbox[3] * scale_y)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
        
        return frame


# =============================================================================
# Multi-View Triangulation
# =============================================================================

class MultiViewTriangulator:
    """Triangulates 3D points from multiple calibrated camera views.
    
    Uses Direct Linear Transformation (DLT) with RANSAC for robust
    3D reconstruction from 2D keypoint detections.
    """
    
    def __init__(self, calibration_data: Dict[str, Dict]):
        """Initialize triangulator with camera calibration data.
        
        Args:
            calibration_data: Dictionary mapping camera_id -> calibration info
                              Each entry should have 'intrinsics' and 'extrinsics' keys
        """
        self.calibration = calibration_data
        self.projection_matrices = {}
        self.camera_positions = {}
        
        # Pre-compute projection matrices for each camera
        self._compute_projection_matrices()
    
    def _compute_projection_matrices(self):
        """Pre-compute projection matrices P = K @ [R|t] for each camera."""
        for cam_id, cam_data in self.calibration.items():
            try:
                # Get intrinsic matrix K
                K = np.array(cam_data['intrinsics']['K'], dtype=np.float64)
                
                # Get extrinsic matrix (4x4 camera-to-world transform)
                ext = np.array(cam_data['extrinsics'], dtype=np.float64)
                
                # The extrinsics are camera-to-world transforms
                # We need world-to-camera for projection: inv(ext)
                ext_inv = np.linalg.inv(ext)
                
                # Extract R and t from world-to-camera transform
                R = ext_inv[:3, :3]  # 3x3 rotation
                t = ext_inv[:3, 3:4]  # 3x1 translation
                
                # Projection matrix: P = K @ [R | t]
                Rt = np.hstack([R, t])  # 3x4
                P = K @ Rt  # 3x4
                
                self.projection_matrices[cam_id] = P
                
                # Store camera position in world coordinates (from camera-to-world transform)
                self.camera_positions[cam_id] = ext[:3, 3]
                
            except Exception as e:
                print(f"[Triangulator] Error computing projection for camera {cam_id}: {e}")
    
    def triangulate_point(self, observations: Dict[str, Tuple[float, float, float]]) -> Optional[Tuple[np.ndarray, float]]:
        """Triangulate a single 3D point from multiple 2D observations.
        
        Uses DLT (Direct Linear Transformation) method.
        
        Args:
            observations: Dict mapping camera_id -> (u, v, confidence)
                         u, v are pixel coordinates, confidence is detection confidence
        
        Returns:
            Tuple of (3D point [X, Y, Z], aggregated confidence) or None if failed
        """
        if len(observations) < MIN_VIEWS_FOR_TRIANGULATION:
            return None
        
        # Filter to cameras with valid projection matrices
        valid_obs = {
            cam_id: obs for cam_id, obs in observations.items()
            if cam_id in self.projection_matrices
        }
        
        if len(valid_obs) < MIN_VIEWS_FOR_TRIANGULATION:
            return None
        
        # Build DLT system: A @ X = 0
        A = []
        weights = []
        
        for cam_id, (u, v, conf) in valid_obs.items():
            P = self.projection_matrices[cam_id]
            
            # Two equations per observation (from homogeneous projection)
            # x * (P[2,:] @ X) = P[0,:] @ X
            # y * (P[2,:] @ X) = P[1,:] @ X
            A.append(u * P[2, :] - P[0, :])
            A.append(v * P[2, :] - P[1, :])
            
            # Weight by confidence
            weights.extend([conf, conf])
        
        A = np.array(A)  # Shape: (2*n_views, 4)
        weights = np.array(weights)
        
        # Weighted least squares: solve A @ X = 0
        # Apply weights
        W = np.diag(weights)
        A_weighted = W @ A
        
        # SVD solution
        try:
            _, _, Vt = np.linalg.svd(A_weighted)
            X_homogeneous = Vt[-1, :]  # Last row of V^T
            
            # Convert from homogeneous coordinates
            if abs(X_homogeneous[3]) < 1e-10:
                return None
            
            X = X_homogeneous[:3] / X_homogeneous[3]
            
            # Aggregate confidence (weighted average)
            total_conf = sum(obs[2] for obs in valid_obs.values())
            avg_conf = total_conf / len(valid_obs)
            
            return X, avg_conf
            
        except Exception as e:
            print(f"[Triangulator] SVD error: {e}")
            return None
    
    def triangulate_point_ransac(self, observations: Dict[str, Tuple[float, float, float]],
                                  n_iterations: int = 50,
                                  reprojection_threshold: float = 10.0) -> Optional[Tuple[np.ndarray, float, int]]:
        """Triangulate with RANSAC for robustness to outliers.
        
        Args:
            observations: Dict mapping camera_id -> (u, v, confidence)
            n_iterations: Number of RANSAC iterations
            reprojection_threshold: Maximum reprojection error in pixels
        
        Returns:
            Tuple of (3D point, confidence, num_inliers) or None if failed
        """
        if len(observations) < MIN_VIEWS_FOR_TRIANGULATION:
            return None
        
        camera_ids = list(observations.keys())
        n_cameras = len(camera_ids)
        
        if n_cameras < MIN_VIEWS_FOR_TRIANGULATION:
            return None
        
        best_point = None
        best_conf = 0
        best_inliers = 0
        
        # If only 2 cameras, just triangulate directly
        if n_cameras == 2:
            result = self.triangulate_point(observations)
            if result is not None:
                return result[0], result[1], 2
            return None
        
        # RANSAC loop
        for _ in range(n_iterations):
            # Sample 2 random cameras
            sample_ids = np.random.choice(camera_ids, size=2, replace=False)
            sample_obs = {cam_id: observations[cam_id] for cam_id in sample_ids}
            
            # Triangulate from sample
            result = self.triangulate_point(sample_obs)
            if result is None:
                continue
            
            point_3d, _ = result
            
            # Count inliers (cameras with low reprojection error)
            inliers = []
            for cam_id, (u, v, conf) in observations.items():
                if cam_id not in self.projection_matrices:
                    continue
                
                # Reproject 3D point
                P = self.projection_matrices[cam_id]
                X_hom = np.append(point_3d, 1.0)
                proj = P @ X_hom
                
                if abs(proj[2]) < 1e-10:
                    continue
                
                u_proj = proj[0] / proj[2]
                v_proj = proj[1] / proj[2]
                
                # Compute reprojection error
                error = np.sqrt((u - u_proj)**2 + (v - v_proj)**2)
                
                if error < reprojection_threshold:
                    inliers.append((cam_id, observations[cam_id]))
            
            if len(inliers) > best_inliers:
                # Re-triangulate using all inliers
                inlier_obs = dict(inliers)
                final_result = self.triangulate_point(inlier_obs)
                
                if final_result is not None:
                    best_point, best_conf = final_result
                    best_inliers = len(inliers)
        
        if best_point is not None and best_inliers >= MIN_VIEWS_FOR_TRIANGULATION:
            return best_point, best_conf, best_inliers
        
        return None
    
    def triangulate_pose(self, poses_2d: List[Pose2D],
                          min_keypoint_conf: float = DEFAULT_KEYPOINT_CONF,
                          use_ransac: bool = True) -> Optional[Pose3D]:
        """Triangulate full 3D pose from multiple 2D pose detections.
        
        Args:
            poses_2d: List of Pose2D objects from different cameras (same person)
            min_keypoint_conf: Minimum confidence for keypoints to be used
            use_ransac: Whether to use RANSAC for robust triangulation
        
        Returns:
            Pose3D object with triangulated keypoints, or None if failed
        """
        if len(poses_2d) < MIN_VIEWS_FOR_TRIANGULATION:
            return None
        
        # Initialize output arrays
        keypoints_3d = np.zeros((17, 3), dtype=np.float64)
        confidences = np.zeros(17, dtype=np.float64)
        num_views = np.zeros(17, dtype=np.int32)
        
        # Triangulate each keypoint independently
        for kpt_idx in range(17):
            # Collect observations from all cameras
            observations = {}
            
            for pose in poses_2d:
                if pose.confidences[kpt_idx] >= min_keypoint_conf:
                    u = pose.keypoints[kpt_idx, 0]
                    v = pose.keypoints[kpt_idx, 1]
                    conf = pose.confidences[kpt_idx]
                    observations[pose.camera_id] = (u, v, conf)
            
            # Triangulate this keypoint
            if len(observations) >= MIN_VIEWS_FOR_TRIANGULATION:
                if use_ransac:
                    result = self.triangulate_point_ransac(observations)
                    if result is not None:
                        keypoints_3d[kpt_idx] = result[0]
                        confidences[kpt_idx] = result[1]
                        num_views[kpt_idx] = result[2]
                else:
                    result = self.triangulate_point(observations)
                    if result is not None:
                        keypoints_3d[kpt_idx] = result[0]
                        confidences[kpt_idx] = result[1]
                        num_views[kpt_idx] = len(observations)
        
        # Check if we have enough valid keypoints
        valid_count = np.sum(num_views >= MIN_VIEWS_FOR_TRIANGULATION)
        if valid_count < 5:  # Need at least 5 keypoints for a meaningful pose
            return None
        
        return Pose3D(
            keypoints_3d=keypoints_3d,
            confidences=confidences,
            num_views_per_keypoint=num_views,
            person_id=poses_2d[0].person_id if poses_2d else 0
        )


# =============================================================================
# Multi-Person Matching
# =============================================================================

def match_persons_across_views(all_poses: Dict[str, List[Pose2D]],
                                calibration_data: Dict,
                                max_distance: float = 0.5) -> List[List[Pose2D]]:
    """Match detected persons across multiple camera views.
    
    Uses epipolar geometry and appearance-based matching to associate
    detections of the same person across different views.
    
    Args:
        all_poses: Dict mapping camera_id -> List of Pose2D detections
        calibration_data: Camera calibration data
        max_distance: Maximum matching distance threshold
    
    Returns:
        List of pose groups, where each group contains poses of the same person
        from different cameras
    """
    # For simplicity, if only one person per camera, assume same person
    # TODO: Implement more sophisticated multi-person matching
    
    person_groups = []
    
    # Get all cameras with detections
    cameras_with_poses = {
        cam_id: poses for cam_id, poses in all_poses.items() if len(poses) > 0
    }
    
    if len(cameras_with_poses) == 0:
        return []
    
    # Simple strategy: match by person index if all cameras have same count
    # Otherwise, take first person from each camera
    counts = [len(poses) for poses in cameras_with_poses.values()]
    
    if len(set(counts)) == 1 and counts[0] > 0:
        # Same number of persons in all cameras
        n_persons = counts[0]
        for person_idx in range(n_persons):
            group = []
            for cam_id, poses in cameras_with_poses.items():
                if person_idx < len(poses):
                    group.append(poses[person_idx])
            if len(group) >= MIN_VIEWS_FOR_TRIANGULATION:
                person_groups.append(group)
    else:
        # Different counts - just take first person from each camera
        group = []
        for cam_id, poses in cameras_with_poses.items():
            if len(poses) > 0:
                group.append(poses[0])
        if len(group) >= MIN_VIEWS_FOR_TRIANGULATION:
            person_groups.append(group)
    
    return person_groups


# =============================================================================
# Main Pipeline
# =============================================================================

class HumanPoseEstimator:
    """Complete 3D human pose estimation pipeline.
    
    Combines 2D detection (YOLO-Pose) with multi-view triangulation
    for accurate 3D human pose estimation.
    """
    
    def __init__(self, calibration_data: Dict[str, Dict],
                 pose_detector: PoseDetector):
        """Initialize the pipeline.
        
        Args:
            calibration_data: Camera calibration data
            pose_detector: Initialized PoseDetector for 2D detection
        """
        self.pose_detector = pose_detector
        self.triangulator = MultiViewTriangulator(calibration_data)
        self.calibration_data = calibration_data
        
        # Cache for 2D poses per camera (for visualization)
        self.current_2d_poses: Dict[str, List[Pose2D]] = {}
        self.current_3d_poses: List[Pose3D] = []
        self.lock = threading.Lock()
    
    def process_frames(self, frames: Dict[str, np.ndarray],
                        keypoint_conf: float = DEFAULT_KEYPOINT_CONF) -> Tuple[Dict[str, List[Pose2D]], List[Pose3D]]:
        """Process frames from all cameras and compute 3D poses.
        
        Args:
            frames: Dict mapping camera_id -> BGR frame
            keypoint_conf: Minimum keypoint confidence threshold
        
        Returns:
            Tuple of (2D poses per camera, list of 3D poses)
        """
        # Stage 1: 2D Pose Detection on each camera
        all_2d_poses = {}
        for cam_id, frame in frames.items():
            poses = self.pose_detector.detect(frame, cam_id, conf=0.5)
            all_2d_poses[cam_id] = poses
        
        # Match persons across views
        person_groups = match_persons_across_views(
            all_2d_poses, self.calibration_data
        )
        
        # Stage 2: Triangulate each person group
        poses_3d = []
        for group in person_groups:
            pose_3d = self.triangulator.triangulate_pose(
                group, 
                min_keypoint_conf=keypoint_conf,
                use_ransac=True
            )
            if pose_3d is not None:
                poses_3d.append(pose_3d)
        
        # Update cache
        with self.lock:
            self.current_2d_poses = all_2d_poses
            self.current_3d_poses = poses_3d
        
        return all_2d_poses, poses_3d
    
    def get_current_poses(self) -> Tuple[Dict[str, List[Pose2D]], List[Pose3D]]:
        """Get the most recent pose detections.
        
        Returns:
            Tuple of (2D poses per camera, list of 3D poses)
        """
        with self.lock:
            return self.current_2d_poses.copy(), self.current_3d_poses.copy()
    
    def draw_2d_poses(self, frame: np.ndarray, camera_id: str,
                       min_conf: float = DEFAULT_KEYPOINT_CONF) -> np.ndarray:
        """Draw 2D poses on a camera frame.
        
        Args:
            frame: BGR image to draw on
            camera_id: Camera ID to get poses for
            min_conf: Minimum confidence for drawing
            
        Returns:
            Frame with drawn poses
        """
        with self.lock:
            poses = self.current_2d_poses.get(camera_id, [])
        
        for pose in poses:
            self.pose_detector.draw_pose(frame, pose, min_conf=min_conf)
        
        return frame


# =============================================================================
# Factory Functions
# =============================================================================

def load_pose_model(model_path: Optional[str] = None, 
                    warmup: bool = True) -> Tuple[Optional[PoseDetector], str]:
    """Load YOLO-Pose model with automatic device selection.
    
    Args:
        model_path: Path to YOLO-Pose weights (.pt file).
                   If None, downloads default yolo11n-pose.pt
        warmup: Whether to run a warmup inference
        
    Returns:
        Tuple: (PoseDetector instance, device string) or (None, None) on failure
    """
    try:
        from ultralytics import YOLO
        
        # Detect best device
        device = get_best_device()
        
        # Use default model if not specified
        if model_path is None:
            model_path = DEFAULT_POSE_MODEL
            print(f"[Pose] Using default model: {model_path}")
        
        # Load model (will download if not present)
        print(f"[Pose] Loading model: {model_path}")
        model = YOLO(model_path)
        
        if warmup:
            print(f"[Pose] Warming up on {device}...")
            _ = model(np.zeros((320, 320, 3), dtype=np.uint8), 
                     verbose=False, device=device)
        
        detector = PoseDetector(model, device=device)
        print(f"[Pose] Model ready on {device}")
        return detector, device
        
    except ImportError:
        print("[Pose] ultralytics not installed, pose detection disabled")
        return None, "cpu"
    except Exception as e:
        print(f"[Pose] Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return None, "cpu"


def create_pose_estimator(calibration_data: Dict[str, Dict],
                           model_path: Optional[str] = None) -> Optional[HumanPoseEstimator]:
    """Create a complete pose estimation pipeline.
    
    Args:
        calibration_data: Camera calibration data
        model_path: Optional path to YOLO-Pose model
        
    Returns:
        HumanPoseEstimator instance or None on failure
    """
    detector, device = load_pose_model(model_path)
    
    if detector is None:
        print("[Pose] Failed to create pose estimator - detector not loaded")
        return None
    
    estimator = HumanPoseEstimator(calibration_data, detector)
    print(f"[Pose] Pipeline ready with {len(calibration_data)} cameras")
    return estimator


def create_empty_3d_pose() -> Dict:
    """Create an empty 3D pose dictionary for API responses."""
    return {
        'detected': False,
        'keypoints_3d': [],
        'confidences': [],
        'num_views': [],
        'skeleton_connections': SKELETON_CONNECTIONS,
        'keypoint_names': KEYPOINT_NAMES
    }


def pose_3d_to_dict(pose: Optional[Pose3D]) -> Dict:
    """Convert Pose3D to API-friendly dictionary."""
    if pose is None:
        return create_empty_3d_pose()
    
    result = pose.to_dict()
    result['detected'] = True
    result['keypoint_names'] = KEYPOINT_NAMES
    return result


def poses_2d_to_dict(poses: Dict[str, List[Pose2D]]) -> Dict[str, List[Dict]]:
    """Convert 2D poses dictionary to API-friendly format."""
    result = {}
    for cam_id, pose_list in poses.items():
        result[cam_id] = [pose.to_dict() for pose in pose_list]
    return result
