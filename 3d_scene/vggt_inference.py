"""
VGGT (Visual Geometry Grounded Transformer) inference module.

This module provides 3D point cloud reconstruction from multiple camera views
using the VGGT framework. It processes synchronized video frames from multiple
cameras and generates a colored point cloud.
"""

import os
import sys
import cv2
import torch
import numpy as np
import threading
import time

# Add VGGT framework to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "frameworks"))
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map


class VGGTDetector:
    """VGGT-based 3D point cloud generator from multi-view images.
    
    This class wraps the VGGT framework to provide 3D reconstruction
    from multiple camera views. It generates colored point clouds with
    confidence filtering.
    
    Attributes:
        device: Torch device (cuda/cpu)
        dtype: Torch dtype for inference
        model: VGGT model instance
        conf_threshold_pct: Confidence threshold percentile for filtering points
    """
    
    def __init__(self, weights_path="weights/vggt.pt", conf_threshold_pct=50.0, max_points=50000):
        """Initialize the VGGT detector.
        
        Args:
            weights_path: Path to VGGT model weights
            conf_threshold_pct: Confidence threshold as percentile (0-100)
                               Points below this percentile will be filtered out
            max_points: Maximum number of points to return (for performance)
        """
        self.weights_path = weights_path
        self.conf_threshold_pct = conf_threshold_pct
        self.max_points = max_points
        self.model = None
        self.device = None
        self.dtype = None
        
        # Inference lock for thread safety
        self.lock = threading.Lock()
        
        # Cached results
        self.last_point_cloud = None
        self.last_inference_time = 0
        
    def load_model(self):
        """Load the VGGT model and weights."""
        # Determine device and dtype
        if torch.cuda.is_available():
            self.device = "cuda"
            # bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+)
            capability = torch.cuda.get_device_capability()
            self.dtype = torch.bfloat16 if capability[0] >= 8 else torch.float16
        else:
            self.device = "cpu"
            self.dtype = torch.float32
        
        print(f"[VGGT] Using device: {self.device} with dtype: {self.dtype}")
        
        # Initialize model
        self.model = VGGT()
        
        # Load weights
        if not os.path.exists(self.weights_path):
            print(f"[VGGT] Warning: Weights file not found at {self.weights_path}")
            return False
        
        try:
            state_dict = torch.load(self.weights_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(state_dict)
            self.model = self.model.to(self.device)
            self.model.eval()
            print(f"[VGGT] Model loaded successfully from {self.weights_path}")
            return True
        except Exception as e:
            print(f"[VGGT] Error loading model: {e}")
            return False
    
    def preprocess_frames(self, frames):
        """Preprocess frames for VGGT inference.
        
        Args:
            frames: List of numpy arrays (BGR images from OpenCV)
            
        Returns:
            Preprocessed tensor ready for model input
        """
        # Convert BGR to RGB and normalize
        processed_frames = []
        for frame in frames:
            # Convert BGR to RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Resize to VGGT expected size (518x518 or similar)
            # The model uses 14x14 patches, so dimensions should be divisible by 14
            h, w = rgb.shape[:2]
            # Resize maintaining aspect ratio, then pad if needed
            target_size = 518
            scale = target_size / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            # Make dimensions divisible by 14
            new_h = (new_h // 14) * 14
            new_w = (new_w // 14) * 14
            if new_h == 0:
                new_h = 14
            if new_w == 0:
                new_w = 14
            
            resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # Normalize to [0, 1]
            normalized = resized.astype(np.float32) / 255.0
            processed_frames.append(normalized)
        
        # Stack frames: (N, H, W, 3) -> (1, N, 3, H, W)
        stacked = np.stack(processed_frames, axis=0)  # (N, H, W, 3)
        stacked = np.transpose(stacked, (0, 3, 1, 2))  # (N, 3, H, W)
        tensor = torch.from_numpy(stacked).unsqueeze(0)  # (1, N, 3, H, W)
        
        return tensor.to(self.device)
    
    def run_inference(self, frames):
        """Run VGGT inference on a list of frames.
        
        Args:
            frames: List of numpy arrays (BGR images from OpenCV), one per camera
            
        Returns:
            Dictionary with point cloud data:
            {
                'points': np.array of shape (N, 3) - XYZ coordinates
                'colors': np.array of shape (N, 3) - RGB colors (0-255)
                'num_points': int - number of points
                'inference_time': float - time taken for inference
                'success': bool - whether inference succeeded
            }
        """
        if self.model is None:
            return {
                'points': np.array([]),
                'colors': np.array([]),
                'num_points': 0,
                'inference_time': 0,
                'success': False
            }
        
        with self.lock:
            start_time = time.perf_counter()
            
            try:
                # Preprocess frames
                images = self.preprocess_frames(frames)
                
                # Run inference
                with torch.no_grad():
                    if self.device == "cuda":
                        with torch.amp.autocast(device_type=self.device, dtype=self.dtype):
                            predictions = self.model(images)
                    else:
                        predictions = self.model(images.to(self.dtype))
                
                # Process predictions
                point_cloud = self._process_predictions(predictions, images)
                
                inference_time = time.perf_counter() - start_time
                point_cloud['inference_time'] = inference_time
                point_cloud['success'] = True
                
                # Cache results
                self.last_point_cloud = point_cloud
                self.last_inference_time = time.time()
                
                print(f"[VGGT] Inference complete: {point_cloud['num_points']} points in {inference_time:.3f}s")
                
                return point_cloud
                
            except Exception as e:
                print(f"[VGGT] Inference error: {e}")
                import traceback
                traceback.print_exc()
                return {
                    'points': np.array([]),
                    'colors': np.array([]),
                    'num_points': 0,
                    'inference_time': time.perf_counter() - start_time,
                    'success': False
                }
    
    def _process_predictions(self, predictions, images):
        """Process model predictions to extract point cloud.
        
        Args:
            predictions: Model output dictionary
            images: Input images tensor
            
        Returns:
            Dictionary with processed point cloud data
        """
        # Convert pose encoding to extrinsic and intrinsic matrices
        extrinsic, intrinsic = pose_encoding_to_extri_intri(
            predictions["pose_enc"], 
            images.shape[-2:]
        )
        predictions["extrinsic"] = extrinsic
        predictions["intrinsic"] = intrinsic
        
        # Convert tensors to numpy
        for key in list(predictions.keys()):
            val = predictions[key]
            if isinstance(val, torch.Tensor):
                t = val.detach().cpu()
                if t.dtype == torch.bfloat16:
                    t = t.float()
                predictions[key] = t.numpy().squeeze(0)  # Remove batch dimension
        
        # Generate world points from depth map
        depth_map = predictions["depth"]
        world_points = unproject_depth_map_to_point_map(
            depth_map, 
            predictions["extrinsic"], 
            predictions["intrinsic"]
        )
        
        # Get images for colors (NCHW -> NHWC)
        images_np = predictions["images"]
        if images_np.ndim == 4 and images_np.shape[1] == 3:
            images_np = np.transpose(images_np, (0, 2, 3, 1))
        
        # Flatten points and colors
        points_flat = world_points.reshape(-1, 3)
        colors_flat = (images_np.reshape(-1, 3) * 255).astype(np.uint8)
        
        # Get confidence values
        conf = predictions.get("depth_conf", np.ones(world_points.shape[:-1]))
        conf_flat = conf.reshape(-1)
        
        # Apply confidence threshold (percentile-based)
        if self.conf_threshold_pct > 0:
            conf_threshold = np.percentile(conf_flat, self.conf_threshold_pct)
        else:
            conf_threshold = 0
        
        mask = (conf_flat >= conf_threshold) & (conf_flat > 1e-5)
        
        # Filter out black/very dark background pixels
        color_sum = colors_flat.sum(axis=1)
        mask = mask & (color_sum >= 16)
        
        # Apply mask
        points_filtered = points_flat[mask]
        colors_filtered = colors_flat[mask]
        
        # Subsample if too many points
        if len(points_filtered) > self.max_points:
            indices = np.random.choice(len(points_filtered), self.max_points, replace=False)
            points_filtered = points_filtered[indices]
            colors_filtered = colors_filtered[indices]
        
        return {
            'points': points_filtered.astype(np.float32),
            'colors': colors_filtered,
            'num_points': len(points_filtered)
        }
    
    def get_last_point_cloud(self):
        """Get the last computed point cloud (for polling).
        
        Returns:
            Dictionary with last point cloud data or None if no inference has run
        """
        return self.last_point_cloud


def load_vggt_detector(weights_path="weights/vggt.pt", conf_threshold_pct=50.0, max_points=100000):
    """Load and initialize a VGGT detector.
    
    Args:
        weights_path: Path to VGGT weights file
        conf_threshold_pct: Confidence threshold percentile for point filtering
        max_points: Maximum number of points to return
        
    Returns:
        VGGTDetector instance if successful, None otherwise
    """
    detector = VGGTDetector(
        weights_path=weights_path,
        conf_threshold_pct=conf_threshold_pct,
        max_points=max_points
    )
    
    if detector.load_model():
        return detector
    return None


def create_empty_point_cloud():
    """Create an empty point cloud result structure.
    
    Returns:
        Dictionary with empty point cloud data
    """
    return {
        'points': np.array([]),
        'colors': np.array([]),
        'num_points': 0,
        'inference_time': 0,
        'success': False
    }
