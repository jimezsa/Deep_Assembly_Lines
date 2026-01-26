"""
VGGT (Visual Geometry Grounded Transformer) inference module - OPTIMIZED.

This module provides 3D point cloud reconstruction from multiple camera views
using the VGGT framework. 

Optimizations:
- Pre-allocated tensors and buffers
- CUDA streams for async processing
- Optimized preprocessing pipeline
- Efficient point cloud filtering
- Binary-ready output format
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
    """VGGT-based 3D point cloud generator - OPTIMIZED VERSION.
    
    Optimizations:
    - Pre-allocated GPU tensors
    - Fused preprocessing operations
    - Async CUDA operations
    - Efficient numpy operations for post-processing
    """
    
    def __init__(self, weights_path="weights/vggt.pt", conf_threshold_pct=50.0, max_points=100000):
        """Initialize the VGGT detector.
        
        Args:
            weights_path: Path to VGGT model weights
            conf_threshold_pct: Confidence threshold as percentile (0-100)
            max_points: Maximum number of points to return
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
        self._last_hash = None  # For change detection
        
        # Pre-allocated buffers (set after first inference)
        self._target_size = 518  # Default VGGT input size
        self._prealloc_initialized = False
        
    def load_model(self):
        """Load the VGGT model and weights."""
        # Determine device and dtype
        if torch.cuda.is_available():
            self.device = "cuda"
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
            
            # Note: torch.compile disabled - causes graph breaks with VGGT model
            # due to .item() calls in rope.py. Use eager mode for reliability.
            
            print(f"[VGGT] Model loaded successfully from {self.weights_path}")
            return True
        except Exception as e:
            print(f"[VGGT] Error loading model: {e}")
            return False
    
    def preprocess_frames_fast(self, frames):
        """Fast preprocessing with minimal allocations.
        
        Args:
            frames: List of numpy arrays (BGR images from OpenCV)
            
        Returns:
            Preprocessed tensor ready for model input
        """
        n_frames = len(frames)
        
        # Get target dimensions from first frame
        h, w = frames[0].shape[:2]
        target_size = self._target_size
        scale = target_size / max(h, w)
        new_h = max(14, (int(h * scale) // 14) * 14)
        new_w = max(14, (int(w * scale) // 14) * 14)
        
        # Pre-allocate output array
        output = np.empty((n_frames, 3, new_h, new_w), dtype=np.float32)
        
        for i, frame in enumerate(frames):
            # Fused BGR->RGB conversion with resize
            # cv2.resize is already optimized for INTER_AREA
            resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # Convert BGR to RGB and normalize in one step
            # Transpose HWC -> CHW and normalize
            rgb = resized[:, :, ::-1]  # BGR to RGB (view, no copy)
            output[i] = np.transpose(rgb, (2, 0, 1)) / 255.0
        
        # Convert to tensor with pin_memory for faster GPU transfer
        tensor = torch.from_numpy(output).unsqueeze(0)  # (1, N, 3, H, W)
        
        if self.device == "cuda":
            tensor = tensor.pin_memory().to(self.device, non_blocking=True)
        else:
            tensor = tensor.to(self.device)
        
        return tensor
    
    def run_inference(self, frames):
        """Run VGGT inference with optimizations.
        
        Args:
            frames: List of numpy arrays (BGR images from OpenCV)
            
        Returns:
            Dictionary with point cloud data including binary-ready arrays
        """
        if self.model is None:
            return self._empty_result()
        
        with self.lock:
            start_time = time.perf_counter()
            
            try:
                # Fast preprocessing
                images = self.preprocess_frames_fast(frames)
                
                # Run inference (simplified - no CUDA streams for reliability)
                print(f"[VGGT] Running model inference...")
                with torch.inference_mode():
                    if self.device == "cuda":
                        with torch.amp.autocast(device_type=self.device, dtype=self.dtype):
                            predictions = self.model(images)
                    else:
                        predictions = self.model(images.to(self.dtype))
                print(f"[VGGT] Model inference complete")
                
                # Process predictions (optimized)
                point_cloud = self._process_predictions_fast(predictions, images)
                
                inference_time = time.perf_counter() - start_time
                point_cloud['inference_time'] = inference_time
                point_cloud['success'] = True
                
                # Cache results
                self.last_point_cloud = point_cloud
                self.last_inference_time = time.time()
                
                return point_cloud
                
            except Exception as e:
                print(f"[VGGT] Inference error: {e}")
                import traceback
                traceback.print_exc()
                return self._empty_result(time.perf_counter() - start_time)
    
    def _process_predictions_fast(self, predictions, images):
        """Optimized prediction processing with minimal allocations."""
        
        # Helper to convert tensor to numpy (handles bfloat16)
        def to_numpy(tensor):
            if tensor is None:
                return None
            t = tensor.detach()
            # Convert bfloat16/float16 to float32 for numpy compatibility
            if t.dtype in (torch.bfloat16, torch.float16):
                t = t.float()
            return t.cpu().numpy()
        
        # Convert pose encoding to extrinsic and intrinsic matrices
        extrinsic, intrinsic = pose_encoding_to_extri_intri(
            predictions["pose_enc"], 
            images.shape[-2:]
        )
        
        # Convert tensors to numpy efficiently
        depth_map = to_numpy(predictions["depth"]).squeeze(0)
        depth_conf = predictions.get("depth_conf", None)
        if depth_conf is not None:
            depth_conf = to_numpy(depth_conf).squeeze(0)
        
        extrinsic = to_numpy(extrinsic).squeeze(0) if isinstance(extrinsic, torch.Tensor) else extrinsic
        intrinsic = to_numpy(intrinsic).squeeze(0) if isinstance(intrinsic, torch.Tensor) else intrinsic
        
        # Generate world points
        world_points = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)
        
        # Get images for colors (convert bfloat16 if needed)
        images_np = to_numpy(images).squeeze(0)  # (N, 3, H, W)
        images_np = np.transpose(images_np, (0, 2, 3, 1))  # (N, H, W, 3)
        
        # Flatten efficiently using reshape (not flatten)
        total_points = np.prod(world_points.shape[:-1])
        points_flat = world_points.reshape(-1, 3)
        colors_flat = (images_np.reshape(-1, 3) * 255).astype(np.uint8)
        
        # Confidence filtering (vectorized)
        if depth_conf is not None:
            conf_flat = depth_conf.reshape(-1)
        else:
            conf_flat = np.ones(total_points, dtype=np.float32)
        
        # Compute threshold using percentile
        if self.conf_threshold_pct > 0:
            conf_threshold = np.percentile(conf_flat, self.conf_threshold_pct)
        else:
            conf_threshold = 0
        
        # Create combined mask (vectorized operations)
        color_sum = colors_flat[:, 0].astype(np.int32) + colors_flat[:, 1].astype(np.int32) + colors_flat[:, 2].astype(np.int32)
        mask = (conf_flat >= conf_threshold) & (conf_flat > 1e-5) & (color_sum >= 16)
        
        # Apply mask
        points_filtered = points_flat[mask]
        colors_filtered = colors_flat[mask]
        
        # Subsample if needed (random choice is fast)
        n_points = len(points_filtered)
        if n_points > self.max_points:
            # Use stride-based sampling instead of random (faster, deterministic)
            stride = n_points // self.max_points
            indices = np.arange(0, n_points, stride)[:self.max_points]
            points_filtered = points_filtered[indices]
            colors_filtered = colors_filtered[indices]
        
        # Convert to float32 for transmission (already is, but ensure contiguous)
        # Ensure shape is (N, 3) for proper binary serialization
        points_out = np.ascontiguousarray(points_filtered.reshape(-1, 3), dtype=np.float32)
        colors_out = np.ascontiguousarray(colors_filtered.reshape(-1, 3), dtype=np.uint8)
        
        num_points = len(points_out)
        print(f"[VGGT] Processed: {num_points} points, shapes: points={points_out.shape}, colors={colors_out.shape}")
        
        return {
            'points': points_out,
            'colors': colors_out,
            'num_points': num_points
        }
    
    def _empty_result(self, inference_time=0):
        """Create empty result structure."""
        return {
            'points': np.array([], dtype=np.float32).reshape(0, 3),
            'colors': np.array([], dtype=np.uint8).reshape(0, 3),
            'num_points': 0,
            'inference_time': inference_time,
            'success': False
        }
    
    def get_last_point_cloud(self):
        """Get the last computed point cloud."""
        return self.last_point_cloud
    
    def get_point_cloud_binary(self):
        """Get point cloud as binary data for fast transmission.
        
        Returns:
            tuple: (points_bytes, colors_bytes, num_points) or None
        """
        if self.last_point_cloud is None or not self.last_point_cloud.get('success', False):
            return None
        
        pc = self.last_point_cloud
        return (
            pc['points'].tobytes(),
            pc['colors'].tobytes(),
            pc['num_points']
        )


def load_vggt_detector(weights_path="weights/vggt.pt", conf_threshold_pct=50.0, max_points=100000):
    """Load and initialize a VGGT detector."""
    detector = VGGTDetector(
        weights_path=weights_path,
        conf_threshold_pct=conf_threshold_pct,
        max_points=max_points
    )
    
    if detector.load_model():
        return detector
    return None


def create_empty_point_cloud():
    """Create an empty point cloud result structure."""
    return {
        'points': np.array([], dtype=np.float32).reshape(0, 3),
        'colors': np.array([], dtype=np.uint8).reshape(0, 3),
        'num_points': 0,
        'inference_time': 0,
        'success': False
    }
