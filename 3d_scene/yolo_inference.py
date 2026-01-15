"""
YOLO inference module for instance segmentation.

This module provides YOLO-based instance segmentation with visualization
utilities for drawing masks and labels on video frames.
"""

import os
import cv2
import numpy as np


# =============================================================================
# Configuration
# =============================================================================

# Class names for YOLO detection
CLASSES = ["person", "case", "case_top", "battery", "screw", "tool"]

# Pre-defined colors for each class (BGR format for OpenCV)
CLASS_COLORS = np.array([
    [50, 0, 0],       # person: blueish
    [0, 165, 255],    # case: orange
    [0, 40, 75],      # case_top: yellow
    [192, 192, 192],  # battery: silver
    [140, 0, 140],    # screw: violet
    [0, 200, 0]       # tool: green
], dtype=np.uint8)

# Default model path
DEFAULT_MODEL_PATH = os.path.join(
    'yolov11_finetuned', 'runs', 'segment', 
    'yolov11n_seg_custom', 'weights', 'best.pt'
)


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
            print("[Device] MPS (Apple Silicon GPU) available")
            return "mps"
        elif torch.cuda.is_available():
            print(f"[Device] CUDA available: {torch.cuda.get_device_name(0)}")
            return "cuda"
        else:
            print("[Device] Using CPU")
            return "cpu"
    except ImportError:
        print("[Device] PyTorch not available, using CPU")
        return "cpu"
    except Exception as e:
        print(f"[Device] Error detecting device: {e}, using CPU")
        return "cpu"


# =============================================================================
# YOLO Wrapper
# =============================================================================

class YOLODetector:
    """Wrapper for YOLO instance segmentation model.
    
    Provides a clean interface for running YOLO inference and drawing
    segmentation masks on video frames.
    
    Attributes:
        model: The loaded YOLO model
        device: Device used for inference ('mps', 'cuda', or 'cpu')
        classes: List of class names
        colors: Array of BGR colors for each class
    """
    
    def __init__(self, model, device="cpu", classes=None, colors=None):
        """Initialize the YOLO detector.
        
        Args:
            model: Loaded YOLO model from ultralytics
            device: Device for inference ('mps', 'cuda', or 'cpu')
            classes: Optional list of class names (defaults to CLASSES)
            colors: Optional numpy array of BGR colors (defaults to CLASS_COLORS)
        """
        self.model = model
        self.device = device
        self.classes = classes if classes is not None else CLASSES
        self.colors = colors if colors is not None else CLASS_COLORS
        self.font = cv2.FONT_HERSHEY_SIMPLEX
    
    def detect(self, frame, conf=0.35, iou=0.45, max_det=20, imgsz=640):
        """Run YOLO inference on a frame.
        
        Args:
            frame: BGR image (numpy array from OpenCV)
            conf: Confidence threshold
            iou: IoU threshold for NMS
            max_det: Maximum number of detections
            imgsz: Input image size for the model
            
        Returns:
            YOLO results object or None on error
        """
        try:
            results = self.model(
                frame,
                verbose=False,
                conf=conf,
                iou=iou,
                max_det=max_det,
                imgsz=imgsz,
                device=self.device
            )
            return results
        except Exception as e:
            print(f"[YOLO] Detection error: {e}")
            return None
    
    def draw_predictions(self, frame, results, target_width=None, target_height=None, 
                         alpha=0.4):
        """Draw YOLO segmentation masks and labels on frame.
        
        Args:
            frame: BGR image to draw on
            results: YOLO results from detect()
            target_width: Target width for scaling (optional)
            target_height: Target height for scaling (optional)
            alpha: Opacity for mask overlay (0-1)
            
        Returns:
            Frame with drawn predictions
        """
        if results is None:
            return frame
        
        # Check if any results have masks
        has_masks = any(r.masks is not None and len(r.masks.xy) > 0 for r in results)
        if not has_masks:
            return frame
        
        # Use frame dimensions if target not specified
        if target_width is None:
            target_width = frame.shape[1]
        if target_height is None:
            target_height = frame.shape[0]
        
        # Create overlay
        overlay = frame.copy()
        
        for r in results:
            if r.masks is None:
                continue
            
            masks_xy = r.masks.xy
            boxes_cls = r.boxes.cls.cpu().numpy().astype(np.int32)
            boxes_conf = r.boxes.conf.cpu().numpy()
            
            for j, poly_np in enumerate(masks_xy):
                if len(poly_np) < 3:
                    continue
                
                class_id = boxes_cls[j]
                conf = boxes_conf[j]
                
                # Scale polygon to target resolution
                h_scale = target_height / r.orig_shape[0]
                w_scale = target_width / r.orig_shape[1]
                scaled_poly = poly_np.copy()
                scaled_poly[:, 0] *= w_scale
                scaled_poly[:, 1] *= h_scale
                
                # Get color for this class
                color = tuple(int(c) for c in self.colors[class_id % len(self.colors)])
                pts = scaled_poly.astype(np.int32).reshape((-1, 1, 2))
                
                # Draw filled polygon on overlay
                cv2.fillPoly(overlay, [pts], color)
                # Draw polygon outline on frame
                cv2.polylines(frame, [pts], True, color, 1)
                
                # Draw label
                class_name = self.classes[class_id] if class_id < len(self.classes) else '?'
                label = f"{class_name} {conf:.2f}"
                tx, ty = int(scaled_poly[0][0]), int(scaled_poly[0][1]) - 3
                if ty < 10:
                    ty = int(scaled_poly[0][1]) + 10
                
                cv2.putText(frame, label, (tx, ty), self.font, 0.2, 
                           (255, 255, 255), 1, cv2.LINE_AA)
        
        # Blend overlay with frame
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        return frame


# =============================================================================
# Loading Functions
# =============================================================================

def load_yolo_model(model_path=None, warmup=True):
    """Load YOLO model with automatic device selection.
    
    Args:
        model_path: Path to YOLO weights (.pt file). 
                   Defaults to DEFAULT_MODEL_PATH
        warmup: Whether to run a warmup inference
        
    Returns:
        tuple: (YOLODetector instance, device string) or (None, None) on failure
    """
    if model_path is None:
        model_path = DEFAULT_MODEL_PATH
    
    try:
        from ultralytics import YOLO
        
        # Detect best device
        device = get_best_device()
        
        if not os.path.exists(model_path):
            print(f"[YOLO] Model not found: {model_path}")
            return None, None
        
        print(f"[YOLO] Loading model: {model_path}")
        model = YOLO(model_path)
        
        if warmup:
            print(f"[YOLO] Warming up on {device}...")
            _ = model(np.zeros((320, 320, 3), dtype=np.uint8), 
                     verbose=False, device=device)
        
        detector = YOLODetector(model, device=device)
        print(f"[YOLO] Model ready on {device}")
        return detector, device
        
    except ImportError:
        print("[YOLO] ultralytics not installed, YOLO disabled")
        return None, None
    except Exception as e:
        print(f"[YOLO] Failed to load model: {e}")
        return None, None
