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
    (0, 0, 70),   # person: bluelish
    (255, 165, 0), # case: orange
    (180, 100, 0),  # case_top:
    (0, 100, 100),  # battery: Grenn
    (128, 0, 128),   # screw: violet
    (0, 100, 0)  # tool: Green
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


def draw_battery_status_overlay(frame, slot_states, error_msg=None, success=False):
    """Draw battery insertion status overlay on frame.
    
    Draws a 2x3 grid showing slot status:
    - Green square: correctly inserted
    - Red square: wrongly inserted
    - Gray square: empty
    
    Layout:
    2 4 6  (top row)
    1 3 5  (bottom row)
    
    Args:
        frame: BGR image to draw on
        slot_states: Dict mapping slot_id -> state ('correct', 'wrong', 'empty')
        error_msg: Optional error message to display at bottom
        success: Whether sequence is complete and successful
        
    Returns:
        Frame with overlay drawn
    """
    h, w = frame.shape[:2]
    
    # Grid parameters (upper right corner)
    grid_size = 20  # Size of each square
    grid_margin = 10  # Margin from edge
    grid_spacing = 2  # Spacing between squares
    
    # Calculate grid position (upper right)
    grid_start_x = w - (3 * grid_size + 2 * grid_spacing + grid_margin)
    grid_start_y = grid_margin
    
    # Slot layout mapping: (row, col) -> slot_id
    # Row 0 (top): 2, 4, 6
    # Row 1 (bottom): 1, 3, 5
    layout = [
        (1, 0, 1),  # row 1, col 0 -> slot 1
        (0, 0, 2),  # row 0, col 0 -> slot 2
        (1, 1, 3),  # row 1, col 1 -> slot 3
        (0, 1, 4),  # row 0, col 1 -> slot 4
        (1, 2, 5),  # row 1, col 2 -> slot 5
        (0, 2, 6),  # row 0, col 2 -> slot 6
    ]
    
    # Color mapping
    color_map = {
        'correct': (0, 255, 0),    # Green
        'wrong': (0, 0, 255),      # Red
        'empty': (100, 100, 100)   # Gray
    }
    
    # Draw semi-transparent background for grid
    overlay = frame.copy()
    bg_x1 = grid_start_x - 5
    bg_y1 = grid_start_y - 5
    bg_x2 = grid_start_x + 3 * grid_size + 2 * grid_spacing + 5
    bg_y2 = grid_start_y + 2 * grid_size + grid_spacing + 5
    cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
    
    # Draw grid squares
    for row, col, slot_id in layout:
        x = grid_start_x + col * (grid_size + grid_spacing)
        y = grid_start_y + row * (grid_size + grid_spacing)
        
        state = slot_states.get(slot_id, 'empty')
        color = color_map[state]
        
        cv2.rectangle(frame, (x, y), (x + grid_size, y + grid_size), color, -1)
        cv2.rectangle(frame, (x, y), (x + grid_size, y + grid_size), (255, 255, 255), 1)
        
        # Draw slot number
        text = str(slot_id)
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
        text_x = x + (grid_size - text_size[0]) // 2
        text_y = y + (grid_size + text_size[1]) // 2
        cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.4, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Draw error message at bottom if present
    if error_msg:
        # Extract expected slot number from message
        # error_msg format: "WRONG SLOT: X inserted (expected Y)"
        if "expected" in error_msg:
            expected = error_msg.split("expected ")[-1].rstrip(")")
            text = f"WRONG SLOT: expected {expected}"
        else:
            text = error_msg
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        
        # Get text size for background
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Position at bottom center
        text_x = (w - text_w) // 2
        text_y = h - 30
        
        # Draw semi-transparent background
        overlay = frame.copy()
        bg_padding = 10
        cv2.rectangle(overlay, 
                     (text_x - bg_padding, text_y - text_h - bg_padding),
                     (text_x + text_w + bg_padding, text_y + baseline + bg_padding),
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Draw text
        cv2.putText(frame, text, (text_x, text_y), font, font_scale, 
                   (0, 0, 255), thickness, cv2.LINE_AA)
    
    # Draw success message at bottom if complete
    if success:
        text = "Correct sequence completed"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        
        # Get text size for background
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Position at bottom center
        text_x = (w - text_w) // 2
        text_y = h - 30
        
        # Draw semi-transparent background
        overlay = frame.copy()
        bg_padding = 10
        cv2.rectangle(overlay, 
                     (text_x - bg_padding, text_y - text_h - bg_padding),
                     (text_x + text_w + bg_padding, text_y + baseline + bg_padding),
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Draw text
        cv2.putText(frame, text, (text_x, text_y), font, font_scale, 
                   (0, 255, 0), thickness, cv2.LINE_AA)
    
    return frame


def draw_lstm_status_overlay(frame, lstm_status):
    """Draw LSTM error detection status in upper left corner.
    
    Args:
        frame: BGR image to draw on
        lstm_status: Dict with 'error_detected', 'confidence', 'ready'
        
    Returns:
        Frame with LSTM overlay drawn
    """
    if lstm_status is None or not lstm_status.get('ready', False):
        return frame
    
    h, w = frame.shape[:2]
    
    # Position in upper left corner
    margin = 10
    box_width = 80
    box_height = 25
    
    x1 = margin
    y1 = margin
    x2 = x1 + box_width
    y2 = y1 + box_height
    
    # Determine status
    error_detected = lstm_status.get('error_detected', False)
    confidence = lstm_status.get('confidence', 0.0)
    
    # Colors: Green for OK, Red for ERROR
    if error_detected:
        bg_color = (0, 0, 200)  # Red background
        text = "WRONG"
        text_color = (255, 255, 255)
    else:
        bg_color = (0, 200, 0)  # Green background
        text = "OK"
        text_color = (255, 255, 255)
    
    # Draw semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), bg_color, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Draw border
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
    
    # Draw text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = x1 + (box_width - text_size[0]) // 2
    text_y = y1 + (box_height + text_size[1]) // 2
    
    cv2.putText(frame, text, (text_x, text_y), font, font_scale, 
               text_color, thickness, cv2.LINE_AA)
    
    # Optional: Show confidence as small text below
    conf_text = f"{confidence:.0%}"
    conf_font_scale = 0.3
    conf_thickness = 1
    conf_y = y2 + 12
    cv2.putText(frame, conf_text, (x1 + 5, conf_y), font, conf_font_scale,
               (255, 255, 255), conf_thickness, cv2.LINE_AA)
    
    return frame

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
    
    def __init__(self, model, device="cpu", classes=None, colors=None, 
                 target_width=640, target_height=360):
        """Initialize the YOLO detector.
        
        Args:
            model: Loaded YOLO model from ultralytics
            device: Device for inference ('mps', 'cuda', or 'cpu')
            classes: Optional list of class names (defaults to CLASSES)
            colors: Optional numpy array of BGR colors (defaults to CLASS_COLORS)
            target_width: Display width for pre-allocated overlay
            target_height: Display height for pre-allocated overlay
        """
        self.model = model
        self.device = device
        self.classes = classes if classes is not None else CLASSES
        self.colors = colors if colors is not None else CLASS_COLORS
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.2
        self.font_thickness = 1
        
        # Pre-allocate overlay buffer (reused each frame to avoid allocation)
        self.target_width = target_width
        self.target_height = target_height
        self.overlay = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        
        # Enable CUDA optimizations
        try:
            import torch
            if torch.cuda.is_available():
                torch.backends.cudnn.benchmark = True
        except ImportError:
            pass
    
    def warmup(self, imgsz=640):
        """Warmup the model with a dummy inference."""
        try:
            import torch
            dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
            _ = self.model(dummy, verbose=False, device=self.device)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"[YOLO] Model warmed up on {self.device}")
        except Exception as e:
            print(f"[YOLO] Warmup warning: {e}")
    
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
                     alpha=0.4, battery_status=None, lstm_status=None):
        """Draw YOLO segmentation masks and labels on frame (optimized).
        
        Uses pre-allocated overlay buffer to avoid memory allocation each frame.
        
        Args:
            frame: BGR image to draw on (modified in place)
            results: YOLO results from detect()
            target_width: Target width for scaling (optional)
            target_height: Target height for scaling (optional)
            alpha: Opacity for mask overlay (0-1)
            battery_status: Optional dict with battery sequence status:
                           {'slot_states': dict, 'error_msg': str, 'success': bool}
            
        Returns:
            Frame with drawn predictions
        """
        if results is None:
            # Still draw battery status overlay if available
            if battery_status:
                frame = draw_battery_status_overlay(
                    frame, 
                    battery_status.get('slot_states', {}),
                    battery_status.get('error_msg'),
                    battery_status.get('success', False)
                )
            if lstm_status:
                frame = draw_lstm_status_overlay(frame, lstm_status)
            return frame
        
        # Quick check for masks (avoid iterator for speed)
        result = results[0] if results else None
        if result is None or result.masks is None or len(result.masks.xy) == 0:
            # Still draw battery status overlay if available
            if battery_status:
                frame = draw_battery_status_overlay(
                    frame, 
                    battery_status.get('slot_states', {}),
                    battery_status.get('error_msg'),
                    battery_status.get('success', False)
                )
            if lstm_status:
                frame = draw_lstm_status_overlay(frame, lstm_status)
            return frame
        
        # Use frame dimensions if target not specified
        if target_width is None:
            target_width = frame.shape[1]
        if target_height is None:
            target_height = frame.shape[0]
        
        # Resize pre-allocated overlay if needed, then copy frame into it
        if self.overlay.shape[0] != target_height or self.overlay.shape[1] != target_width:
            self.overlay = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        np.copyto(self.overlay, frame)
        
        # Pre-compute scale factors once
        h_scale = target_height / result.orig_shape[0]
        w_scale = target_width / result.orig_shape[1]
        
        # Get data from GPU once
        masks_xy = result.masks.xy
        boxes_cls = result.boxes.cls.cpu().numpy().astype(np.int32)
        boxes_conf = result.boxes.conf.cpu().numpy()
        
        for j, poly_np in enumerate(masks_xy):
            if len(poly_np) < 3:
                continue
            
            class_id = boxes_cls[j]
            conf = boxes_conf[j]
            
            # Scale polygon (in-place multiplication)
            scaled_poly = poly_np * [w_scale, h_scale]
            pts = scaled_poly.astype(np.int32).reshape((-1, 1, 2))
            
            # Get color for this class
            color = tuple(int(c) for c in self.colors[class_id % len(self.colors)])
            
            # Draw filled polygon on overlay
            cv2.fillPoly(self.overlay, [pts], color)
            # Draw polygon outline on frame
            cv2.polylines(frame, [pts], True, color, 1)
            
            # Draw label (simplified)
            class_name = self.classes[class_id] if class_id < len(self.classes) else '?'
            label = f"{class_name} {conf:.2f}"
            tx, ty = int(scaled_poly[0][0]), int(scaled_poly[0][1]) - 5
            if ty < 15:
                ty = int(scaled_poly[0][1]) + 15
            
            cv2.putText(frame, label, (tx, ty), self.font, self.font_scale, 
                       (255, 255, 255), self.font_thickness, cv2.LINE_AA)
        
        # Blend overlay with frame (in-place)
        cv2.addWeighted(self.overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Draw battery status overlay if available
        if battery_status:
            frame = draw_battery_status_overlay(
                frame, 
                battery_status.get('slot_states', {}),
                battery_status.get('error_msg'),
                battery_status.get('success', False)
            )
        
        # Draw LSTM status overlay if available
        if lstm_status:
            frame = draw_lstm_status_overlay(frame, lstm_status)
        
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