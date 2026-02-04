"""
LSTM-based error detection for assembly tasks - FIXED VERSION

This module provides real-time error detection by analyzing sequences of 
YOLO detections using a trained LSTM model.

CRITICAL FIXES:
1. Window size matches training 
2. Frame stride matches training 
3. Post-processing matches training exactly
4. Multi-camera fusion support added
"""

import os
import numpy as np
import torch
import pickle
import scipy.ndimage as ndi
from collections import deque
from itertools import combinations

# =============================================================================
# Configuration
# =============================================================================

DEFAULT_MODEL_PATH = "3d_scene/final_lstm_model/final_model.pt"
DEFAULT_CONFIG_PATH = "3d_scene/final_lstm_model/inference_config.pkl"

# Feature extraction parameters (must match training)
MAX_BAT = 6  # Maximum number of batteries to encode

# =============================================================================
# Feature Extraction (from training pipeline)
# =============================================================================

def pairwise_dists(centroids, max_len):
    """Flattened pairwise Euclidean distances, padded/truncated."""
    if len(centroids) < 2:
        return [0.0] * max_len

    pts = np.asarray(centroids, dtype=np.float32)
    dists = [
        float(np.linalg.norm(pts[i] - pts[j]))
        for i, j in combinations(range(len(pts)), 2)
    ]

    if len(dists) < max_len:
        dists += [0.0] * (max_len - len(dists))
    else:
        dists = dists[:max_len]

    return dists


def extract_features_from_yolo(yolo_results, frame_shape):
    """Extract features from YOLO detection results.
    
    Args:
        yolo_results: YOLO results object (from YOLODetector.detect())
        frame_shape: Tuple of (height, width) of the frame
        
    Returns:
        numpy array of shape (36,) with extracted features
    """
    # Initialize empty detection
    frame_record = {
        "case": None,
        "batteries": []
    }
    
    if yolo_results is None:
        return build_feat_from_frame(frame_record, frame_shape)
    
    # Extract detections from YOLO results
    result = yolo_results[0] if yolo_results else None
    if result is None or result.masks is None:
        return build_feat_from_frame(frame_record, frame_shape)
    
    # Class IDs (must match YOLO training)
    CASE_ID = 1
    BATTERY_ID = 3
    
    try:
        boxes_cls = result.boxes.cls.cpu().numpy().astype(np.int32)
        boxes_xyxy = result.boxes.xyxy.cpu().numpy()
        masks_xy = result.masks.xy
        
        for j, poly_np in enumerate(masks_xy):
            class_id = boxes_cls[j]
            
            if class_id == CASE_ID:
                # Extract case bounding box
                x1, y1, x2, y2 = map(float, boxes_xyxy[j])
                frame_record["case"] = {
                    "bbox": [x1, y1, x2, y2]
                }
            
            elif class_id == BATTERY_ID:
                # Extract battery centroid
                if len(poly_np) > 0:
                    cx = float(np.mean(poly_np[:, 0]))
                    cy = float(np.mean(poly_np[:, 1]))
                    frame_record["batteries"].append({
                        "centroid": (cx, cy)
                    })
    except Exception as e:
        print(f"[LSTM] Feature extraction error: {e}")
    
    return build_feat_from_frame(frame_record, frame_shape)


def build_feat_from_frame(frame, frame_shape=None, max_bat=MAX_BAT):
    """Build 36-dimensional feature vector from frame detections.
    
    Matches the feature extraction from training pipeline.
    
    Args:
        frame: Dict with 'case' and 'batteries' keys
        frame_shape: Optional tuple of (height, width)
        max_bat: Maximum number of batteries to encode
        
    Returns:
        numpy array of shape (36,)
    """
    bats = frame.get("batteries", [])

    centroids = [
        b.get("centroid", (0.0, 0.0)) for b in bats
    ]
    centroids = sorted(
        centroids, key=lambda c: (float(c[0]), float(c[1]))
    )[:max_bat]

    # 1) number of batteries
    n_batts = float(len(bats))

    # 2) flattened centroids
    cent_flat = []
    for cx, cy in centroids:
        cent_flat.extend([float(cx), float(cy)])
    while len(cent_flat) < max_bat * 2:
        cent_flat.append(0.0)

    # 3) pairwise distances
    max_pw = (max_bat * (max_bat - 1)) // 2
    pw = pairwise_dists(centroids, max_pw)

    # 4) case bbox center + size
    case = frame.get("case")
    if case and isinstance(case, dict) and case.get("bbox") is not None:
        try:
            x1, y1, x2, y2 = case["bbox"]
            case_cx = (x1 + x2) / 2.0
            case_cy = (y1 + y2) / 2.0
            case_w = x2 - x1
            case_h = y2 - y1
        except Exception:
            case_cx = case_cy = case_w = case_h = 0.0
    else:
        case_cx = case_cy = case_w = case_h = 0.0

    # 5) centroid mean + std
    if centroids:
        pts = np.asarray(centroids, dtype=np.float32)
        mean_cx, mean_cy = pts[:, 0].mean(), pts[:, 1].mean()
        std_cx, std_cy = pts[:, 0].std(), pts[:, 1].std()
    else:
        mean_cx = mean_cy = std_cx = std_cy = 0.0

    feat = (
        [n_batts]
        + cent_flat
        + pw
        + [case_cx, case_cy, case_w, case_h]
        + [mean_cx, mean_cy, std_cx, std_cy]
    )

    return np.asarray(feat, dtype=np.float32)


# =============================================================================
# LSTM Model Definition (must match training)
# =============================================================================

class FrameLSTM(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=128,
        num_layers=2,
        bidirectional=False,
        dropout=0.3,
    ):
        super().__init__()

        self.lstm = torch.nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.head = torch.nn.Linear(out_dim, 1)

    def forward(self, x, mask=None):
        """
        x: (B, T, F)
        """
        h, _ = self.lstm(x)
        logits = self.head(h)
        return logits.squeeze(-1)


# =============================================================================
# Post-processing Functions
# =============================================================================

def smooth_and_minrun(probs, threshold, smooth_k=5, min_run=3):
    """Apply temporal smoothing and minimum run length filtering.
    
    Args:
        probs: Array of probabilities
        threshold: Binary classification threshold
        smooth_k: Kernel size for smoothing
        min_run: Minimum consecutive frames for an error event
        
    Returns:
        Binary predictions array
    """
    if len(probs) < smooth_k:
        # Not enough frames for smoothing
        return (probs > threshold).astype(int)
    
    sm = ndi.uniform_filter1d(probs, size=smooth_k)
    preds = (sm > threshold).astype(int)
    
    if min_run <= 1:
        return preds
    
    out = preds.copy()
    i = 0
    while i < len(preds):
        if preds[i] == 1:
            j = i
            while j < len(preds) and preds[j] == 1:
                j += 1
            if j - i < min_run:
                out[i:j] = 0
            i = j
        else:
            i += 1
    return out


def apply_min_error_persistence(preds, min_duration_sec=3.0, fps=2.0):
    """Enforce minimum error duration (streaming-safe, causal).
    
    Args:
        preds: Binary predictions (1D array)
        min_duration_sec: Minimum error duration in seconds
        fps: Effective frame rate (frames per second)
        
    Returns:
        Binary predictions with enforced minimum duration
    """
    min_frames = int(min_duration_sec * fps)
    output = preds.copy()
    
    i = 0
    while i < len(preds):
        if preds[i] == 1:
            j = i
            while j < len(preds) and preds[j] == 1:
                j += 1
            
            error_duration = j - i
            
            if error_duration < min_frames:
                end_idx = min(i + min_frames, len(preds))
                output[i:end_idx] = 1
                i = end_idx
            else:
                i = j
        else:
            i += 1
    
    return output


# =============================================================================
# LSTM Error Detector
# =============================================================================

class LSTMErrorDetector:
    """Real-time error detection using LSTM on YOLO detections.
    
    Maintains a sliding window buffer and runs LSTM inference to detect
    assembly errors in real-time.
    
    Matches training:
    - Window size
    - Frame stride 
    - Post-processing: smooth_and_minrun + min_error_persistence
    """
    
    def __init__(self, model, config, device="cpu"):
        """Initialize the error detector.
        
        Args:
            model: Loaded FrameLSTM model
            config: Configuration dict with threshold, smooth_k, min_run
            device: Device for inference ('cpu', 'cuda', or 'mps')
        """
        self.model = model
        self.model.eval()
        self.device = device
        
        # Configuration from training
        self.threshold = config.get('threshold', 0.5)
        self.smooth_k = config.get('smooth_k', 5)
        self.min_run = config.get('min_run', 3)
        self.min_error_duration_sec = config.get('min_error_duration_sec', 3.0)
        self.effective_fps = config.get('effective_fps', 2.0)
        self.input_dim = config.get('input_dim', 36)
        
        # CRITICAL FIX: Window size must match training
        self.window_size = config.get('window_size', 15)  # From training config
        self.feature_buffer = deque(maxlen=self.window_size)
        
        # CRITICAL FIX: Frame stride - only process every Nth frame to match training
        self.frame_stride = 5  # Match training 
        self.frame_counter = 0
        
        # Cache for predictions (to avoid recomputing)
        self.prob_buffer = deque(maxlen=100)  # Keep longer history for post-processing
        self.pred_buffer = deque(maxlen=100)
        
        # Current error state
        self.current_error = False
        self.error_confidence = 0.0
        
        print(f"[LSTM] Initialized error detector:")
        print(f"  Threshold: {self.threshold:.2f}")
        print(f"  Smooth K: {self.smooth_k}")
        print(f"  Min Run: {self.min_run}")
        print(f"  Min Error Duration: {self.min_error_duration_sec}s")
        print(f"  Window Size: {self.window_size} frames (MATCHES TRAINING)")
        print(f"  Frame Stride: {self.frame_stride} (MATCHES TRAINING)")
        print(f"  Effective FPS: {self.effective_fps:.1f}")
        print(f"  Device: {device}")
    
    def reset(self):
        """Reset the detector state."""
        self.feature_buffer.clear()
        self.prob_buffer.clear()
        self.pred_buffer.clear()
        self.current_error = False
        self.error_confidence = 0.0
        self.frame_counter = 0
    
    def add_frame(self, features):
        """Add a new frame's features to the buffer.
        
        Args:
            features: numpy array of shape (36,)
        """
        self.feature_buffer.append(features)
    
    def detect(self, yolo_results, frame_shape):
        """Run error detection on current frame.
        
        CRITICAL: Only processes every Nth frame to match training stride.
        
        Args:
            yolo_results: YOLO detection results
            frame_shape: Tuple of (height, width)
            
        Returns:
            Dict with detection results:
                - error_detected: bool
                - confidence: float (0-1)
                - raw_probability: float
                - buffer_size: int
                - processed: bool (False if frame was skipped due to stride)
        """
        self.frame_counter += 1
        
        # Only process every Nth frame to match training
        if (self.frame_counter % self.frame_stride) != 0:
            # Return cached result without processing
            return {
                'error_detected': self.current_error,
                'confidence': self.error_confidence,
                'raw_probability': self.error_confidence,
                'buffer_size': len(self.feature_buffer),
                'ready': len(self.feature_buffer) >= self.window_size,
                'processed': False  # This frame was skipped
            }
        
        # Extract features from YOLO detections
        features = extract_features_from_yolo(yolo_results, frame_shape)
        
        # Add to buffer
        self.add_frame(features)
        
        # Need minimum frames for meaningful inference
        if len(self.feature_buffer) < self.window_size:
            return {
                'error_detected': False,
                'confidence': 0.0,
                'raw_probability': 0.0,
                'buffer_size': len(self.feature_buffer),
                'ready': False,
                'processed': True
            }
        
        # Prepare input for LSTM (batch_size=1, seq_len=window_size, features=36)
        seq = np.stack(list(self.feature_buffer), axis=0)  # (T, F)
        seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(self.device)  # (1, T, F)
        
        # Run inference
        with torch.no_grad():
            logits = self.model(seq_tensor)  # (1, T)
            probs = torch.sigmoid(logits).cpu().numpy()[0]  # (T,)
        
        # Get current frame probability (last in sequence)
        current_prob = float(probs[-1])
        
        # Update probability buffer
        self.prob_buffer.append(current_prob)
        
        # Apply post-processing to entire buffer (MATCHES TRAINING)
        if len(self.prob_buffer) >= self.smooth_k:
            probs_array = np.array(list(self.prob_buffer))
            
            # Step 1: Smooth and threshold (matches training)
            preds_array = smooth_and_minrun(probs_array, self.threshold, 
                                           self.smooth_k, self.min_run)
            
            # Step 2: Apply minimum error persistence (matches training)
            preds_array = apply_min_error_persistence(preds_array, 
                                                     self.min_error_duration_sec,
                                                     self.effective_fps)
            
            # Update prediction buffer
            self.pred_buffer = deque(preds_array, maxlen=100)
            
            # Current prediction is the last one
            self.current_error = bool(preds_array[-1])
            self.error_confidence = float(probs_array[-1])
        else:
            # Not enough for smoothing, use raw threshold
            self.current_error = current_prob > self.threshold
            self.error_confidence = current_prob
        
        return {
            'error_detected': self.current_error,
            'confidence': self.error_confidence,
            'raw_probability': current_prob,
            'buffer_size': len(self.feature_buffer),
            'ready': True,
            'processed': True
        }
    
    def get_status(self):
        """Get current error detection status.
        
        Returns:
            Dict with current state
        """
        return {
            'error_detected': self.current_error,
            'confidence': self.error_confidence,
            'buffer_size': len(self.feature_buffer),
            'ready': len(self.feature_buffer) >= self.window_size
        }


# =============================================================================
# Multi-Camera Fusion
# =============================================================================

class MultiCameraLSTMFusion:
    """Fuses LSTM predictions across multiple cameras using K-out-of-N voting.
    
    Instead of averaging probabilities, this collects binary predictions from
    each camera and requires K cameras to agree before flagging an error.
    
    This is more interpretable and robust to single-camera noise.
    """
    
    def __init__(self, detectors_dict, config, min_cameras_for_error=None):
        """Initialize multi-camera fusion with K-out-of-N voting.
        
        Args:
            detectors_dict: Dict of camera_id -> LSTMErrorDetector
            config: Shared configuration dict
            min_cameras_for_error: K value (number of cameras that must agree)
                                  None = auto-set to majority (ceil(N/2))
        """
        self.detectors = detectors_dict
        self.config = config
        
        # K-out-of-N voting threshold
        self.num_cameras = len(detectors_dict)
        if min_cameras_for_error is None:
            # Default to majority voting
            self.min_cameras_for_error = (self.num_cameras // 2) + 1
        else:
            self.min_cameras_for_error = min_cameras_for_error
        
        # Configuration for post-processing
        self.threshold = config.get('threshold', 0.5)
        self.smooth_k = config.get('smooth_k', 5)
        self.min_run = config.get('min_run', 3)
        self.min_error_duration_sec = config.get('min_error_duration_sec', 3.0)
        self.effective_fps = config.get('effective_fps', 2.0)
        
        # Buffer for global binary predictions
        self.global_binary_buffer = deque(maxlen=100)
        self.current_fused_error = False
        self.current_fused_confidence = 0.0
        
        print(f"[LSTM Fusion] Initialized with {self.num_cameras} cameras")
        print(f"[LSTM Fusion] K-out-of-N voting: K={self.min_cameras_for_error} "
              f"(requires {self.min_cameras_for_error}/{self.num_cameras} cameras to agree)")
    
    def reset(self):
        """Reset all detectors and fusion state."""
        for detector in self.detectors.values():
            detector.reset()
        self.global_binary_buffer.clear()
        self.current_fused_error = False
        self.current_fused_confidence = 0.0
    
    def set_voting_threshold(self, K):
        """Change the K value for K-out-of-N voting.
        
        Args:
            K: Number of cameras that must agree (1 to num_cameras)
        """
        if 1 <= K <= self.num_cameras:
            self.min_cameras_for_error = K
            print(f"[LSTM Fusion] K-out-of-N voting updated: K={K}/{self.num_cameras}")
        else:
            print(f"[LSTM Fusion] Invalid K={K}, must be 1-{self.num_cameras}")
    
    def detect(self, yolo_results_dict, frame_shape):
        """Run fused detection using K-out-of-N voting on binary predictions.
        
        Algorithm:
          1. Get binary prediction from each camera
          2. Count how many cameras predict ERROR
          3. If count >= K, then global_error = 1, else global_error = 0
          4. Apply post-processing to global binary sequence
        
        Args:
            yolo_results_dict: Dict of camera_id -> yolo_results
            frame_shape: Tuple of (height, width)
            
        Returns:
            Dict with fused detection results
        """
        # Step 1: Collect binary predictions and probabilities from each camera
        camera_binaries = {}
        camera_probs = {}
        all_ready = True
        
        for cam_id, detector in self.detectors.items():
            yolo_results = yolo_results_dict.get(cam_id)
            result = detector.detect(yolo_results, frame_shape)
            
            if result['ready'] and result['processed']:
                # Get binary prediction (after camera's own post-processing)
                camera_binaries[cam_id] = 1 if result['error_detected'] else 0
                camera_probs[cam_id] = result['raw_probability']
            else:
                all_ready = False
        
        if not all_ready or len(camera_binaries) == 0:
            return {
                'error_detected': self.current_fused_error,
                'confidence': self.current_fused_confidence,
                'cameras_ready': len(camera_binaries),
                'cameras_total': len(self.detectors),
                'cameras_in_error': 0,
                'min_cameras_for_error': self.min_cameras_for_error,
                'ready': False
            }
        
        # Step 2: Count how many cameras are in ERROR state
        cameras_in_error = sum(camera_binaries.values())
        
        # Step 3: K-out-of-N voting
        # If >= K cameras say ERROR, then global ERROR
        global_binary = 1 if cameras_in_error >= self.min_cameras_for_error else 0
        
        # Add to buffer
        self.global_binary_buffer.append(global_binary)
        
        # Step 4: Apply post-processing to global binary sequence
        if len(self.global_binary_buffer) >= self.smooth_k:
            # Convert deque to numpy array
            global_sequence = np.array(list(self.global_binary_buffer))
            
            # Apply minimum run length filter
            processed_sequence = self._apply_min_run(global_sequence, self.min_run)
            
            # Apply minimum error persistence
            processed_sequence = apply_min_error_persistence(
                processed_sequence,
                self.min_error_duration_sec,
                self.effective_fps
            )
            
            # Update current state
            self.current_fused_error = bool(processed_sequence[-1])
            
            # Confidence = average probability of cameras in error state
            if cameras_in_error > 0:
                error_cam_probs = [camera_probs[cid] for cid, binary in camera_binaries.items() if binary == 1]
                self.current_fused_confidence = float(np.mean(error_cam_probs))
            else:
                # No cameras in error, use average of all
                self.current_fused_confidence = float(np.mean(list(camera_probs.values())))
        else:
            # Not enough history, use raw result
            self.current_fused_error = bool(global_binary)
            self.current_fused_confidence = float(np.mean(list(camera_probs.values())))
        
        return {
            'error_detected': self.current_fused_error,
            'confidence': self.current_fused_confidence,
            'cameras_ready': len(camera_binaries),
            'cameras_total': len(self.detectors),
            'cameras_in_error': cameras_in_error,
            'min_cameras_for_error': self.min_cameras_for_error,
            'camera_binaries': camera_binaries,  # Which cameras detected error
            'camera_probs': camera_probs,  # Raw probabilities
            'ready': True
        }
    
    def _apply_min_run(self, binary_sequence, min_run):
        """Apply minimum run length to binary sequence.
        
        Remove error runs shorter than min_run frames.
        
        Args:
            binary_sequence: 1D numpy array of 0s and 1s
            min_run: Minimum consecutive frames for an error
            
        Returns:
            Filtered binary sequence
        """
        if min_run <= 1:
            return binary_sequence
        
        output = binary_sequence.copy()
        i = 0
        while i < len(binary_sequence):
            if binary_sequence[i] == 1:
                # Found start of error run
                j = i
                while j < len(binary_sequence) and binary_sequence[j] == 1:
                    j += 1
                # Run length is j - i
                if j - i < min_run:
                    # Too short, remove it
                    output[i:j] = 0
                i = j
            else:
                i += 1
        return output
    
    def get_status(self):
        """Get current fused detection status."""
        cameras_ready = sum(1 for d in self.detectors.values() if d.get_status()['ready'])
        
        return {
            'error_detected': self.current_fused_error,
            'confidence': self.current_fused_confidence,
            'cameras_ready': cameras_ready,
            'cameras_total': len(self.detectors),
            'min_cameras_for_error': self.min_cameras_for_error,
            'ready': cameras_ready >= self.min_cameras_for_error  # !!!
        }

# =============================================================================
# Loading Functions
# =============================================================================

def load_lstm_model(model_path=None, config_path=None, device=None):
    """Load trained LSTM error detection model.
    
    Args:
        model_path: Path to model weights (.pt file)
        config_path: Path to inference config (.pkl file)
        device: Device for inference (auto-detected if None)
        
    Returns:
        LSTMErrorDetector instance or None on failure
    """
    if model_path is None:
        model_path = DEFAULT_MODEL_PATH
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH
    
    try:
        # Auto-detect device
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        
        # Load configuration
        if not os.path.exists(config_path):
            print(f"[LSTM] Config not found: {config_path}")
            return None
        
        with open(config_path, 'rb') as f:
            config = pickle.load(f)
        
        # Load model weights
        if not os.path.exists(model_path):
            print(f"[LSTM] Model not found: {model_path}")
            return None
        
        print(f"[LSTM] Loading model: {model_path}")
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        
        # Create model
        input_dim = config.get('input_dim', 36)
        model = FrameLSTM(input_dim=input_dim)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        # Create detector
        detector = LSTMErrorDetector(model, config, device=device)
        
        print(f"[LSTM] Model ready on {device}")
        print(f"[LSTM] Expected performance: Event F1={config['expected_performance']['fused_event_f1']:.3f}")
        
        return detector
        
    except Exception as e:
        print(f"[LSTM] Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return None


def load_multi_camera_lstm(model_path=None, config_path=None, camera_ids=None, device=None):
    """Load LSTM models for multiple cameras with fusion.
    
    Args:
        model_path: Path to model weights (.pt file)
        config_path: Path to inference config (.pkl file)
        camera_ids: List of camera IDs to create detectors for
        device: Device for inference (auto-detected if None)
        
    Returns:
        MultiCameraLSTMFusion instance or None on failure
    """
    if model_path is None:
        model_path = DEFAULT_MODEL_PATH
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH
    
    try:
        # Auto-detect device
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        
        # Load configuration
        if not os.path.exists(config_path):
            print(f"[LSTM] Config not found: {config_path}")
            return None
        
        with open(config_path, 'rb') as f:
            config = pickle.load(f)
        
        # Load model weights once
        if not os.path.exists(model_path):
            print(f"[LSTM] Model not found: {model_path}")
            return None
        
        print(f"[LSTM Multi-Camera] Loading model: {model_path}")
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        
        input_dim = config.get('input_dim', 36)
        
        # Create detector for each camera (sharing same model weights)
        detectors = {}
        for cam_id in camera_ids:
            # Create model instance
            model = FrameLSTM(input_dim=input_dim)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            model.eval()
            
            # Create detector
            detector = LSTMErrorDetector(model, config, device=device)
            detectors[cam_id] = detector
        
        # Create fusion manager
        fusion = MultiCameraLSTMFusion(detectors, config)
        
        print(f"[LSTM Multi-Camera] Ready on {device}")
        print(f"[LSTM Multi-Camera] Expected fused performance: Event F1={config['expected_performance']['fused_event_f1']:.3f}")
        
        return fusion
        
    except Exception as e:
        print(f"[LSTM Multi-Camera] Failed to load: {e}")
        import traceback
        traceback.print_exc()
        return None