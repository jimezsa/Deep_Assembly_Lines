import os
import cv2
import asyncio
import json
import yaml
from aiohttp import web
import numpy as np
import time
import threading
from concurrent.futures import ThreadPoolExecutor

# Import inference modules
from dope_inference import load_dope_detector, create_empty_pose
from yolo_inference import load_yolo_model, YOLODetector
from vggt_inference import load_vggt_detector, create_empty_point_cloud
from battery_fsm_module import BatterySequenceTracker
from screw_sequence_tracker import ScrewSequenceTracker, get_tracker

# Try to import orjson for faster JSON serialization
try:
    import orjson

    USE_ORJSON = True
except ImportError:
    USE_ORJSON = False

# =============================================================================
# Configuration
# =============================================================================

# Camera recording directory
RECORDING_DIR = "data/recording_10"
CALIBRATION_FILE = "data/cams_calibrations.yml"

# Camera to run YOLO on
YOLO_CAMERA_ID = "137322071489"

# Cameras for LSTM fusion (must include YOLO_CAMERA_ID)
LSTM_FUSION_CAMERAS = [
    "137322071489",  # Primary YOLO camera
    # "138422075916",
    "141722071426",
    # "141722079467",
    "142122070087",
]

# DOPE configuration
DOPE_ENABLED = True  # Set to False to completely disable DOPE inference
DOPE_CONFIG_PATH = "3d_scene/config/config_pose.yaml"
# DOPE_CAMERA_INFO_PATH = "3d_scene/config/camera_info.yaml"

# DOPE optimization settings
DOPE_USE_FP16 = (
    False  # Use half-precision for faster GPU inference (can cause detection issues)
)
DOPE_STOP_AT_STAGE = 6  # Stop at stage (1-6). Lower = faster but less accurate.
# Recommended: 6 for accuracy, 4-5 for speed

# Multiple object configurations for DOPE detection (each with its own camera)
DOPE_OBJECTS = {
    "tool": {
        "weights_path": "weights/dope_tool.pth",
        "class_name": "tool",
        "obj_path": "data/scanned_objects/e-screw-driver/eScrewDriver.obj",
        "camera_id": "142122070087",
        "camera_info_path": "3d_scene/config/camera_info_87.yaml",
    },
    "case": {
        "weights_path": "weights/dope_case.pth",
        "class_name": "case",
        "obj_path": "data/scanned_objects/case/case.obj",
        "camera_id": "135122071615",
        "camera_info_path": "3d_scene/config/camera_info_15.yaml",
    },
}

# VGGT configuration - 3D point cloud reconstruction from multi-view cameras
VGGT_WEIGHTS_PATH = "weights/vggt.pt"
VGGT_CONF_THRESHOLD_PCT = 30.0  # Filter out bottom 50% low-confidence points
VGGT_MAX_POINTS = 100000
# Camera IDs used for VGGT inference (order matters - matches the 7 input frames)
VGGT_CAMERA_IDS = [
    "135122071615",
    "137322071489",
    "141722071426",
    "141722073953",
    "141722075184",
    "141722079467",
    "142122070087",
]

# =============================================================================
# Global State
# =============================================================================

calibration_data = {}
yolo_detector = None  # YOLODetector instance
yolo_device = "cpu"
sync_manager = None
streaming_active = False  # Controls whether frame processing is active
battery_tracker = None
lstm_fusion = None  # Single-camera LSTM error detection

# DOPE 6D pose estimation state (multi-object support)
dope_detectors = {}  # object_name -> DOPEDetector instance
current_object_poses = {}  # object_name -> pose dict
pose_lock = threading.Lock()
first_detection_flags = {}  # object_name -> bool (True if first detection captured)

# VGGT 3D reconstruction state
vggt_detector = None  # VGGTDetector instance
vggt_enabled = False  # Whether VGGT inference is enabled (can be toggled from UI)
current_point_cloud = None  # Current point cloud data
point_cloud_lock = threading.Lock()

# Distance from tool tip to case (in cm, sent from frontend)
current_tool_case_distance = 0.0  # Distance in centimeters
current_nearest_screw = None  # Nearest screw position name
distance_lock = threading.Lock()
# Screw sequence tracker instance
screw_tracker = None  # ScrewSequenceTracker instance


# =============================================================================
# Video Manager
# =============================================================================


class SyncedVideoManager:
    """Manages synchronized playback across all cameras (optimized for speed)."""

    def __init__(self):
        self.cameras = {}  # camera_id -> dict with cap, cached_frame, etc.
        self.camera_ids_list = []  # Pre-computed list for iteration
        self.current_frame = 0
        self.total_frames = 0
        self.fps = 30
        self.target_width = 320
        self.target_height = 180
        self.lock = threading.Lock()

        # YOLO state
        self.yolo_detector = None  # YOLODetector instance
        self.yolo_camera_id = None
        self.yolo_inference_interval = 3  # Run YOLO every N frames
        self.yolo_inference_counter = 0
        self.cached_yolo_results = None
        # Multi-camera YOLO for LSTM fusion
        self.lstm_fusion_cameras = []  # List of camera IDs to run YOLO on

        # DOPE 6D pose detection state (multi-object, per-camera support)
        self.dope_detectors = (
            {}
        )  # object_name -> {"detector": DOPEDetector, "camera_id": str}
        self.dope_camera_ids = set()  # All cameras used for DOPE
        self.dope_objects_by_camera = (
            {}
        )  # camera_id -> [(obj_name, detector), ...] - pre-computed
        self.dope_inference_interval = (
            15  # Run DOPE every N frames (reduced from 60 due to optimizations)
        )
        self.dope_inference_counter = 0
        self.cached_dope_results = {}  # object_name -> detection result

        # VGGT 3D reconstruction state
        self.vggt_detector = None  # VGGTDetector instance
        self.vggt_camera_ids = []  # Camera IDs used for VGGT (in order)
        self.vggt_inference_interval = 3  # Run VGGT every N frames (configurable)
        self.vggt_inference_counter = 0
        self.vggt_enabled = False  # Toggle for VGGT inference
        self.cached_vggt_result = None  # Cached point cloud result

        # FPS tracking (simplified)
        self.fps_start_time = time.perf_counter()
        self.fps_frame_count = 0
        self.current_fps = 0.0

        # Pre-allocated constants (avoid repeated allocations)
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.jpeg_params = [cv2.IMWRITE_JPEG_QUALITY, 75]
        self.resize_dims = (self.target_width, self.target_height)

        # Thread pool for parallel camera processing
        self.executor = ThreadPoolExecutor(max_workers=4)

    def add_camera(self, camera_id, video_path):
        """Add a camera to the manager."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file: {video_path}")

        # Optimize video capture for speed
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer delay

        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.cameras[camera_id] = {
            "cap": cap,
            "video_path": video_path,
            "total_frames": total_frames,
            "fps": fps,
            "cached_jpeg": None,
            "frame_buffer": None,  # Pre-allocated frame storage
        }
        self.camera_ids_list.append(camera_id)

        # Use the minimum total frames across all cameras
        if self.total_frames == 0:
            self.total_frames = total_frames
            self.fps = fps
        else:
            self.total_frames = min(self.total_frames, total_frames)

        print(
            f"[Camera {camera_id}] Loaded: {video_path} ({total_frames} frames @ {fps}fps)"
        )

    def set_yolo_detector(self, detector, camera_id):
        """Set YOLO detector for a specific camera.

        Args:
            detector: YOLODetector instance
            camera_id: Camera ID to run YOLO on
        """
        self.yolo_detector = detector
        self.yolo_camera_id = camera_id
        # Warmup the model
        detector.warmup(imgsz=self.target_width)
        print(f"[YOLO] Enabled on camera {camera_id} (device: {detector.device})")

    def set_lstm_fusion_cameras(self, camera_ids):
        """Set which cameras to run YOLO on for LSTM fusion.

        Args:
            camera_ids: List of camera IDs to run YOLO detection on
        """
        self.lstm_fusion_cameras = camera_ids
        print(
            f"[LSTM] YOLO will run on {len(camera_ids)} cameras for fusion: {camera_ids}"
        )

    def set_lstm_fusion(self, lstm_detector):
        """Set LSTM error detector for single-camera monitoring.

        Args:
            lstm_detector: LSTMErrorDetector instance
        """
        self.lstm_fusion = lstm_detector
        print(f"[LSTM] Error detection enabled")

    def set_dope_detector(self, detector, camera_id, object_name="tool"):
        """Set DOPE detector for 6D pose estimation on a specific camera.

        Args:
            detector: DOPEDetector instance
            camera_id: Camera ID to run DOPE on
            object_name: Name of the object being detected
        """
        self.dope_detectors[object_name] = {
            "detector": detector,
            "camera_id": camera_id,
        }
        self.dope_camera_ids.add(camera_id)

        # Pre-compute objects by camera for faster lookup
        if camera_id not in self.dope_objects_by_camera:
            self.dope_objects_by_camera[camera_id] = []
        self.dope_objects_by_camera[camera_id].append((object_name, detector))

        print(f"[DOPE] Enabled '{object_name}' detector on camera {camera_id}")

    def set_vggt_detector(self, detector, camera_ids, inference_interval=6):
        """Set VGGT detector for 3D point cloud reconstruction.

        Args:
            detector: VGGTDetector instance
            camera_ids: List of camera IDs to use for VGGT (in order)
            inference_interval: Run VGGT every N frames
        """
        self.vggt_detector = detector
        self.vggt_camera_ids = camera_ids
        self.vggt_inference_interval = inference_interval
        print(
            f"[VGGT] Enabled on cameras: {camera_ids} (every {inference_interval} frames)"
        )

    def set_vggt_enabled(self, enabled):
        """Enable or disable VGGT inference.

        Args:
            enabled: Boolean to enable/disable VGGT
        """
        self.vggt_enabled = enabled
        print(f"[VGGT] {'Enabled' if enabled else 'Disabled'}")

    def set_vggt_interval(self, interval):
        """Set how often VGGT inference runs.

        Args:
            interval: Run VGGT every N frames
        """
        self.vggt_inference_interval = max(1, interval)
        print(
            f"[VGGT] Inference interval set to every {self.vggt_inference_interval} frames"
        )

    def reset_playback(self):
        """Reset playback to the beginning."""
        global current_object_poses, current_point_cloud, first_detection_flags, battery_tracker
        with self.lock:
            self.current_frame = 0
            self.yolo_inference_counter = 0
            self.cached_yolo_results = None
            self.dope_inference_counter = 0
            self.cached_dope_results = {}
            self.vggt_inference_counter = 0
            self.cached_vggt_result = None
            self.fps_start_time = time.perf_counter()
            self.fps_frame_count = 0

            # Reset all object poses
            with pose_lock:
                current_object_poses = {
                    name: create_empty_pose() for name in self.dope_detectors
                }
                # Reset first detection flags for all objects
                for obj_name in self.dope_detectors:
                    first_detection_flags[obj_name] = False

            # Reset point cloud
            with point_cloud_lock:
                current_point_cloud = None

            # Seek all cameras to frame 0 (only time we seek)
            for cam_data in self.cameras.values():
                cam_data["cap"].set(cv2.CAP_PROP_POS_FRAMES, 0)
                cam_data["cached_jpeg"] = None

            # Reset battery tracker
            if battery_tracker:
                battery_tracker.reset()

            # Reset LSTM fusion
            if hasattr(self, "lstm_fusion") and self.lstm_fusion is not None:
                self.lstm_fusion.reset()

            print("[SyncManager] Playback reset to frame 0")

    def _process_camera(self, camera_id, run_inference, collect_raw_for_vggt=False):
        """Process a single camera frame (for parallel execution).

        Args:
            camera_id: Camera ID to process
            run_inference: Tuple of (run_yolo, run_dope) booleans
            collect_raw_for_vggt: If True, also return raw frame for VGGT

        Returns:
            Tuple of (camera_id, jpeg_bytes, dope_results_dict or None, ended, raw_frame or None)
        """
        run_yolo, run_dope = run_inference
        cam_data = self.cameras[camera_id]
        cap = cam_data["cap"]
        dope_results = {}
        raw_frame = None

        # Read next frame
        ret, frame = cap.read()

        if not ret:
            return (camera_id, None, None, None, True, None)  # Signal video end

        # Store raw frame for VGGT if needed
        if (
            collect_raw_for_vggt
            and self.vggt_camera_ids
            and camera_id in self.vggt_camera_ids
        ):
            raw_frame = frame.copy()

        # Run YOLO on designated camera(s) - either main YOLO camera or fusion cameras
        yolo_results = None
        should_run_yolo = (
            camera_id == self.yolo_camera_id or camera_id in self.lstm_fusion_cameras
        )
        if should_run_yolo and self.yolo_detector is not None and run_yolo:
            yolo_results = self.yolo_detector.detect(frame)

        # Process battery sequence tracking
        battery_status = None
        if camera_id == self.yolo_camera_id and battery_tracker is not None:
            if yolo_results is not None:
                battery_status = battery_tracker.process_yolo_frame(
                    yolo_results, self.current_frame
                )
            else:
                battery_status = battery_tracker._get_state()

        # Run DOPE on this camera if applicable
        if camera_id in self.dope_objects_by_camera and run_dope:
            for obj_name, detector in self.dope_objects_by_camera[camera_id]:
                try:
                    result = detector.detect(frame)
                    if result is not None and result["detected"]:
                        result["object_name"] = obj_name
                        result["camera_id"] = camera_id
                        dope_results[obj_name] = result
                except Exception:
                    pass  # Silently handle errors in hot path

        # Draw DOPE detections
        if camera_id in self.dope_objects_by_camera:
            for obj_name, detector in self.dope_objects_by_camera[camera_id]:
                cached = self.cached_dope_results.get(obj_name)
                if cached is not None:
                    frame = detector.draw_detection(frame, cached)
                    # frame = frame
        # Resize for display (using pre-allocated dimensions)
        frame = cv2.resize(frame, self.resize_dims, interpolation=cv2.INTER_LINEAR)

        # Apply YOLO overlay
        if camera_id == self.yolo_camera_id and self.yolo_detector is not None:
            cached_yolo = (
                yolo_results if yolo_results is not None else self.cached_yolo_results
            )

            # Get LSTM status for display
            lstm_status = None
            if hasattr(self, "lstm_fusion") and self.lstm_fusion is not None:
                lstm_status = self.lstm_fusion.get_status()

                # DEBUG: Print LSTM status periodically
                if self.current_frame % 30 == 0:  # Every 30 frames
                    print(
                        f"[LSTM Debug] Frame {self.current_frame}: "
                        f"ready={lstm_status.get('ready')}, "
                        f"error={lstm_status.get('error_detected')}, "
                        f"conf={lstm_status.get('confidence', 0):.2f}, "
                        f"buffer={lstm_status.get('buffer_size')}"
                    )

            frame = self.yolo_detector.draw_predictions(
                frame,
                cached_yolo,
                target_width=self.target_width,
                target_height=self.target_height,
                battery_status=battery_status,
                lstm_status=lstm_status,  # NEW: Add LSTM overlay
            )
            cv2.putText(
                frame,
                f"FPS: {self.current_fps:.1f}",
                (5, 15),
                self.font,
                0.45,
                (0, 255, 0),
                1,
            )
            if yolo_results is not None:
                self.cached_yolo_results = yolo_results

        # Draw DOPE status
        if camera_id in self.dope_objects_by_camera:
            objs = self.dope_objects_by_camera[camera_id]
            detected = sum(
                1
                for name, _ in objs
                if name in self.cached_dope_results
                and self.cached_dope_results[name].get("detected")
            )
            color = (0, 255, 0) if detected > 0 else (0, 165, 255)
            cv2.putText(
                frame,
                f"DOPE: {detected}/{len(objs)}",
                (5, 15),
                self.font,
                0.4,
                color,
                1,
            )

        # Encode as JPEG using pre-allocated params
        _, jpeg = cv2.imencode(".jpg", frame, self.jpeg_params)

        return (camera_id, jpeg.tobytes(), yolo_results, dope_results, False, raw_frame)

    def advance_frame(self):
        """Advance to the next frame and update all cameras (optimized parallel processing)."""
        global current_object_poses, current_point_cloud

        # Calculate FPS outside the heavy processing
        self.fps_frame_count += 1
        now = time.perf_counter()
        elapsed = now - self.fps_start_time
        if elapsed >= 1.0:  # Update FPS every second
            self.current_fps = self.fps_frame_count / elapsed
            self.fps_frame_count = 0
            self.fps_start_time = now

        # Determine which inferences to run this frame
        run_yolo = (self.yolo_inference_counter % self.yolo_inference_interval) == 0
        run_dope = (self.dope_inference_counter % self.dope_inference_interval) == 0

        # Debug VGGT conditions
        vggt_interval_ok = (
            self.vggt_inference_counter % self.vggt_inference_interval
        ) == 0
        run_vggt = (
            self.vggt_enabled and self.vggt_detector is not None and vggt_interval_ok
        )

        # Log VGGT state periodically
        if self.current_frame % 100 == 0:
            print(
                f"[VGGT Debug] enabled={self.vggt_enabled}, detector={self.vggt_detector is not None}, "
                f"interval_ok={vggt_interval_ok}, camera_ids={self.vggt_camera_ids}"
            )

        self.current_frame += 1
        self.yolo_inference_counter += 1
        self.dope_inference_counter += 1
        self.vggt_inference_counter += 1

        # Process cameras in parallel for heavy inference frames, sequential otherwise
        video_ended = False
        vggt_raw_frames = {}  # camera_id -> raw frame for VGGT

        if run_dope and len(self.dope_objects_by_camera) > 1:
            # Parallel processing when running DOPE on multiple cameras
            futures = []
            for camera_id in self.camera_ids_list:
                future = self.executor.submit(
                    self._process_camera, camera_id, (run_yolo, run_dope), run_vggt
                )
                futures.append(future)

            # Collect results
            yolo_result = None
            yolo_results_for_fusion = {}  # Collect all YOLO results for fusion
            for future in futures:
                camera_id, jpeg_data, yolo_results, dope_results, ended, raw_frame = (
                    future.result()
                )

                # Save YOLO results from ALL fusion cameras (not just primary)
                if camera_id in self.lstm_fusion_cameras and yolo_results is not None:
                    yolo_results_for_fusion[camera_id] = yolo_results

                # Also save primary YOLO camera result (for backward compatibility)
                if camera_id == self.yolo_camera_id and yolo_results is not None:
                    yolo_result = yolo_results

                if ended:
                    video_ended = True
                else:
                    with self.lock:
                        self.cameras[camera_id]["cached_jpeg"] = jpeg_data
                    if dope_results:
                        with pose_lock:
                            for obj_name, result in dope_results.items():
                                result["fresh"] = True
                                # Only update case on first detection, always update other objects
                                if obj_name == "case":
                                    if not first_detection_flags.get(obj_name, False):
                                        current_object_poses[obj_name] = result
                                        first_detection_flags[obj_name] = True
                                        print(
                                            f"[DOPE] First detection of '{obj_name}' captured and locked"
                                        )
                                else:
                                    current_object_poses[obj_name] = result
                                self.cached_dope_results[obj_name] = result
                    if raw_frame is not None:
                        vggt_raw_frames[camera_id] = raw_frame

            # NEW: Now run LSTM fusion with all collected results
            if hasattr(self, "lstm_fusion") and self.lstm_fusion is not None:
                frame_shape = (self.target_height, self.target_width)

                from lstm_inference import MultiCameraLSTMFusion

                if isinstance(self.lstm_fusion, MultiCameraLSTMFusion):
                    # Multi-camera: use all collected YOLO results
                    if len(yolo_results_for_fusion) > 0:
                        lstm_result = self.lstm_fusion.detect(
                            yolo_results_for_fusion, frame_shape
                        )
                        if lstm_result.get("error_detected", False) and lstm_result.get(
                            "ready", False
                        ):
                            print(
                                f"[LSTM Fusion] ⚠️ ERROR DETECTED! Confidence: {lstm_result['confidence']:.2%}, "
                                f"Cameras in error: {lstm_result.get('cameras_in_error')}/{lstm_result.get('cameras_total')}"
                            )

            # Run LSTM detection - collect YOLO results from all cameras
            if hasattr(self, "lstm_fusion") and self.lstm_fusion is not None:
                frame_shape = (self.target_height, self.target_width)

                # Check if multi-camera or single-camera
                from lstm_inference import MultiCameraLSTMFusion

                if isinstance(self.lstm_fusion, MultiCameraLSTMFusion):
                    # Multi-camera: collect YOLO results from ALL fusion cameras
                    # We need to collect results that were saved during parallel processing
                    # This happens in the next section after we collect all futures
                    pass  # Will be handled after collecting all results
                else:
                    # Single camera
                    if yolo_result is not None:
                        lstm_result = self.lstm_fusion.detect(yolo_result, frame_shape)
                        if lstm_result.get("error_detected", False) and lstm_result.get(
                            "ready", False
                        ):
                            print(
                                f"[LSTM] ⚠️ ERROR DETECTED! Confidence: {lstm_result['confidence']:.2%}"
                            )
        else:
            # Sequential processing for lighter frames
            yolo_result = None
            yolo_results_for_fusion = {}
            for camera_id in self.camera_ids_list:
                camera_id, jpeg_data, yolo_results, dope_results, ended, raw_frame = (
                    self._process_camera(camera_id, (run_yolo, run_dope), run_vggt)
                )
                # Save YOLO results from ALL fusion cameras (not just primary)
                if camera_id in self.lstm_fusion_cameras and yolo_results is not None:
                    yolo_results_for_fusion[camera_id] = yolo_results

                # Also save primary YOLO camera result (for backward compatibility)
                if camera_id == self.yolo_camera_id and yolo_results is not None:
                    yolo_result = yolo_results

                if ended:
                    video_ended = True
                else:
                    self.cameras[camera_id]["cached_jpeg"] = jpeg_data
                    if dope_results:
                        with pose_lock:
                            for obj_name, result in dope_results.items():
                                result["fresh"] = True
                                # Only update case on first detection, always update other objects
                                if obj_name == "case":
                                    if not first_detection_flags.get(obj_name, False):
                                        current_object_poses[obj_name] = result
                                        first_detection_flags[obj_name] = True
                                        print(
                                            f"[DOPE] First detection of '{obj_name}' captured and locked"
                                        )
                                else:
                                    current_object_poses[obj_name] = result
                                self.cached_dope_results[obj_name] = result
                    if raw_frame is not None:
                        vggt_raw_frames[camera_id] = raw_frame

            # Run LSTM detection - collect YOLO results from all cameras
            if hasattr(self, "lstm_fusion") and self.lstm_fusion is not None:
                frame_shape = (self.target_height, self.target_width)

                # Check if multi-camera or single-camera
                from lstm_inference import MultiCameraLSTMFusion

                if isinstance(self.lstm_fusion, MultiCameraLSTMFusion):
                    # Multi-camera: use all collected YOLO results
                    if len(yolo_results_for_fusion) > 0:
                        lstm_result = self.lstm_fusion.detect(
                            yolo_results_for_fusion, frame_shape
                        )
                        if lstm_result.get("error_detected", False) and lstm_result.get(
                            "ready", False
                        ):
                            print(
                                f"[LSTM Fusion] ⚠️ ERROR DETECTED! Confidence: {lstm_result['confidence']:.2%}, "
                                f"Cameras in error: {lstm_result.get('cameras_in_error')}/{lstm_result.get('cameras_total')}"
                            )
                else:
                    # Single camera
                    if yolo_result is not None:
                        lstm_result = self.lstm_fusion.detect(yolo_result, frame_shape)
                        if lstm_result.get("error_detected", False) and lstm_result.get(
                            "ready", False
                        ):
                            print(
                                f"[LSTM] ⚠️ ERROR DETECTED! Confidence: {lstm_result['confidence']:.2%}"
                            )

        # Run VGGT inference if enabled and we have all required frames
        if run_vggt:
            print(
                f"[VGGT] run_vggt=True, collected_frames={len(vggt_raw_frames)}, required={len(self.vggt_camera_ids)}, "
                f"collected_ids={list(vggt_raw_frames.keys())}, required_ids={self.vggt_camera_ids}"
            )

        if run_vggt and len(vggt_raw_frames) == len(self.vggt_camera_ids):
            # Collect frames in the correct order
            vggt_frames = [vggt_raw_frames[cam_id] for cam_id in self.vggt_camera_ids]
            print(f"[VGGT] Triggering inference with {len(vggt_frames)} frames")

            # Run VGGT inference in background thread to not block frame processing
            def run_vggt_async():
                global current_point_cloud
                result = self.vggt_detector.run_inference(vggt_frames)
                with point_cloud_lock:
                    current_point_cloud = result
                    self.cached_vggt_result = result
                print(
                    f"[VGGT] Inference complete: success={result.get('success')}, points={result.get('num_points')}"
                )

            self.executor.submit(run_vggt_async)

        # Handle video loop
        if video_ended:
            self._reset_all_captures()

    def _reset_all_captures(self):
        """Reset all video captures to frame 0."""
        global current_object_poses, current_point_cloud, first_detection_flags
        self.current_frame = 0
        self.yolo_inference_counter = 0
        self.cached_yolo_results = None
        self.dope_inference_counter = 0
        self.cached_dope_results = {}
        self.vggt_inference_counter = 0
        self.cached_vggt_result = None
        self.fps_start_time = time.perf_counter()
        self.fps_frame_count = 0

        # Reset LSTM fusion
        if hasattr(self, "lstm_fusion") and self.lstm_fusion is not None:
            self.lstm_fusion.reset()

        for cam_data in self.cameras.values():
            cam_data["cap"].set(cv2.CAP_PROP_POS_FRAMES, 0)

        with pose_lock:
            current_object_poses = {
                name: create_empty_pose() for name in self.dope_detectors
            }
            # Reset first detection flags for all objects
            for obj_name in self.dope_detectors:
                first_detection_flags[obj_name] = False

        with point_cloud_lock:
            current_point_cloud = None

    def get_frame(self, camera_id):
        """Get cached frame for a camera."""
        with self.lock:
            if camera_id in self.cameras:
                return self.cameras[camera_id]["cached_jpeg"]
            return None

    def get_camera_ids(self):
        """Get list of camera IDs."""
        return list(self.cameras.keys())

    def release(self):
        """Release all video captures and shutdown executor."""
        self.executor.shutdown(wait=False)
        for cam_data in self.cameras.values():
            if cam_data["cap"]:
                cam_data["cap"].release()


async def frame_update_loop():
    """Background task that advances frames at a fixed rate when streaming is active."""
    global sync_manager, streaming_active

    frame_interval = 1.0 / 30  # Target 30 FPS (optimized sequential reading)
    loop = asyncio.get_running_loop()

    while True:
        if streaming_active:
            start_time = time.perf_counter()

            # Advance frame in thread pool (blocking operations)
            await loop.run_in_executor(None, sync_manager.advance_frame)

            # Maintain frame rate
            elapsed = time.perf_counter() - start_time
            if elapsed < frame_interval:
                await asyncio.sleep(frame_interval - elapsed)
        else:
            # When not streaming, just sleep to avoid busy waiting
            await asyncio.sleep(0.1)


# =============================================================================
# Model Loading
# =============================================================================


def init_yolo():
    """Initialize YOLO detector."""
    global yolo_detector, yolo_device, battery_tracker

    yolo_detector, yolo_device = load_yolo_model()

    battery_tracker = BatterySequenceTracker()
    print("[BatteryFSM] Tracker initialized")

    return yolo_detector


def init_lstm_fusion():
    """Initialize multi-camera LSTM error detection with fusion."""
    global lstm_fusion, yolo_device

    try:
        from lstm_inference import load_multi_camera_lstm

        # Use multiple cameras for better detection
        # You can adjust which cameras to use - here using 3 cameras as example
        fusion_cameras = [
            "137322071489",  # Your YOLO camera
            "135122071615",  # Another camera
            "141722071426",  # Another camera
        ]

        # Load multi-camera LSTM with K-out-of-N voting
        # K=2 means at least 2 cameras must agree to flag an error
        lstm_fusion = load_multi_camera_lstm(
            camera_ids=fusion_cameras, device=yolo_device
        )

        if lstm_fusion:
            print(
                f"[LSTM] Multi-camera error detection enabled ({len(fusion_cameras)} cameras)"
            )
            lstm_fusion.set_voting_threshold(1)
        return lstm_fusion
    except Exception as e:
        print(f"[LSTM] Failed to initialize: {e}")
        return None


def load_dope_models():
    """Load DOPE models for 6D pose estimation (multi-object support).

    Uses optimized inference with:
    - FP16 half-precision (configurable via DOPE_USE_FP16)
    - Configurable stage count (DOPE_STOP_AT_STAGE)
    """
    global dope_detectors, current_object_poses, first_detection_flags

    dope_detectors = {}
    current_object_poses = {}
    first_detection_flags = {}

    # Check if DOPE is enabled
    if not DOPE_ENABLED:
        print("[DOPE] Disabled via DOPE_ENABLED=False, skipping model loading")
        return dope_detectors

    print(
        f"[DOPE] Optimization settings: FP16={DOPE_USE_FP16}, stop_at_stage={DOPE_STOP_AT_STAGE}"
    )

    for obj_name, obj_config in DOPE_OBJECTS.items():
        weights_path = obj_config["weights_path"]
        class_name = obj_config["class_name"]
        camera_info_path = obj_config["camera_info_path"]

        detector = load_dope_detector(
            weights_path=weights_path,
            config_path=DOPE_CONFIG_PATH,
            camera_info_path=camera_info_path,
            class_name=class_name,
            use_fp16=DOPE_USE_FP16,
            stop_at_stage=DOPE_STOP_AT_STAGE,
        )

        if detector is not None:
            dope_detectors[obj_name] = detector
            current_object_poses[obj_name] = create_empty_pose()
            first_detection_flags[obj_name] = False  # Track first detection
            print(f"[DOPE] Loaded detector for '{obj_name}'")
        else:
            print(
                f"[DOPE] Warning: Could not load detector for '{obj_name}' (weights: {weights_path})"
            )

    return dope_detectors


def load_vggt_model():
    """Load VGGT model for 3D point cloud reconstruction."""
    global vggt_detector

    vggt_detector = load_vggt_detector(
        weights_path=VGGT_WEIGHTS_PATH,
        conf_threshold_pct=VGGT_CONF_THRESHOLD_PCT,
        max_points=VGGT_MAX_POINTS,
    )

    if vggt_detector is not None:
        print(f"[VGGT] Model loaded successfully")
    else:
        print(
            f"[VGGT] Warning: Could not load VGGT model (weights: {VGGT_WEIGHTS_PATH})"
        )

    return vggt_detector


def load_calibration():
    """Load camera calibration data from YAML file and transform to checkerboard origin."""
    global calibration_data

    with open(CALIBRATION_FILE, "r") as f:
        raw_calibration = yaml.safe_load(f)

    # Find the master camera with checkerboard transform
    checkerboard_to_master = None
    for cam_id, cam_data in raw_calibration.items():
        if cam_data.get("master", False) and "checkerboard" in cam_data:
            checkerboard_to_master = np.array(
                cam_data["checkerboard"], dtype=np.float64
            )
            print(
                f"[Calibration] Found checkerboard transform in master camera {cam_id}"
            )
            break

    if checkerboard_to_master is None:
        print(
            "[Calibration] WARNING: No checkerboard transform found, using master camera as origin"
        )
        calibration_data = raw_calibration
        return calibration_data

    # Compute inverse: master_to_checkerboard (this transforms points from master to checkerboard frame)
    master_to_checkerboard = np.linalg.inv(checkerboard_to_master)
    print(f"[Calibration] Transforming all cameras to checkerboard origin")

    # Transform all camera extrinsics to be relative to checkerboard
    calibration_data = {}
    for cam_id, cam_data in raw_calibration.items():
        cam_extrinsics = np.array(cam_data["extrinsics"], dtype=np.float64)

        # New extrinsics: camera_to_checkerboard = master_to_checkerboard @ camera_to_master
        # Since extrinsics are camera_to_master (or camera_to_world where world=master)
        # We need: camera_to_checkerboard = master_to_checkerboard @ camera_to_master
        new_extrinsics = master_to_checkerboard @ cam_extrinsics

        # Copy camera data with transformed extrinsics
        calibration_data[cam_id] = {
            "extrinsics": new_extrinsics.tolist(),
            "intrinsics": cam_data["intrinsics"],
            "number": cam_data.get("number", 0),
            "master": cam_data.get("master", False),
        }

        # Keep checkerboard info if present
        if "checkerboard" in cam_data:
            calibration_data[cam_id]["checkerboard"] = cam_data["checkerboard"]

    print(f"[Calibration] Loaded {len(calibration_data)} cameras (checkerboard origin)")
    return calibration_data


def init_sync_manager():
    """Initialize synchronized video manager."""
    global sync_manager, yolo_detector, lstm_fusion, dope_detectors, vggt_detector

    sync_manager = SyncedVideoManager()

    # Get all camera directories
    camera_dirs = [
        d
        for d in os.listdir(RECORDING_DIR)
        if os.path.isdir(os.path.join(RECORDING_DIR, d)) and d.isdigit()
    ]

    for cam_id in sorted(camera_dirs):
        cam_dir = os.path.join(RECORDING_DIR, cam_id)
        video_file = os.path.join(cam_dir, f"{cam_id}.mp4")

        if os.path.exists(video_file):
            try:
                sync_manager.add_camera(cam_id, video_file)
            except Exception as e:
                print(f"[Camera {cam_id}] Failed to load: {e}")

    # Set YOLO detector on designated camera
    if yolo_detector is not None:
        sync_manager.set_yolo_detector(yolo_detector, YOLO_CAMERA_ID)

    # Configure LSTM fusion cameras
    if lstm_fusion is not None:
        from lstm_inference import MultiCameraLSTMFusion

        if isinstance(lstm_fusion, MultiCameraLSTMFusion):
            # Multi-camera: set which cameras to run YOLO on
            sync_manager.set_lstm_fusion_cameras(LSTM_FUSION_CAMERAS)

    # Set LSTM fusion for error detection
    if lstm_fusion is not None:
        sync_manager.set_lstm_fusion(lstm_fusion)

    # Set DOPE detectors with per-object camera assignments
    for obj_name, detector in dope_detectors.items():
        camera_id = DOPE_OBJECTS[obj_name].get("camera_id")
        sync_manager.set_dope_detector(detector, camera_id, obj_name)

    # Set VGGT detector with camera IDs
    if vggt_detector is not None:
        # Filter to only cameras that exist in our video manager
        valid_vggt_cameras = [
            cam_id for cam_id in VGGT_CAMERA_IDS if cam_id in sync_manager.cameras
        ]
        if len(valid_vggt_cameras) == len(VGGT_CAMERA_IDS):
            sync_manager.set_vggt_detector(
                vggt_detector, valid_vggt_cameras, inference_interval=20
            )
        else:
            print(
                f"[VGGT] Warning: Not all VGGT cameras available. Need: {VGGT_CAMERA_IDS}, Have: {list(sync_manager.cameras.keys())}"
            )

    print(
        f"[SyncManager] Initialized with {len(sync_manager.cameras)} cameras, synced to {sync_manager.total_frames} frames"
    )


# =============================================================================
# API Helpers
# =============================================================================


def fast_json_dumps(obj):
    """Fast JSON serialization using orjson if available."""
    if USE_ORJSON:
        return orjson.dumps(obj).decode("utf-8")
    return json.dumps(obj)


def to_serializable(obj):
    """Convert numpy arrays and nested structures to JSON-serializable format."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (list, tuple)):
        return [to_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    return obj


# Pre-computed response headers
NO_CACHE_HEADERS = {
    "Cache-Control": "no-cache, no-store, must-revalidate",
    "Pragma": "no-cache",
    "Expires": "0",
}


# =============================================================================
# API Routes
# =============================================================================


async def index(request):
    """Serve HTML page."""
    html_path = os.path.join(os.path.dirname(__file__), "web_interface.html")
    with open(html_path, "r") as f:
        content = f.read()
    return web.Response(content_type="text/html", text=content)


# Cache for calibration JSON (computed once at startup)
_calibration_json_cache = None


async def get_calibration(request):
    """Return camera calibration data as JSON (cached)."""
    global _calibration_json_cache

    if _calibration_json_cache is None:
        result = {}
        for cam_id, cam_data in calibration_data.items():
            result[cam_id] = {
                "extrinsics": cam_data.get("extrinsics", []),
                "intrinsics": cam_data.get("intrinsics", {}),
                "number": cam_data.get("number", 0),
                "master": cam_data.get("master", False),
            }
        _calibration_json_cache = fast_json_dumps(result)

    return web.Response(content_type="application/json", text=_calibration_json_cache)


async def get_cameras(request):
    """Return list of available cameras."""
    cameras = sync_manager.get_camera_ids()
    return web.Response(content_type="application/json", text=fast_json_dumps(cameras))


async def get_frame(request):
    """Get current frame from a specific camera as JPEG."""
    camera_id = request.match_info.get("camera_id")

    jpeg_data = sync_manager.get_frame(camera_id)

    if jpeg_data is None:
        return web.Response(
            status=404, text=f"Camera {camera_id} not found or no frame"
        )

    return web.Response(
        body=jpeg_data,
        content_type="image/jpeg",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


async def start_streaming(request):
    """Start the video streaming/processing."""
    global streaming_active
    streaming_active = True
    print("[Server] Streaming STARTED")
    return web.Response(
        content_type="application/json",
        text='{"status":"started","streaming":true}',  # Pre-built static response
    )


async def stop_streaming(request):
    """Stop the video streaming/processing."""
    global streaming_active
    streaming_active = False
    print("[Server] Streaming STOPPED")
    return web.Response(
        content_type="application/json",
        text='{"status":"stopped","streaming":false}',  # Pre-built static response
    )


async def reset_streaming(request):
    """Reset playback to the beginning."""
    global sync_manager
    sync_manager.reset_playback()
    return web.Response(
        content_type="application/json",
        text='{"status":"reset","frame":0}',  # Pre-built static response
    )


async def get_streaming_status(request):
    """Get current streaming status."""
    global streaming_active
    return web.Response(
        content_type="application/json",
        text='{"streaming":true}' if streaming_active else '{"streaming":false}',
    )


async def get_tool_pose(request):
    """Get current 6D pose of the detected tool object (legacy endpoint).

    Returns position (x, y, z) in meters and quaternion (x, y, z, w).
    The pose is in camera coordinate system.
    """
    global current_object_poses
    with pose_lock:
        pose_data = current_object_poses.get("tool", create_empty_pose()).copy()

    return web.Response(
        content_type="application/json",
        text=fast_json_dumps(to_serializable(pose_data)),
        headers=NO_CACHE_HEADERS,
    )


async def get_object_poses(request):
    """Get current 6D poses of all detected objects.

    Returns a dictionary of object_name -> pose data.
    Each pose contains position (x, y, z) in meters and quaternion (x, y, z, w).
    The poses are in camera coordinate system.
    """
    global current_object_poses
    with pose_lock:
        poses_data = {name: pose.copy() for name, pose in current_object_poses.items()}

    return web.Response(
        content_type="application/json",
        text=fast_json_dumps(to_serializable(poses_data)),
        headers=NO_CACHE_HEADERS,
    )


# Cache for DOPE objects config (computed once)
_dope_objects_json_cache = None


async def get_dope_objects(request):
    """Get list of configured DOPE objects and their OBJ file paths (cached)."""
    global _dope_objects_json_cache

    if _dope_objects_json_cache is None:
        objects_info = {}
        for obj_name, obj_config in DOPE_OBJECTS.items():
            objects_info[obj_name] = {
                "obj_path": obj_config["obj_path"],
                "class_name": obj_config["class_name"],
                "camera_id": obj_config["camera_id"],
                "loaded": obj_name in dope_detectors,
            }
        _dope_objects_json_cache = fast_json_dumps(objects_info)

    return web.Response(content_type="application/json", text=_dope_objects_json_cache)


# =============================================================================
# VGGT Point Cloud API Routes
# =============================================================================

# Cache for VGGT binary data to avoid recomputation
_vggt_binary_cache = {"data": None, "num_points": 0, "version": 0}
_vggt_version_counter = 0


async def get_vggt_point_cloud(request):
    """Get current VGGT point cloud data (JSON format - legacy).

    Note: Use /api/vggt/pointcloud/binary for better performance.
    """
    global current_point_cloud

    with point_cloud_lock:
        if current_point_cloud is None or not current_point_cloud.get("success", False):
            return web.Response(
                content_type="application/json",
                text='{"success":false,"num_points":0}',
                headers=NO_CACHE_HEADERS,
            )

        # Convert numpy arrays to lists for JSON serialization
        result = {
            "success": True,
            "num_points": current_point_cloud["num_points"],
            "points": current_point_cloud["points"].tolist(),
            "colors": current_point_cloud["colors"].tolist(),
            "inference_time": current_point_cloud.get("inference_time", 0),
        }

    return web.Response(
        content_type="application/json",
        text=fast_json_dumps(result),
        headers=NO_CACHE_HEADERS,
    )


async def get_vggt_point_cloud_binary(request):
    """Get VGGT point cloud as binary data for ultra-fast transmission.

    Binary format:
    - Header (12 bytes): num_points (uint32), version (uint32), inference_time_ms (uint32)
    - Points (num_points * 12 bytes): float32 x, y, z for each point
    - Colors (num_points * 3 bytes): uint8 r, g, b for each point

    Total size for 50K points: ~750KB vs ~4MB for JSON
    """
    global current_point_cloud, _vggt_binary_cache, _vggt_version_counter

    with point_cloud_lock:
        if current_point_cloud is None or not current_point_cloud.get("success", False):
            # Return minimal header indicating no data
            header = np.array([0, 0, 0], dtype=np.uint32).tobytes()
            return web.Response(
                body=header,
                content_type="application/octet-stream",
                headers=NO_CACHE_HEADERS,
            )

        num_points = current_point_cloud["num_points"]
        inference_time = current_point_cloud.get("inference_time", 0)

        # Always rebuild binary data (point cloud content may change even with same count)
        _vggt_version_counter += 1

        # Build binary data - ensure arrays are contiguous and correctly shaped
        points = current_point_cloud["points"]
        colors = current_point_cloud["colors"]

        # Ensure correct shape (N, 3) and make contiguous
        if points.ndim == 1:
            points = points.reshape(-1, 3)
        if colors.ndim == 1:
            colors = colors.reshape(-1, 3)

        # Flatten to 1D for binary transfer: [x0,y0,z0,x1,y1,z1,...]
        points_flat = np.ascontiguousarray(points.flatten(), dtype=np.float32)
        colors_flat = np.ascontiguousarray(colors.flatten(), dtype=np.uint8)

        # Header: num_points, version, inference_time_ms
        inference_time_ms = int(inference_time * 1000)
        header = np.array(
            [num_points, _vggt_version_counter, inference_time_ms], dtype=np.uint32
        )

        # Combine into single bytes object
        binary_data = header.tobytes() + points_flat.tobytes() + colors_flat.tobytes()

        # Debug logging (remove in production)
        print(
            f"[VGGT Binary] Sending {num_points} points, {len(binary_data)} bytes, v{_vggt_version_counter}"
        )

    return web.Response(
        body=binary_data,
        content_type="application/octet-stream",
        headers=NO_CACHE_HEADERS,
    )


async def get_vggt_point_cloud_version(request):
    """Get just the version/count of current point cloud (for change detection).

    Returns minimal JSON to check if point cloud has changed.
    Always returns current state - version increments on each new point cloud.
    """
    global current_point_cloud

    with point_cloud_lock:
        if current_point_cloud is None or not current_point_cloud.get("success", False):
            return web.Response(
                content_type="application/json",
                text='{"v":0,"n":0}',
                headers=NO_CACHE_HEADERS,
            )

        # Use inference time as a simple change indicator
        # (each inference produces new data)
        version = int(current_point_cloud.get("inference_time", 0) * 10000)

        return web.Response(
            content_type="application/json",
            text=f'{{"v":{version},"n":{current_point_cloud["num_points"]}}}',
            headers=NO_CACHE_HEADERS,
        )


async def get_vggt_status(request):
    """Get VGGT status including enabled state and configuration."""
    global sync_manager, vggt_detector

    status = {
        "loaded": vggt_detector is not None,
        "enabled": sync_manager.vggt_enabled if sync_manager else False,
        "inference_interval": (
            sync_manager.vggt_inference_interval if sync_manager else 20
        ),
        "camera_ids": VGGT_CAMERA_IDS,
        "conf_threshold_pct": VGGT_CONF_THRESHOLD_PCT,
        "max_points": VGGT_MAX_POINTS,
    }

    return web.Response(
        content_type="application/json",
        text=fast_json_dumps(status),
        headers=NO_CACHE_HEADERS,
    )


async def set_vggt_enabled(request):
    """Enable or disable VGGT inference."""
    global sync_manager, vggt_enabled

    try:
        data = await request.json()
        enabled = data.get("enabled", False)

        if sync_manager:
            sync_manager.set_vggt_enabled(enabled)
            vggt_enabled = enabled

        return web.Response(
            content_type="application/json",
            text=fast_json_dumps({"success": True, "enabled": enabled}),
        )
    except Exception as e:
        return web.Response(
            content_type="application/json",
            text=fast_json_dumps({"success": False, "error": str(e)}),
            status=400,
        )


async def set_dope_interval(request):
    """Set DOPE inference interval (how often to run). Higher = faster playback."""
    global sync_manager

    try:
        data = await request.json()
        interval = int(data.get("interval", 6))
        interval = max(1, min(60, interval))  # Clamp to 1-60

        if sync_manager:
            sync_manager.dope_inference_interval = interval
            print(f"[DOPE] Inference interval set to every {interval} frames")

        return web.Response(
            content_type="application/json",
            text=fast_json_dumps({"success": True, "interval": interval}),
        )
    except Exception as e:
        return web.Response(
            content_type="application/json",
            text=fast_json_dumps({"success": False, "error": str(e)}),
            status=400,
        )


async def set_vggt_interval(request):
    """Set VGGT inference interval (how often to run)."""
    global sync_manager

    try:
        data = await request.json()
        interval = int(data.get("interval", 20))

        if sync_manager:
            sync_manager.set_vggt_interval(interval)

        return web.Response(
            content_type="application/json",
            text=fast_json_dumps({"success": True, "interval": interval}),
        )
    except Exception as e:
        return web.Response(
            content_type="application/json",
            text=fast_json_dumps({"success": False, "error": str(e)}),
            status=400,
        )


# =============================================================================
# Tool-Case Distance API Routes
# =============================================================================


async def set_tool_case_distance(request):
    """Set the current distance from tool tip to case (sent from frontend).

    Expects JSON: {"distance_cm": float, "nearest_screw": str (optional)}
    Also updates the screw sequence tracker if nearest_screw is provided.
    """
    global current_tool_case_distance, current_nearest_screw, screw_tracker

    try:
        data = await request.json()
        distance_cm = float(data.get("distance_cm", 0.0))
        nearest_screw = data.get("nearest_screw", None)

        with distance_lock:
            current_tool_case_distance = distance_cm
            current_nearest_screw = nearest_screw

        # Update screw sequence tracker
        if screw_tracker is not None:
            screw_tracker.update(distance_cm=distance_cm, nearest_screw=nearest_screw)

        # Only log when close (avoid spam)
        if distance_cm < 10.0:
            print(f"[Distance] Tool to {nearest_screw or 'case'}: {distance_cm:.2f} cm")

        return web.Response(
            content_type="application/json",
            text=fast_json_dumps(
                {
                    "success": True,
                    "distance_cm": distance_cm,
                    "nearest_screw": nearest_screw,
                }
            ),
        )
    except Exception as e:
        return web.Response(
            content_type="application/json",
            text=fast_json_dumps({"success": False, "error": str(e)}),
            status=400,
        )


async def get_tool_case_distance(request):
    """Get the current distance from tool tip to case in centimeters.

    Returns JSON: {"distance_cm": float}
    """
    global current_tool_case_distance

    with distance_lock:
        distance_cm = current_tool_case_distance

    return web.Response(
        content_type="application/json",
        text=fast_json_dumps({"distance_cm": distance_cm}),
        headers=NO_CACHE_HEADERS,
    )


async def get_tool_case_distance(request):
    """Get the current distance from tool tip to case in centimeters.

    Returns JSON: {"distance_cm": float}
    """
    global current_tool_case_distance

    with distance_lock:
        distance_cm = current_tool_case_distance

    return web.Response(
        content_type="application/json",
        text=fast_json_dumps({"distance_cm": distance_cm}),
        headers=NO_CACHE_HEADERS,
    )


# =============================================================================
# Screw Sequence Tracking API Routes
# =============================================================================


async def get_screw_status(request):
    """Get current screw sequence tracking status.

    Returns comprehensive status including:
    - Current state (idle, approaching, screwing, completed)
    - Active screw being worked on
    - Expected vs actual sequence
    - Progress and errors
    """
    global screw_tracker

    if screw_tracker is None:
        return web.Response(
            content_type="application/json",
            text=fast_json_dumps({"error": "Tracker not initialized"}),
            status=503,
        )

    status = screw_tracker.get_status()

    return web.Response(
        content_type="application/json",
        text=fast_json_dumps(status),
        headers=NO_CACHE_HEADERS,
    )


async def reset_screw_sequence(request):
    """Reset the screw sequence tracker to initial state."""
    global screw_tracker

    if screw_tracker is None:
        return web.Response(
            content_type="application/json",
            text=fast_json_dumps({"error": "Tracker not initialized"}),
            status=503,
        )

    screw_tracker.reset()

    return web.Response(
        content_type="application/json",
        text=fast_json_dumps({"success": True, "message": "Sequence reset"}),
    )


async def set_screw_tracking_enabled(request):
    """Enable or disable screw sequence tracking."""
    global screw_tracker

    if screw_tracker is None:
        return web.Response(
            content_type="application/json",
            text=fast_json_dumps({"error": "Tracker not initialized"}),
            status=503,
        )

    try:
        data = await request.json()
        enabled = data.get("enabled", True)

        screw_tracker.set_enabled(enabled)

        return web.Response(
            content_type="application/json",
            text=fast_json_dumps({"success": True, "enabled": enabled}),
        )
    except Exception as e:
        return web.Response(
            content_type="application/json",
            text=fast_json_dumps({"success": False, "error": str(e)}),
            status=400,
        )


async def set_screw_tracking_mode(request):
    """Set screw tracking mode: '3d' (frame-based) or 'time' (duration-based)."""
    global screw_tracker

    if screw_tracker is None:
        return web.Response(
            content_type="application/json",
            text=fast_json_dumps({"error": "Tracker not initialized"}),
            status=503,
        )

    try:
        data = await request.json()
        mode = data.get("mode", "3d")

        if mode not in ["3d", "time"]:
            return web.Response(
                content_type="application/json",
                text=fast_json_dumps(
                    {"success": False, "error": 'Invalid mode. Use "3d" or "time"'}
                ),
                status=400,
            )

        screw_tracker.set_mode(mode)

        return web.Response(
            content_type="application/json",
            text=fast_json_dumps({"success": True, "mode": mode}),
        )
    except Exception as e:
        return web.Response(
            content_type="application/json",
            text=fast_json_dumps({"success": False, "error": str(e)}),
            status=400,
        )


async def on_shutdown(app):
    """Cleanup on shutdown."""
    global streaming_active
    streaming_active = False
    if sync_manager:
        sync_manager.release()


# =============================================================================
# Server
# =============================================================================


async def run_server(host="0.0.0.0", port=8085):
    """Run server."""
    app = web.Application()

    # Routes
    app.router.add_get("/", index)
    app.router.add_get("/api/calibration", get_calibration)
    app.router.add_get("/api/cameras", get_cameras)
    app.router.add_get("/api/frame/{camera_id}", get_frame)
    app.router.add_post("/api/stream/start", start_streaming)
    app.router.add_post("/api/stream/stop", stop_streaming)
    app.router.add_post("/api/stream/reset", reset_streaming)
    app.router.add_get("/api/stream/status", get_streaming_status)
    app.router.add_get("/api/tool/pose", get_tool_pose)
    app.router.add_get("/api/objects/poses", get_object_poses)
    app.router.add_get("/api/objects/config", get_dope_objects)

    # DOPE control routes
    app.router.add_post("/api/dope/interval", set_dope_interval)

    # VGGT routes
    app.router.add_get("/api/vggt/pointcloud", get_vggt_point_cloud)
    app.router.add_get("/api/vggt/pointcloud/binary", get_vggt_point_cloud_binary)
    app.router.add_get("/api/vggt/pointcloud/version", get_vggt_point_cloud_version)
    app.router.add_get("/api/vggt/status", get_vggt_status)
    app.router.add_post("/api/vggt/enable", set_vggt_enabled)
    app.router.add_post("/api/vggt/interval", set_vggt_interval)

    # Tool-Case distance routes
    app.router.add_post("/api/distance/tool-case", set_tool_case_distance)
    app.router.add_get("/api/distance/tool-case", get_tool_case_distance)

    # Screw sequence tracking routes
    app.router.add_get("/api/screw/status", get_screw_status)
    app.router.add_post("/api/screw/reset", reset_screw_sequence)
    app.router.add_post("/api/screw/enable", set_screw_tracking_enabled)
    app.router.add_post("/api/screw/mode", set_screw_tracking_mode)
    # Static files
    app.router.add_static("/videos/", "data", show_index=False)

    app.on_shutdown.append(on_shutdown)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()

    yolo_status = f"enabled on {yolo_device}" if yolo_detector else "disabled"
    vggt_status = "loaded (disabled by default)" if vggt_detector else "not loaded"
    print(f"\n{'='*60}")
    print(f"  3D Scene Multi-Camera Server (Synchronized)")
    print(f"{'='*60}")
    print(f"  Web Interface:  http://{host}:{port}")
    print(f"  YOLO Camera:    {YOLO_CAMERA_ID} ({yolo_status})")
    dope_status = "enabled" if DOPE_ENABLED else "DISABLED"
    print(f"  DOPE Status:    {dope_status}")
    if DOPE_ENABLED:
        print(f"  DOPE Objects:")
        for obj_name, obj_config in DOPE_OBJECTS.items():
            loaded = "loaded" if obj_name in dope_detectors else "not loaded"
            print(f"    - {obj_name}: camera {obj_config['camera_id']} ({loaded})")
        print(f"  DOPE Settings:  FP16={DOPE_USE_FP16}, stages={DOPE_STOP_AT_STAGE}")
        print(f"  DOPE Interval:  Every {sync_manager.dope_inference_interval} frames")
    print(f"  VGGT Status:    {vggt_status}")
    print(f"  VGGT Cameras:   {VGGT_CAMERA_IDS}")
    print(
        f"  VGGT Interval:  Every {sync_manager.vggt_inference_interval} frames (configurable)"
    )
    screw_status = "enabled" if screw_tracker else "not initialized"
    print(f"  Screw Tracker:  {screw_status}")
    print(f"  Screw Order:    BL → TR → BR → TL (diagonal pattern)")
    print(f"  Total Frames:   {sync_manager.total_frames}")
    print(f"  Cameras:        {len(sync_manager.cameras)}")
    print(f"  Streaming:      Waiting for Start command")
    print(f"{'='*60}\n")

    # Start frame update loop (but it won't process until streaming_active is True)
    asyncio.create_task(frame_update_loop())

    try:
        await asyncio.Future()
    except KeyboardInterrupt:
        pass
    finally:
        await runner.cleanup()


# =============================================================================
# Main Entry Point
# =============================================================================


def init_screw_tracker():
    """Initialize the screw sequence tracker."""
    global screw_tracker

    screw_tracker = ScrewSequenceTracker()
    print(f"[ScrewTracker] Initialized - tracking screw sequence")
    return screw_tracker


def main():
    """Main entry point."""
    print(f"\n{'='*60}")
    print(f"  Initializing 3D Scene Multi-Camera Server")
    print(f"{'='*60}\n")

    # Load YOLO detector
    init_yolo()

    # Initialize LSTM error detection
    init_lstm_fusion()

    # Load DOPE models for 6D pose estimation (multi-object)
    load_dope_models()

    # Load VGGT model for 3D point cloud reconstruction
    load_vggt_model()

    # Load calibration data
    load_calibration()

    # Initialize synchronized video manager
    init_sync_manager()

    # Initialize screw sequence tracker
    init_screw_tracker()

    # Run server
    asyncio.run(run_server())


if __name__ == "__main__":
    main()
