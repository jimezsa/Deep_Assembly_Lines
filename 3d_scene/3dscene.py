import os
import cv2
import asyncio
import json
import yaml
from aiohttp import web
import numpy as np
import time
import threading

# Import inference modules
from dope_inference import load_dope_detector, create_empty_pose
from yolo_inference import load_yolo_model, YOLODetector

# =============================================================================
# Configuration
# =============================================================================

# Camera recording directory
RECORDING_DIR = "data/recording_3"
CALIBRATION_FILE = "data/cams_calibrations.yml"

# Camera to run YOLO on
YOLO_CAMERA_ID = "137322071489"

# DOPE configuration
DOPE_CONFIG_PATH = "3d_scene/config/config_pose.yaml"
DOPE_CAMERA_INFO_PATH = "3d_scene/config/camera_info.yaml"

# Multiple object configurations for DOPE detection (each with its own camera)
DOPE_OBJECTS = {
    "tool": {
        "weights_path": "weights/dope_tool.pth",
        "class_name": "tool",
        "obj_path": "data/scanned_objects/e-screw-driver/eScrewDriver.obj",
        "camera_id": "142122070087"
    },
    "case": {
        "weights_path": "weights/dope_case.pth",
        "class_name": "case",
        "obj_path": "data/scanned_objects/case/case.obj",
        "camera_id": "135122071615"
    }
}

# =============================================================================
# Global State
# =============================================================================

calibration_data = {}
yolo_detector = None  # YOLODetector instance
yolo_device = "cpu"
sync_manager = None
streaming_active = False  # Controls whether frame processing is active

# DOPE 6D pose estimation state (multi-object support)
dope_detectors = {}  # object_name -> DOPEDetector instance
current_object_poses = {}  # object_name -> pose dict
pose_lock = threading.Lock()


# =============================================================================
# Video Manager
# =============================================================================

class SyncedVideoManager:
    """Manages synchronized playback across all cameras."""
    
    def __init__(self):
        self.cameras = {}  # camera_id -> dict with cap, cached_frame, etc.
        self.current_frame = 0
        self.total_frames = 0
        self.fps = 30
        self.target_width = 320
        self.target_height = 180
        self.lock = threading.Lock()
        
        # YOLO state
        self.yolo_detector = None  # YOLODetector instance
        self.yolo_camera_id = None
        self.yolo_inference_interval = 3  # Run YOLO every 3 frames (like WebRTC)
        self.yolo_inference_counter = 0
        self.cached_yolo_results = None
        
        # DOPE 6D pose detection state (multi-object, per-camera support)
        self.dope_detectors = {}  # object_name -> {"detector": DOPEDetector, "camera_id": str}
        self.dope_camera_ids = set()  # All cameras used for DOPE
        self.dope_inference_interval = 5  # Run DOPE every 5 frames
        self.dope_inference_counter = 0
        self.cached_dope_results = {}  # object_name -> detection result
        
        # FPS tracking
        self.fps_start_time = time.perf_counter()
        self.fps_frame_count = 0
        self.current_fps = 0.0
        
        # Font settings
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        
    def add_camera(self, camera_id, video_path):
        """Add a camera to the manager."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file: {video_path}")
        
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.cameras[camera_id] = {
            'cap': cap,
            'video_path': video_path,
            'total_frames': total_frames,
            'fps': fps,
            'cached_jpeg': None
        }
        
        # Use the minimum total frames across all cameras
        if self.total_frames == 0:
            self.total_frames = total_frames
            self.fps = fps
        else:
            self.total_frames = min(self.total_frames, total_frames)
        
        print(f"[Camera {camera_id}] Loaded: {video_path} ({total_frames} frames @ {fps}fps)")
    
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
    
    def set_dope_detector(self, detector, camera_id, object_name="tool"):
        """Set DOPE detector for 6D pose estimation on a specific camera.
        
        Args:
            detector: DOPEDetector instance
            camera_id: Camera ID to run DOPE on
            object_name: Name of the object being detected
        """
        self.dope_detectors[object_name] = {
            "detector": detector,
            "camera_id": camera_id
        }
        self.dope_camera_ids.add(camera_id)
        print(f"[DOPE] Enabled '{object_name}' detector on camera {camera_id}")
    
    def reset_playback(self):
        """Reset playback to the beginning."""
        global current_object_poses
        with self.lock:
            self.current_frame = 0
            self.yolo_inference_counter = 0
            self.cached_yolo_results = None
            self.dope_inference_counter = 0
            self.cached_dope_results = {}
            self.fps_start_time = time.perf_counter()
            self.fps_frame_count = 0
            
            # Reset all object poses
            with pose_lock:
                current_object_poses = {name: create_empty_pose() for name in self.dope_detectors}
            
            # Seek all cameras to frame 0 (only time we seek)
            for cam_data in self.cameras.values():
                cam_data['cap'].set(cv2.CAP_PROP_POS_FRAMES, 0)
                cam_data['cached_jpeg'] = None
            
            print("[SyncManager] Playback reset to frame 0")
    
    def advance_frame(self):
        """Advance to the next frame and update all cameras (optimized sequential read)."""
        with self.lock:
            self.current_frame += 1
            
            # Track FPS
            self.fps_frame_count += 1
            elapsed = time.perf_counter() - self.fps_start_time
            if elapsed > 0:
                self.current_fps = self.fps_frame_count / elapsed
            if self.fps_frame_count >= 100:  # Reset less frequently
                self.fps_frame_count = 0
                self.fps_start_time = time.perf_counter()
            
            # Read and cache frames for all cameras
            for camera_id, cam_data in self.cameras.items():
                cap = cam_data['cap']
                
                # Read next frame SEQUENTIALLY (much faster than seeking)
                ret, frame = cap.read()
                
                # Loop video if we reach the end
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.current_frame = 0
                    self.yolo_inference_counter = 0
                    self.cached_yolo_results = None
                    self.dope_inference_counter = 0
                    self.cached_dope_result = None
                    self.fps_start_time = time.perf_counter()
                    self.fps_frame_count = 0
                    ret, frame = cap.read()
                    if not ret:
                        cam_data['cached_jpeg'] = None
                        continue
                
                # Run YOLO on the designated camera
                if camera_id == self.yolo_camera_id and self.yolo_detector is not None:
                    if self.yolo_inference_counter % self.yolo_inference_interval == 0:
                        self.cached_yolo_results = self.yolo_detector.detect(frame)
                
                # Run DOPE 6D pose detection (per-object camera assignment)
                if camera_id in self.dope_camera_ids and self.dope_detectors:
                    if self.dope_inference_counter % self.dope_inference_interval == 0:
                        global current_object_poses
                        # Run detectors assigned to this camera
                        for obj_name, dope_info in self.dope_detectors.items():
                            if dope_info["camera_id"] != camera_id:
                                continue
                            detector = dope_info["detector"]
                            try:
                                result = detector.detect(frame)
                                with pose_lock:
                                    if result is not None and result["detected"]:
                                        result["object_name"] = obj_name
                                        result["camera_id"] = camera_id
                                        current_object_poses[obj_name] = result
                                        current_object_poses[obj_name]["fresh"] = True
                                        self.cached_dope_results[obj_name] = result
                                    else:
                                        if obj_name in current_object_poses:
                                            current_object_poses[obj_name]["fresh"] = False
                            except Exception as e:
                                print(f"[DOPE] Error detecting {obj_name}: {e}")
                    
                    # Draw DOPE detections for objects assigned to this camera
                    for obj_name, dope_info in self.dope_detectors.items():
                        if dope_info["camera_id"] != camera_id:
                            continue
                        if obj_name in self.cached_dope_results:
                            detector = dope_info["detector"]
                            frame = detector.draw_detection(frame, self.cached_dope_results[obj_name])
                
                # Resize for display
                frame = cv2.resize(frame, (self.target_width, self.target_height), 
                                  interpolation=cv2.INTER_LINEAR)
                
                # Apply YOLO overlay for the designated camera
                if camera_id == self.yolo_camera_id and self.yolo_detector is not None:
                    frame = self.yolo_detector.draw_predictions(
                        frame, self.cached_yolo_results,
                        target_width=self.target_width, 
                        target_height=self.target_height
                    )
                    cv2.putText(frame, f"FPS: {self.current_fps:.1f}", (5, 15),
                                self.font, 0.45, (0, 255, 0), 1)
                
                # Draw DOPE status for cameras running DOPE
                if camera_id in self.dope_camera_ids and self.dope_detectors:
                    # Count objects detected on this camera
                    camera_objects = [name for name, info in self.dope_detectors.items() if info["camera_id"] == camera_id]
                    detected_on_cam = sum(1 for name in camera_objects if name in self.cached_dope_results and self.cached_dope_results[name].get("detected"))
                    status = f"{detected_on_cam}/{len(camera_objects)} detected"
                    color = (0, 255, 0) if detected_on_cam > 0 else (0, 165, 255)
                    cv2.putText(frame, f"DOPE: {status}", (5, 15), self.font, 0.4, color, 1)
                
                # Encode as JPEG
                _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                cam_data['cached_jpeg'] = jpeg.tobytes()
            
            self.yolo_inference_counter += 1
            self.dope_inference_counter += 1
    
    def get_frame(self, camera_id):
        """Get cached frame for a camera."""
        with self.lock:
            if camera_id in self.cameras:
                return self.cameras[camera_id]['cached_jpeg']
            return None
    
    def get_camera_ids(self):
        """Get list of camera IDs."""
        return list(self.cameras.keys())
    
    def release(self):
        """Release all video captures."""
        for cam_data in self.cameras.values():
            if cam_data['cap']:
                cam_data['cap'].release()


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
    global yolo_detector, yolo_device
    
    yolo_detector, yolo_device = load_yolo_model()
    return yolo_detector


def load_dope_models():
    """Load DOPE models for 6D pose estimation (multi-object support)."""
    global dope_detectors, current_object_poses
    
    dope_detectors = {}
    current_object_poses = {}
    
    for obj_name, obj_config in DOPE_OBJECTS.items():
        weights_path = obj_config["weights_path"]
        class_name = obj_config["class_name"]
        
        detector = load_dope_detector(
            weights_path=weights_path,
            config_path=DOPE_CONFIG_PATH,
            camera_info_path=DOPE_CAMERA_INFO_PATH,
            class_name=class_name
        )
        
        if detector is not None:
            dope_detectors[obj_name] = detector
            current_object_poses[obj_name] = create_empty_pose()
            print(f"[DOPE] Loaded detector for '{obj_name}'")
        else:
            print(f"[DOPE] Warning: Could not load detector for '{obj_name}' (weights: {weights_path})")
    
    return dope_detectors


def load_calibration():
    """Load camera calibration data from YAML file and transform to checkerboard origin."""
    global calibration_data
    
    with open(CALIBRATION_FILE, 'r') as f:
        raw_calibration = yaml.safe_load(f)
    
    # Find the master camera with checkerboard transform
    checkerboard_to_master = None
    for cam_id, cam_data in raw_calibration.items():
        if cam_data.get('master', False) and 'checkerboard' in cam_data:
            checkerboard_to_master = np.array(cam_data['checkerboard'], dtype=np.float64)
            print(f"[Calibration] Found checkerboard transform in master camera {cam_id}")
            break
    
    if checkerboard_to_master is None:
        print("[Calibration] WARNING: No checkerboard transform found, using master camera as origin")
        calibration_data = raw_calibration
        return calibration_data
    
    # Compute inverse: master_to_checkerboard (this transforms points from master to checkerboard frame)
    master_to_checkerboard = np.linalg.inv(checkerboard_to_master)
    print(f"[Calibration] Transforming all cameras to checkerboard origin")
    
    # Transform all camera extrinsics to be relative to checkerboard
    calibration_data = {}
    for cam_id, cam_data in raw_calibration.items():
        cam_extrinsics = np.array(cam_data['extrinsics'], dtype=np.float64)
        
        # New extrinsics: camera_to_checkerboard = master_to_checkerboard @ camera_to_master
        # Since extrinsics are camera_to_master (or camera_to_world where world=master)
        # We need: camera_to_checkerboard = master_to_checkerboard @ camera_to_master
        new_extrinsics = master_to_checkerboard @ cam_extrinsics
        
        # Copy camera data with transformed extrinsics
        calibration_data[cam_id] = {
            'extrinsics': new_extrinsics.tolist(),
            'intrinsics': cam_data['intrinsics'],
            'number': cam_data.get('number', 0),
            'master': cam_data.get('master', False)
        }
        
        # Keep checkerboard info if present
        if 'checkerboard' in cam_data:
            calibration_data[cam_id]['checkerboard'] = cam_data['checkerboard']
    
    print(f"[Calibration] Loaded {len(calibration_data)} cameras (checkerboard origin)")
    return calibration_data


def init_sync_manager():
    """Initialize synchronized video manager."""
    global sync_manager, yolo_detector, dope_detectors

    sync_manager = SyncedVideoManager()

    # Get all camera directories
    camera_dirs = [d for d in os.listdir(RECORDING_DIR)
                   if os.path.isdir(os.path.join(RECORDING_DIR, d)) and d.isdigit()]

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
    
    # Set DOPE detectors with per-object camera assignments
    for obj_name, detector in dope_detectors.items():
        camera_id = DOPE_OBJECTS[obj_name].get("camera_id", "142122070087")
        sync_manager.set_dope_detector(detector, camera_id, obj_name)
    
    print(f"[SyncManager] Initialized with {len(sync_manager.cameras)} cameras, synced to {sync_manager.total_frames} frames")


# =============================================================================
# API Routes
# =============================================================================

async def index(request):
    """Serve HTML page."""
    html_path = os.path.join(os.path.dirname(__file__), "web_interface.html")
    with open(html_path, "r") as f:
        content = f.read()
    return web.Response(content_type="text/html", text=content)


async def get_calibration(request):
    """Return camera calibration data as JSON."""
    result = {}
    for cam_id, cam_data in calibration_data.items():
        result[cam_id] = {
            'extrinsics': cam_data.get('extrinsics', []),
            'intrinsics': cam_data.get('intrinsics', {}),
            'number': cam_data.get('number', 0),
            'master': cam_data.get('master', False)
        }
    
    return web.Response(
        content_type="application/json",
        text=json.dumps(result)
    )


async def get_cameras(request):
    """Return list of available cameras."""
    cameras = sync_manager.get_camera_ids()
    return web.Response(
        content_type="application/json",
        text=json.dumps(cameras)
    )


async def get_frame(request):
    """Get current frame from a specific camera as JPEG."""
    camera_id = request.match_info.get('camera_id')
    
    jpeg_data = sync_manager.get_frame(camera_id)
    
    if jpeg_data is None:
        return web.Response(status=404, text=f"Camera {camera_id} not found or no frame")
    
    return web.Response(
        body=jpeg_data,
        content_type="image/jpeg",
        headers={
            'Cache-Control': 'no-cache, no-store, must-revalidate',
            'Pragma': 'no-cache',
            'Expires': '0'
        }
    )


async def start_streaming(request):
    """Start the video streaming/processing."""
    global streaming_active
    streaming_active = True
    print("[Server] Streaming STARTED")
    return web.Response(
        content_type="application/json",
        text=json.dumps({"status": "started", "streaming": True})
    )


async def stop_streaming(request):
    """Stop the video streaming/processing."""
    global streaming_active
    streaming_active = False
    print("[Server] Streaming STOPPED")
    return web.Response(
        content_type="application/json",
        text=json.dumps({"status": "stopped", "streaming": False})
    )


async def reset_streaming(request):
    """Reset playback to the beginning."""
    global sync_manager
    sync_manager.reset_playback()
    return web.Response(
        content_type="application/json",
        text=json.dumps({"status": "reset", "frame": 0})
    )


async def get_streaming_status(request):
    """Get current streaming status."""
    global streaming_active
    return web.Response(
        content_type="application/json",
        text=json.dumps({"streaming": streaming_active})
    )


async def get_tool_pose(request):
    """Get current 6D pose of the detected tool object (legacy endpoint).
    
    Returns position (x, y, z) in meters and quaternion (x, y, z, w).
    The pose is in camera coordinate system.
    """
    global current_object_poses
    with pose_lock:
        # Return tool pose for backward compatibility
        pose_data = current_object_poses.get("tool", create_empty_pose()).copy()
    
    # Convert numpy arrays to lists for JSON serialization
    def to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (list, tuple)):
            return [to_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        return obj
    
    pose_data = to_serializable(pose_data)
    
    return web.Response(
        content_type="application/json",
        text=json.dumps(pose_data),
        headers={
            'Cache-Control': 'no-cache, no-store, must-revalidate',
            'Pragma': 'no-cache',
            'Expires': '0'
        }
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
    
    # Convert numpy arrays to lists for JSON serialization
    def to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (list, tuple)):
            return [to_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        return obj
    
    poses_data = to_serializable(poses_data)
    
    return web.Response(
        content_type="application/json",
        text=json.dumps(poses_data),
        headers={
            'Cache-Control': 'no-cache, no-store, must-revalidate',
            'Pragma': 'no-cache',
            'Expires': '0'
        }
    )


async def get_dope_objects(request):
    """Get list of configured DOPE objects and their OBJ file paths."""
    objects_info = {}
    for obj_name, obj_config in DOPE_OBJECTS.items():
        objects_info[obj_name] = {
            "obj_path": obj_config["obj_path"],
            "class_name": obj_config["class_name"],
            "camera_id": obj_config["camera_id"],
            "loaded": obj_name in dope_detectors
        }
    
    return web.Response(
        content_type="application/json",
        text=json.dumps(objects_info)
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

async def run_server(host="0.0.0.0", port=8080):
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
    
    # Static files
    app.router.add_static('/videos/', 'data', show_index=False)
    
    app.on_shutdown.append(on_shutdown)
    
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()
    
    yolo_status = f"enabled on {yolo_device}" if yolo_detector else "disabled"
    print(f"\n{'='*60}")
    print(f"  3D Scene Multi-Camera Server (Synchronized)")
    print(f"{'='*60}")
    print(f"  Web Interface:  http://{host}:{port}")
    print(f"  YOLO Camera:    {YOLO_CAMERA_ID} ({yolo_status})")
    print(f"  DOPE Objects:")
    for obj_name, obj_config in DOPE_OBJECTS.items():
        loaded = "loaded" if obj_name in dope_detectors else "not loaded"
        print(f"    - {obj_name}: camera {obj_config['camera_id']} ({loaded})")
    print(f"  DOPE Interval:  Every 5 frames")
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

def main():
    """Main entry point."""
    print(f"\n{'='*60}")
    print(f"  Initializing 3D Scene Multi-Camera Server")
    print(f"{'='*60}\n")

    # Load YOLO detector
    init_yolo()

    # Load DOPE models for 6D pose estimation (multi-object)
    load_dope_models()

    # Load calibration data
    load_calibration()

    # Initialize synchronized video manager
    init_sync_manager()

    # Run server
    asyncio.run(run_server())


if __name__ == "__main__":
    main()
