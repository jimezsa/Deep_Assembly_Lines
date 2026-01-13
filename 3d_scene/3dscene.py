import os
import cv2
import asyncio
import json
import yaml
from aiohttp import web
import numpy as np
import time
import threading

# Camera recording directory
RECORDING_DIR = "data/recording_7"
CALIBRATION_FILE = "data/cams_calibrations.yml"

# Camera to run YOLO on
YOLO_CAMERA_ID = "137322071489"

# Define CLASSES for YOLO
CLASSES = ["person", "case", "case_top", "battery", "screw", "tool"]

# Pre-define colors as numpy array for faster access (BGR format)
CLASS_COLORS = np.array([
    [50, 0, 0],       # person: blueish
    [0, 165, 255],    # case: orange
    [0, 40, 75],      # case_top: yellow
    [192, 192, 192],  # battery: silver
    [140, 0, 140],    # screw: violet
    [0, 200, 0]       # tool: green
], dtype=np.uint8)

# Global state
calibration_data = {}
yolo_model = None
yolo_device = "cpu"  # Will be set to 'mps', 'cuda', or 'cpu'
sync_manager = None
streaming_active = False  # Controls whether frame processing is active


def get_best_device():
    """Detect the best available device for inference (MPS > CUDA > CPU)."""
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
        self.yolo_model = None
        self.yolo_camera_id = None
        self.yolo_device = "cpu"
        self.inference_interval = 4
        self.inference_counter = 0
        self.cached_yolo_results = None
        self.yolo_overlay = None
        
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
    
    def set_yolo_model(self, model, camera_id, device="cpu"):
        """Set YOLO model for a specific camera."""
        self.yolo_model = model
        self.yolo_camera_id = camera_id
        self.yolo_device = device
        self.yolo_overlay = np.zeros((self.target_height, self.target_width, 3), dtype=np.uint8)
        print(f"[YOLO] Enabled on camera {camera_id} (device: {device})")
    
    def _draw_yolo_predictions(self, frame, results):
        """Draw YOLO predictions on frame."""
        if results is None:
            return frame
            
        has_masks = any(r.masks is not None and len(r.masks.xy) > 0 for r in results)
        if not has_masks:
            return frame
        
        np.copyto(self.yolo_overlay, frame)
        
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
                h_scale = self.target_height / r.orig_shape[0]
                w_scale = self.target_width / r.orig_shape[1]
                scaled_poly = poly_np.copy()
                scaled_poly[:, 0] *= w_scale
                scaled_poly[:, 1] *= h_scale
                
                color = tuple(int(c) for c in CLASS_COLORS[class_id % len(CLASS_COLORS)])
                pts = scaled_poly.astype(np.int32).reshape((-1, 1, 2))
                
                cv2.fillPoly(self.yolo_overlay, [pts], color)
                cv2.polylines(frame, [pts], True, color, 1)
                
                label = f"{CLASSES[class_id] if class_id < len(CLASSES) else '?'} {conf:.2f}"
                tx, ty = int(scaled_poly[0][0]), int(scaled_poly[0][1]) - 3
                if ty < 10:
                    ty = int(scaled_poly[0][1]) + 10
                
                cv2.putText(frame, label, (tx, ty), self.font, 0.2, 
                           (255, 255, 255), 1, cv2.LINE_AA)
        
        cv2.addWeighted(self.yolo_overlay, 0.4, frame, 0.6, 0, frame)
        return frame
    
    def reset_playback(self):
        """Reset playback to the beginning."""
        with self.lock:
            self.current_frame = 0
            self.inference_counter = 0
            self.cached_yolo_results = None
            self.fps_start_time = time.perf_counter()
            self.fps_frame_count = 0
            
            # Seek all cameras to frame 0
            for cam_data in self.cameras.values():
                cam_data['cap'].set(cv2.CAP_PROP_POS_FRAMES, 0)
                cam_data['cached_jpeg'] = None
            
            print("[SyncManager] Playback reset to frame 0")
    
    def advance_frame(self):
        """Advance to the next frame and update all cameras."""
        with self.lock:
            self.current_frame += 1
            if self.current_frame >= self.total_frames:
                self.current_frame = 0
                self.inference_counter = 0
                self.cached_yolo_results = None
            
            # Track FPS
            self.fps_frame_count += 1
            elapsed = time.perf_counter() - self.fps_start_time
            if elapsed > 0:
                self.current_fps = self.fps_frame_count / elapsed
            if self.fps_frame_count >= 50:
                self.fps_frame_count = 0
                self.fps_start_time = time.perf_counter()
            
            # Read and cache frames for all cameras
            for camera_id, cam_data in self.cameras.items():
                cap = cam_data['cap']
                
                # Seek to current frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
                ret, frame = cap.read()
                
                if not ret:
                    cam_data['cached_jpeg'] = None
                    continue
                
                # Run YOLO on the designated camera
                if camera_id == self.yolo_camera_id and self.yolo_model is not None:
                    # Run inference periodically
                    if self.inference_counter % self.inference_interval == 0:
                        try:
                            results = self.yolo_model(
                                frame,
                                verbose=False,
                                conf=0.35,
                                iou=0.45,
                                max_det=20,
                                imgsz=640,
                                device=self.yolo_device
                            )
                            self.cached_yolo_results = results
                        except Exception as e:
                            print(f"[YOLO] Error: {e}")
                            self.cached_yolo_results = None
                
                # Resize for display
                frame = cv2.resize(frame, (self.target_width, self.target_height), 
                                  interpolation=cv2.INTER_LINEAR)
                
                # Apply YOLO overlay for the designated camera
                if camera_id == self.yolo_camera_id and self.yolo_model is not None:
                    frame = self._draw_yolo_predictions(frame, self.cached_yolo_results)
                    # Draw FPS
                    cv2.putText(frame, f"FPS: {self.current_fps:.1f}", (5, 15),
                                self.font, 0.45, (0, 255, 0), 1)
                
                # Encode as JPEG
                _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                cam_data['cached_jpeg'] = jpeg.tobytes()
            
            self.inference_counter += 1
    
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
    
    frame_interval = 1.0 / 15  # 15 FPS
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


def load_yolo_model():
    """Load YOLO model with best available device (MPS/CUDA/CPU)."""
    global yolo_model, yolo_device
    
    try:
        from ultralytics import YOLO
        
        # Detect best device
        yolo_device = get_best_device()
        
        pt_path = os.path.join('yolov11_finetuned', 'runs', 'segment', 
                               'yolov11n_seg_custom', 'weights', 'best.pt')
        
        if os.path.exists(pt_path):
            print(f"[YOLO] Loading model: {pt_path}")
            yolo_model = YOLO(pt_path)
            
            print(f"[YOLO] Warming up on {yolo_device}...")
            _ = yolo_model(np.zeros((320, 320, 3), dtype=np.uint8), 
                          verbose=False, device=yolo_device)
            print(f"[YOLO] Model ready on {yolo_device}")
            return yolo_model
        else:
            print(f"[YOLO] Model not found: {pt_path}")
            return None
            
    except ImportError:
        print("[YOLO] ultralytics not installed, YOLO disabled")
        return None
    except Exception as e:
        print(f"[YOLO] Failed to load model: {e}")
        return None


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
    global sync_manager, yolo_model
    
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
    
    # Set YOLO model on designated camera
    if yolo_model is not None:
        sync_manager.set_yolo_model(yolo_model, YOLO_CAMERA_ID, yolo_device)
    
    print(f"[SyncManager] Initialized with {len(sync_manager.cameras)} cameras, synced to {sync_manager.total_frames} frames")


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


async def on_shutdown(app):
    """Cleanup on shutdown."""
    global streaming_active
    streaming_active = False
    if sync_manager:
        sync_manager.release()


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
    
    # Static files
    app.router.add_static('/videos/', 'data', show_index=False)
    
    app.on_shutdown.append(on_shutdown)
    
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()
    
    yolo_status = f"enabled on {yolo_device}" if yolo_model else "disabled"
    print(f"\n{'='*60}")
    print(f"  3D Scene Multi-Camera Server (Synchronized)")
    print(f"{'='*60}")
    print(f"  Web Interface:  http://{host}:{port}")
    print(f"  YOLO Camera:    {YOLO_CAMERA_ID} ({yolo_status})")
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


def main():
    """Main entry point."""
    print(f"\n{'='*60}")
    print(f"  Initializing 3D Scene Multi-Camera Server")
    print(f"{'='*60}\n")
    
    # Load YOLO model first
    load_yolo_model()
    
    # Load calibration data
    load_calibration()
    
    # Initialize synchronized video manager
    init_sync_manager()
    
    # Run server
    asyncio.run(run_server())


if __name__ == "__main__":
    main()
