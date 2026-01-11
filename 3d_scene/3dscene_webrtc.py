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
sync_manager = None


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
        self.inference_interval = 2
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
    
    def set_yolo_model(self, model, camera_id):
        """Set YOLO model for a specific camera."""
        self.yolo_model = model
        self.yolo_camera_id = camera_id
        self.yolo_overlay = np.zeros((self.target_height, self.target_width, 3), dtype=np.uint8)
        print(f"[YOLO] Enabled on camera {camera_id}")
    
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
                                imgsz=640
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
    """Background task that advances frames at a fixed rate."""
    global sync_manager
    
    frame_interval = 1.0 / 15  # 15 FPS
    loop = asyncio.get_running_loop()
    
    while True:
        start_time = time.perf_counter()
        
        # Advance frame in thread pool (blocking operations)
        await loop.run_in_executor(None, sync_manager.advance_frame)
        
        # Maintain frame rate
        elapsed = time.perf_counter() - start_time
        if elapsed < frame_interval:
            await asyncio.sleep(frame_interval - elapsed)


def load_yolo_model():
    """Load YOLO model."""
    global yolo_model
    
    try:
        from ultralytics import YOLO
        
        pt_path = os.path.join('yolov11_finetuned', 'runs', 'segment', 
                               'yolov11n_seg_custom', 'weights', 'best.pt')
        
        if os.path.exists(pt_path):
            print(f"[YOLO] Loading model: {pt_path}")
            yolo_model = YOLO(pt_path)
            
            print("[YOLO] Warming up...")
            _ = yolo_model(np.zeros((320, 320, 3), dtype=np.uint8), verbose=False)
            print("[YOLO] Model ready")
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
    """Load camera calibration data from YAML file."""
    global calibration_data
    
    with open(CALIBRATION_FILE, 'r') as f:
        calibration_data = yaml.safe_load(f)
    
    print(f"[Calibration] Loaded {len(calibration_data)} cameras")
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
        sync_manager.set_yolo_model(yolo_model, YOLO_CAMERA_ID)
    
    print(f"[SyncManager] Initialized with {len(sync_manager.cameras)} cameras, synced to {sync_manager.total_frames} frames")


async def index(request):
    """Serve HTML page."""
    html_path = os.path.join(os.path.dirname(__file__), "webrtc_client.html")
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


async def on_shutdown(app):
    """Cleanup on shutdown."""
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
    
    app.on_shutdown.append(on_shutdown)
    
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()
    
    yolo_status = "enabled" if yolo_model else "disabled"
    print(f"\n{'='*60}")
    print(f"  3D Scene Multi-Camera Server (Synchronized)")
    print(f"{'='*60}")
    print(f"  Web Interface:  http://{host}:{port}")
    print(f"  YOLO Camera:    {YOLO_CAMERA_ID} ({yolo_status})")
    print(f"  Total Frames:   {sync_manager.total_frames}")
    print(f"  Cameras:        {len(sync_manager.cameras)}")
    print(f"{'='*60}\n")
    
    # Start frame update loop
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
