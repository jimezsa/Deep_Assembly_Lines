import os
import cv2
import asyncio
import json
from ultralytics import YOLO
import numpy as np
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.contrib.media import MediaRelay
from av import VideoFrame
import time
from fractions import Fraction
import torch
import gc
from concurrent.futures import ThreadPoolExecutor

# Define CLASSES
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


class YOLOVideoStreamTrack(VideoStreamTrack):
    """
    Optimized video stream track for YOLO inference.
    """
    
    def __init__(self, video_path, model, device='cuda:0', use_fp16=False):
        super().__init__()
        self.kind = "video"
        self.video_path = video_path
        self.model = model
        self.device = device
        self.use_fp16 = use_fp16
        self.cap = cv2.VideoCapture(video_path)
        self.frame_count = 0
        
        # Target resolution
        self.target_width = 640
        self.target_height = 360
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video file: {video_path}")
        
        # Get video properties
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS)) or 30
        
        # CUDA setup
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            print(f"[GPU] {torch.cuda.get_device_name(0)}, FP16: {use_fp16}")
        
        # WebRTC timing
        self.VIDEO_CLOCK_RATE = 90000
        self.VIDEO_TIME_BASE = Fraction(1, self.VIDEO_CLOCK_RATE)
        self.pts_counter = 0
        
        # FPS calculation - use simple moving average
        self.fps_start_time = time.perf_counter()
        self.current_fps = 0.0
        
        # Pre-allocate overlay buffer (reused each frame to avoid allocation)
        self.overlay = np.zeros((self.target_height, self.target_width, 3), dtype=np.uint8)
        
        # Executor for blocking operations
        self.executor = ThreadPoolExecutor(max_workers=1)
        
        # Font settings (pre-computed once)
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.4
        self.font_thickness = 1
        
        # Frame skipping for performance (process every Nth frame)
        self.inference_interval = 3  # Process every 3rd frame
        self.inference_counter = 0
        self.cached_results = None  # Store last inference results

        print(f"[VideoTrack] Initialized: {self.target_width}x{self.target_height} @ {self.fps}fps")
        print(f"[VideoTrack] Frame skipping: Processing 1 out of every {self.inference_interval} frames")

    def _draw_fast(self, image, results):
        """
        Fast drawing with minimal allocations.
        Returns the same image (mutated in place).
        """
        # Check for any masks first
        has_masks = any(r.masks is not None and len(r.masks.xy) > 0 for r in results)
        if not has_masks:
            return image
        
        # Reuse overlay buffer - copy frame data into it
        np.copyto(self.overlay, image)
        
        mask_count = 0
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
                
                # Get color (handle out of bounds)
                color = tuple(int(c) for c in CLASS_COLORS[class_id % len(CLASS_COLORS)])
                
                # Convert to int32 for OpenCV
                pts = poly_np.astype(np.int32).reshape((-1, 1, 2))
                
                # Fill on overlay
                cv2.fillPoly(self.overlay, [pts], color)
                
                # Outline on original
                cv2.polylines(image, [pts], True, color, 1)
                
                # Label at first point
                label = f"{CLASSES[class_id] if class_id < len(CLASSES) else '?'} {conf:.2f}"
                tx, ty = int(poly_np[0][0]), int(poly_np[0][1]) - 5
                if ty < 15:
                    ty = int(poly_np[0][1]) + 15
                
                # Simple text (no background rectangle for speed)
                cv2.putText(image, label, (tx, ty), self.font, self.font_scale, 
                           (255, 255, 255), self.font_thickness, cv2.LINE_AA)
                
                mask_count += 1
        
        # Blend overlay with original (alpha = 0.4)
        cv2.addWeighted(self.overlay, 0.4, image, 0.6, 0, image)
        
        return image

    def _process_frame(self):
        """
        Synchronous frame processing - runs in thread pool.
        """
        ret, frame = self.cap.read()
        if not ret:
            # Loop video
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.frame_count = 0
            self.pts_counter = 0
            self.fps_start_time = time.perf_counter()
            self.inference_counter = 0
            self.cached_results = None
            ret, frame = self.cap.read()
            if not ret:
                raise Exception("Failed to read video frame")
        
        # Resize if needed (INTER_LINEAR is fast)
        h, w = frame.shape[:2]
        if w != self.target_width or h != self.target_height:
            frame = cv2.resize(frame, (self.target_width, self.target_height), 
                              interpolation=cv2.INTER_LINEAR)
        
        # Only run YOLO inference every Nth frame
        if self.inference_counter % self.inference_interval == 0:
            # Run YOLO inference
            results = self.model(
                frame, 
                verbose=False,
                device=self.device,
              
                conf=0.35,
                iou=0.45,
                max_det=20,
                imgsz=self.target_width  # Use width since it's larger
            )
            # Cache the results
            self.cached_results = results
        else:
            # Use cached results from previous inference
            results = self.cached_results
        
        # Increment inference counter
        self.inference_counter += 1
        
        # Draw predictions (modifies frame in place) if we have results
        if results is not None:
            self._draw_fast(frame, results)
        
        # Calculate FPS (simple: frames / total_time)
        self.frame_count += 1
        elapsed = time.perf_counter() - self.fps_start_time
        if elapsed > 0:
            self.current_fps = self.frame_count / elapsed
        
        # Reset counter periodically to get recent FPS
        if self.frame_count >= 100:
            self.frame_count = 0
            self.fps_start_time = time.perf_counter()
        
        # Display FPS
        cv2.putText(frame, f"FPS: {self.current_fps:.1f}", (10, 25),
                    self.font, 0.7, (0, 255, 0), 2)
        
        # Create VideoFrame - use bgr24 to avoid color conversion
        video_frame = VideoFrame.from_ndarray(frame, format="bgr24")
        video_frame.pts = self.pts_counter
        video_frame.time_base = self.VIDEO_TIME_BASE
        self.pts_counter += int(self.VIDEO_CLOCK_RATE / self.fps)
        
        return video_frame

    async def recv(self):
        """Get next video frame asynchronously."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.executor, self._process_frame)

    def stop(self):
        """Cleanup resources."""
        if self.cap:
            self.cap.release()
        self.executor.shutdown(wait=False)


# Global state
pcs = set()
relay = MediaRelay()
video_track = None
model = None


async def index(request):
    """Serve HTML page."""
    html_path = os.path.join(os.path.dirname(__file__), "webrtc_client.html")
    with open(html_path, "r") as f:
        content = f.read()
    return web.Response(content_type="text/html", text=content)


async def offer(request):
    """Handle WebRTC offer."""
    params = await request.json()
    offer_desc = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        print(f"Connection: {pc.connectionState}")
        if pc.connectionState in ("failed", "closed"):
            await pc.close()
            pcs.discard(pc)

    await pc.setRemoteDescription(offer_desc)
    
    global video_track
    if video_track:
        pc.addTrack(relay.subscribe(video_track))
        print(f"[Server] Track added")
    else:
        print("[Server] WARNING: No video track!")

    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})
    )


async def on_shutdown(app):
    """Cleanup on shutdown."""
    await asyncio.gather(*[pc.close() for pc in pcs])
    pcs.clear()
    if video_track:
        video_track.stop()


async def run_server(host="0.0.0.0", port=8080):
    """Run WebRTC server."""
    app = web.Application()
    app.router.add_get("/", index)
    app.router.add_post("/offer", offer)
    app.on_shutdown.append(on_shutdown)
    
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()
    
    print(f"\n{'='*50}")
    print(f"WebRTC server: http://{host}:{port}")
    print(f"{'='*50}\n")
    
    try:
        await asyncio.Future()
    except KeyboardInterrupt:
        pass
    finally:
        await runner.cleanup()


def main():
    """Main entry point."""
    global video_track, model
    
    USE_FP16 = True
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    print(f"\n{'='*50}")
    print(f"YOLO WebRTC Server")
    print(f"{'='*50}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"FP16: {USE_FP16}")
        torch.cuda.empty_cache()
        gc.collect()
    else:
        print("WARNING: CUDA not available, using CPU")
    
    # Load model (prefer TensorRT engine)
    engine_path = os.path.join('runs', 'segment', 'yolov11n_seg_custom', 'weights', 'best.engine')
    pt_path = os.path.join('runs', 'segment', 'yolov11n_seg_custom', 'weights', 'best.pt')
    
    model_path = engine_path if os.path.exists(engine_path) else pt_path
    print(f"Model: {model_path}")
    
    model = YOLO(model_path)
    
    # Warmup for .pt models
    if torch.cuda.is_available() and not model_path.endswith(".engine"):
        model.to(device)
        print("Warming up...")
        try:
            _ = model(np.zeros((320, 320, 3), dtype=np.uint8), verbose=False, device=device, half=USE_FP16)
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"Warmup warning: {e}")
    
    # Create video track
    video_path = os.path.join('testdata', 'rec7-89.mp4')
    video_track = YOLOVideoStreamTrack(video_path, model, device=device, use_fp16=USE_FP16)
    print(f"Video: {video_path}")
    print(f"{'='*50}\n")
    
    asyncio.run(run_server())


if __name__ == "__main__":
    main()
