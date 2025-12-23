import os
import cv2
import asyncio
import random
import json
from ultralytics import YOLO
import numpy as np
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.contrib.media import MediaRelay
from av import VideoFrame
import threading
import queue
import time
from fractions import Fraction
import torch
import gc

# Define CLASSES
CLASSES = ["person", "case", "case_top", "battery", "screw", "tool"]

def draw_rect_boxes_and_labels(image, boxes, classes):
    """
    Draws rectangular bounding boxes and class labels on an image.
    """
    img_copy = image.copy()

    # Define a set of distinct colors for classes (BGR format for OpenCV)
    class_colors = [
      (0, 0, 70),   # person: bluelish
      (255, 165, 0), # case: orange
      (180, 100, 0),  # case_top:
      (0, 100, 100),  # battery: Grenn
      (128, 0, 128),   # screw: violet
      (0, 100, 0)  # tool: Green
    ]

    if boxes is not None and len(boxes) > 0:
        for box_data in boxes:
            x1, y1, x2, y2 = map(int, box_data.xyxy[0])
            class_id = int(box_data.cls[0])
            confidence = float(box_data.conf[0]) # Extract confidence score

            # Get color for the class
            color = class_colors[class_id % len(class_colors)] if class_id < len(class_colors) else (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

            # Draw rectangle
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)

            # Put label
            label = classes[class_id] if class_id < len(classes) else f"Unknown Class {class_id}"
            label_with_score = f"{label} {confidence:.2f}" # Append confidence score
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4
            font_thickness = 1
            text_size = cv2.getTextSize(label_with_score, font, font_scale, font_thickness)[0]

            # Place text above the bounding box
            text_x = x1
            text_y = y1 - 10 if y1 - 10 > text_size[1] else y1 + text_size[1] + 10

            # Draw background for text
            cv2.rectangle(img_copy, (text_x, text_y - text_size[1] - 5),
                          (text_x + text_size[0] + 5, text_y + 5), color, -1)
            cv2.putText(img_copy, label_with_score, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

    return img_copy


class YOLOVideoStreamTrack(VideoStreamTrack):
    """
    A video stream track that processes video frames through YOLO model.
    """
    
    def __init__(self, video_path, model, device='cuda:0', use_fp16=False):
        super().__init__()
        self.kind = "video"  # Explicitly set track kind
        self.video_path = video_path
        self.model = model
        self.device = device
        self.use_fp16 = use_fp16
        self.cap = cv2.VideoCapture(video_path)
        self.frame_count = 0
        # Downscale frames aggressively to reduce GPU memory
        self.target_width = 640
        self.target_height = 360
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video file: {video_path}")
        
        # Get video properties
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        if self.fps == 0:
            self.fps = 30  # Default fallback
        
        # Enable CUDA optimizations if available
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True  # Enable cuDNN auto-tuner
            print(f"[GPU] CUDA enabled: {torch.cuda.get_device_name(0)}")
            print(f"[GPU] CUDA version: {torch.version.cuda}")
            print(f"[GPU] Memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB")
            print(f"[GPU] FP16 mode: {'Enabled' if use_fp16 else 'Disabled (more stable)'}")
        else:
            print("[WARNING] CUDA not available, using CPU")
        
        # WebRTC timing
        self.VIDEO_CLOCK_RATE = 90000
        self.VIDEO_TIME_BASE = Fraction(1, self.VIDEO_CLOCK_RATE)
        self.frame_duration = 1.0 / self.fps
        self.pts_counter = 0
        
        # FPS calculation
        self.last_fps_time = time.time()
        self.fps_frame_count = 0
        self.current_fps = 0.0
        self.fps_update_interval = 10  # Update FPS every N frames
        
        print(f"Video FPS: {self.fps}")
        print(f"[VideoTrack] Initialized track, kind={self.kind}, readyState={self.readyState}")

    async def recv(self):
        """
        Get the next video frame with YOLO predictions.
        """
        # Debug: log first call
        if self.frame_count == 0:
            print(f"[VideoTrack] recv() called for first time!")
        
        # Wait for next frame timing
        await asyncio.sleep(self.frame_duration)
        
        ret, frame = self.cap.read()
        if not ret:
            # Loop the video
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.frame_count = 0
            self.pts_counter = 0
            # Reset FPS calculation
            self.last_fps_time = time.time()
            self.fps_frame_count = 0
            ret, frame = self.cap.read()
            if not ret:
                raise Exception("Failed to read video frame")
        
        # Downscale to reduce memory and compute load
        if frame.shape[1] > self.target_width:
            frame = cv2.resize(frame, (self.target_width, self.target_height), interpolation=cv2.INTER_AREA)
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run YOLO inference with GPU optimizations
        try:
            results = self.model(
                img_rgb, 
                verbose=False,
                device=self.device,
                half=self.use_fp16,  # FP16 optional (can cause memory issues)
                conf=0.35,  # Slightly higher to reduce detections
                iou=0.45,   # IoU threshold for NMS
                max_det=20,  # Strongly reduced to save memory
                imgsz=max(self.target_width, self.target_height)  # Keep inference small
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"[WARNING] GPU Out of Memory - clearing cache and retrying...")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
                # Retry with cleared memory
                results = self.model(
                    img_rgb, 
                    verbose=False,
                    device=self.device,
                    half=False,  # Disable FP16 on retry
                    conf=0.35,
                    iou=0.45,
                    max_det=20,
                    imgsz=max(self.target_width, self.target_height)
                )
            else:
                raise
        
        # Collect all detected boxes
        all_detected_boxes = []
        for r in results:
            if hasattr(r, 'boxes') and r.boxes is not None and len(r.boxes) > 0:
                all_detected_boxes.extend(r.boxes)
        
        # Draw predictions
        img_with_predictions = draw_rect_boxes_and_labels(img_rgb.copy(), all_detected_boxes, CLASSES)
        
        # Calculate FPS
        self.fps_frame_count += 1
        if self.fps_frame_count >= self.fps_update_interval:
            current_time = time.time()
            elapsed = current_time - self.last_fps_time
            self.current_fps = self.fps_frame_count / elapsed
            self.last_fps_time = current_time
            self.fps_frame_count = 0
        
        # Display FPS
        cv2.putText(img_with_predictions, f"FPS: {self.current_fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        self.frame_count += 1
        
        # Convert to VideoFrame for WebRTC
        new_frame = VideoFrame.from_ndarray(img_with_predictions, format="rgb24")
        new_frame.pts = self.pts_counter
        new_frame.time_base = self.VIDEO_TIME_BASE
        
        # Increment pts by the number of clock ticks per frame
        self.pts_counter += int(self.VIDEO_CLOCK_RATE / self.fps)
        
        # Periodic memory cleanup to prevent memory leaks
        if self.frame_count % 50 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        # Debug output every 30 frames
        if self.frame_count % 30 == 0:
            gpu_mem = ""
            if torch.cuda.is_available():
                gpu_mem_mb = torch.cuda.memory_allocated(0) / 1024**2
                gpu_util = torch.cuda.memory_reserved(0) / 1024**2
                gpu_mem = f", GPU Mem={gpu_mem_mb:.0f}MB/{gpu_util:.0f}MB"
            print(f"Sent frame {self.frame_count}, FPS={self.current_fps:.1f}, detections={len(all_detected_boxes)}{gpu_mem}")
        
        return new_frame

    def stop(self):
        """Release video capture."""
        if self.cap:
            self.cap.release()


# Global variables
pcs = set()
relay = MediaRelay()
video_track = None
model = None


async def index(request):
    """Serve the main HTML page."""
    content = open(os.path.join(os.path.dirname(__file__), "webrtc_client.html"), "r").read()
    return web.Response(content_type="text/html", text=content)


async def offer(request):
    """Handle WebRTC offer from client."""
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        print(f"Connection state: {pc.connectionState}")
        if pc.connectionState == "failed" or pc.connectionState == "closed":
            await pc.close()
            pcs.discard(pc)

    # Set remote description first
    await pc.setRemoteDescription(offer)
    
    # Create a fresh track for this connection using relay
    global video_track, model
    if video_track:
        # Use relay to share the source track
        local_track = relay.subscribe(video_track)
        pc.addTrack(local_track)
        print(f"[Server] Added relayed video track: {local_track}")
        print(f"[Server] Source track: {video_track}, kind: {video_track.kind}")
    else:
        print("[Server] WARNING: No video track available!")

    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    print(f"[Server] Answer created and sent to client")

    return web.Response(
        content_type="application/json",
        text=json.dumps({
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type
        })
    )


async def on_shutdown(app):
    """Cleanup on shutdown."""
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()
    if video_track:
        video_track.stop()


async def run_server(host="0.0.0.0", port=8080):
    """Run the WebRTC server."""
    app = web.Application()
    app.router.add_get("/", index)
    app.router.add_post("/offer", offer)
    app.on_shutdown.append(on_shutdown)
    
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()
    
    print(f"WebRTC server started at http://{host}:{port}")
    print(f"Open this URL in your browser: http://<jetson-ip>:{port}")
    print("Press Ctrl+C to stop")
    
    # Keep running
    try:
        await asyncio.Future()  # Run forever
    except KeyboardInterrupt:
        pass
    finally:
        await runner.cleanup()


def main():
    """Main function."""
    global video_track, model
    
    # Configuration
    USE_FP16 = False  # Default to FP32 for stability; enable for speed if stable
    
    # Check GPU availability
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"\n{'='*60}")
    print(f"🚀 GPU Acceleration Setup")
    print(f"{'='*60}")
    
    if torch.cuda.is_available():
        print(f"✅ CUDA Available: YES")
        print(f"   Device: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA Version: {torch.version.cuda}")
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"   GPU Memory: {total_mem:.1f} GB")
        print(f"   Using device: {device}")
        print(f"   FP16 Mode: {'Enabled (faster)' if USE_FP16 else 'Disabled (more stable)'}")
        
        # Clear any existing GPU memory
        torch.cuda.empty_cache()
        gc.collect()
    else:
        print(f"⚠️  CUDA Available: NO - Using CPU (slower)")
    
    print(f"{'='*60}\n")
    
    # Choose model: prefer TensorRT engine if present
    engine_path = os.path.join('runs', 'segment', 'yolov11n_seg_custom', 'weights', 'best.engine')
    pt_path = os.path.join('runs', 'segment', 'yolov11n_seg_custom', 'weights', 'best.pt')
    if os.path.exists(engine_path):
        model_path = engine_path
        print(f"⚡ Using TensorRT engine: {model_path}")
    else:
        model_path = pt_path
        print(f"Using PyTorch model: {model_path}")

    model = YOLO(model_path)
    
    # Move model to GPU and optimize (.pt only; .engine handles device internally)
    if torch.cuda.is_available() and not model_path.endswith(".engine"):
        model.to(device)
        # Warm up only for .pt models
        print("🔥 Warming up GPU with tiny dummy inference...")
        try:
            dummy_img = np.zeros((320, 320, 3), dtype=np.uint8)
            _ = model(dummy_img, verbose=False, device=device, half=USE_FP16)
            torch.cuda.empty_cache()
            gc.collect()
            print("✅ GPU warmed up and ready!")
        except Exception as e:
            print(f"⚠️  Warning during warm-up: {e}")
            print("   Continuing anyway...")
    
    print(f"✅ Loaded fine-tuned model from: {model_path}")
    
    # Path to the video file
    video_path = os.path.join('testdata', 'rec7-89.mp4')
    
    # Create video track with GPU support
    video_track = YOLOVideoStreamTrack(video_path, model, device=device, use_fp16=USE_FP16)
    print(f"✅ Loaded video from: {video_path}")
    print(f"[Server] Video track created: {video_track}")
    
    if torch.cuda.is_available():
        print(f"\n💡 Tip: Set USE_FP16=True in code for 2x speed (if no memory errors)")
    
    print(f"\n{'='*60}\n")
    
    # Run the server
    asyncio.run(run_server(host="0.0.0.0", port=8080))


if __name__ == "__main__":
    main()

