# WebRTC Streaming Setup for YOLOv11 on Jetson Orin Nano

This guide explains how to set up real-time video streaming from your Jetson Orin Nano to your local laptop using WebRTC.

## Overview

The system consists of:
- **Server (Jetson Orin Nano)**: Runs YOLO inference and streams processed video via WebRTC
- **Client (Your Laptop)**: Web browser that connects to the server and displays the stream

## Prerequisites

### On Jetson Orin Nano

1. **Python 3.8+** installed
2. **PyTorch** for Jetson (if not already installed)
3. **CUDA** enabled and working
4. Network connectivity between Jetson and your laptop

## Installation Steps

### 1. Install Dependencies on Jetson

```bash
cd /workspaces/Documents/Human-Activity-Understanding-Final-Project/YOLOv11-finetuning

# Install WebRTC dependencies
pip3 install -r requirements_webrtc.txt
```

**Note for Jetson**: If you encounter issues with `av` or `aiortc`, you may need to install system dependencies:

```bash
sudo apt-get update
sudo apt-get install -y \
    libavformat-dev \
    libavcodec-dev \
    libavdevice-dev \
    libavutil-dev \
    libswscale-dev \
    libswresample-dev \
    libavfilter-dev \
    libopus-dev \
    libvpx-dev \
    pkg-config
```

### 2. Configure Firewall (if needed)

Make sure port 8080 is accessible:

```bash
# Check if firewall is active
sudo ufw status

# If active, allow port 8080
sudo ufw allow 8080/tcp
```

### 3. Find Your Jetson's IP Address

```bash
hostname -I
# or
ip addr show
```

Note down the IP address (e.g., `192.168.1.100`)

## Running the Stream

### 1. Start the Server on Jetson

```bash
cd /workspaces/Documents/Human-Activity-Understanding-Final-Project/YOLOv11-finetuning

python3 test_yolov11n-seg_webrtc.py
```

You should see output like:
```
Loaded fine-tuned model from: runs/segment/yolov11n_seg_custom/weights/best.pt
Loaded video from: testdata/rec7-89.mp4
Video FPS: 30
WebRTC server started at http://0.0.0.0:8080
Open this URL in your browser: http://<jetson-ip>:8080
Press Ctrl+C to stop
```

### 2. Connect from Your Laptop

On your local laptop:

1. Open a web browser (Chrome, Firefox, or Edge recommended)
2. Navigate to: `http://<jetson-ip>:8080`
   - Replace `<jetson-ip>` with your Jetson's IP address
   - Example: `http://192.168.1.100:8080`
3. Click the **"▶️ Start Stream"** button
4. Wait a few seconds for the connection to establish
5. You should see the live processed video with YOLO detections!

## Usage

### Controls
- **Start Stream**: Initiates the WebRTC connection and starts receiving video
- **Stop Stream**: Closes the connection and stops the video

### Status Indicators
- 🔴 **Red**: Disconnected
- 🟡 **Yellow**: Connecting
- 🟢 **Green**: Connected and streaming

## Troubleshooting

### Problem: Cannot connect to the stream

**Solutions:**
1. Verify both devices are on the same network
2. Check the Jetson's IP address: `hostname -I`
3. Ensure port 8080 is not blocked by firewall
4. Try accessing from Jetson itself first: `http://localhost:8080`

### Problem: Stream is laggy or stuttering

**Solutions:**
1. Check network bandwidth: `iperf3 -s` on Jetson, `iperf3 -c <jetson-ip>` on laptop
2. Reduce video resolution (modify video_path in script to use lower resolution video)
3. Close other applications on Jetson to free up resources
4. Consider using a wired Ethernet connection instead of WiFi

### Problem: "Module not found" errors

**Solutions:**
```bash
# Reinstall dependencies
pip3 install --upgrade -r requirements_webrtc.txt

# For aiortc build issues on Jetson:
sudo apt-get install python3-dev
pip3 install --no-cache-dir aiortc
```

### Problem: Video file not found

**Solution:**
Make sure the video file exists at `testdata/rec7-89.mp4`. You can modify the `video_path` variable in the script to use a different video file.

### Problem: Model not found

**Solution:**
Ensure your fine-tuned model exists at:
```
runs/segment/yolov11n_seg_custom/weights/best.pt
```

## Performance Optimization

### For Better Performance on Jetson:

1. **Use GPU acceleration** (should be automatic with CUDA)
2. **Adjust video FPS**: The script automatically detects video FPS
3. **Use TensorRT** for faster inference:
   ```python
   model = YOLO(model_path)
   model.export(format='engine')  # Export to TensorRT
   model = YOLO('best.engine')    # Load TensorRT model
   ```

### Network Optimization:

1. **Use 5GHz WiFi** instead of 2.4GHz if possible
2. **Wired connection** (Ethernet) is best for stability
3. **Reduce interference**: Keep devices close to the router

## Architecture Details

### Server Components
- **YOLO Model**: Processes each frame for object detection
- **VideoStreamTrack**: Custom WebRTC track that provides processed frames
- **Web Server**: Serves the HTML client and handles WebRTC signaling
- **MediaRelay**: Manages multiple client connections (if needed)

### Client Components
- **HTML/JavaScript**: Single-page application with WebRTC client
- **RTCPeerConnection**: Handles WebRTC connection and video stream

## Customization

### Change Video Source

Edit `test_yolov11n-seg_webrtc.py`:
```python
# Use a different video file
video_path = os.path.join('testdata', 'your-video.mp4')

# Or use a camera (0 for default camera)
# video_path = 0
```

### Change Port

Edit the server script:
```python
asyncio.run(run_server(host="0.0.0.0", port=9000))  # Use port 9000
```

Then access via: `http://<jetson-ip>:9000`

### Adjust Detection Confidence

In the YOLO inference call:
```python
results = self.model(img_rgb, conf=0.5, verbose=False)  # 50% confidence threshold
```

## Security Notes

⚠️ **Important**: This setup is for local network use only. For production:

1. Add authentication
2. Use HTTPS/WSS with SSL certificates
3. Implement proper error handling
4. Add rate limiting
5. Validate all inputs

## Additional Resources

- [aiortc Documentation](https://aiortc.readthedocs.io/)
- [Ultralytics YOLOv11](https://docs.ultralytics.com/)
- [WebRTC Documentation](https://webrtc.org/getting-started/overview)
- [Jetson Orin Nano Developer Guide](https://developer.nvidia.com/embedded/jetson-orin-nano-developer-kit)

## Support

If you encounter issues:
1. Check the console output on Jetson for error messages
2. Check browser console (F12) for JavaScript errors
3. Verify all dependencies are installed correctly
4. Ensure your model and video files are in the correct locations

