# Quick Start Guide - WebRTC Streaming

## TL;DR

Stream YOLO inference results from your Jetson Orin Nano to your laptop browser in 3 steps:

### Step 1: Install Dependencies (on Jetson)

```bash
cd /workspaces/Documents/Human-Activity-Understanding-Final-Project/yolov11_finetuned

# IMPORTANT: Fix NumPy compatibility first (required for Jetson)
pip3 install numpy==1.26.4 --force-reinstall --index-url https://pypi.org/simple/

# Then install other dependencies
pip3 install -r requirements_webrtc.txt
```

### Step 2: Enable Maximum Performance (Optional but Recommended)

```bash
# Set Jetson to maximum performance mode
sudo nvpmodel -m 0
sudo jetson_clocks
```

### Step 3: Start the Server (on Jetson)

```bash
python3 test_yolov11n-seg_webrtc.py
```

**You should see:**

- ✅ CUDA Available: YES
- 🔥 GPU warming up
- Expected FPS: 25-35 (with GPU acceleration)

### Step 4: Open Browser (on your laptop)

```
http://<jetson-ip>:8080
```

Click "▶️ Start Stream" and enjoy!

---

## Get Jetson IP Address

```bash
hostname -I
```

## Common Issues

**Can't install dependencies?**

```bash
sudo apt-get install -y libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev libswscale-dev
pip3 install --no-cache-dir -r requirements_webrtc.txt
```

**Port blocked?**

```bash
sudo ufw allow 8080/tcp
```

**Different video?** Edit line 174 in `test_yolov11n-seg_webrtc.py`:

```python
video_path = os.path.join('testdata', 'your-video.mp4')
```

For detailed instructions, see [WEBRTC_SETUP.md](WEBRTC_SETUP.md)
