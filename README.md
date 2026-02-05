# Human Activity Understanding - Screw Assembly Tracking

![GitHub Repo Banner](project_image.png)

3D scene visualization system that tracks a Batteries Screw assembly task using multi-camera recordings, DOPE pose estimation, YOLOv11 segmentation and VGGT for 3D Scene reconstruction

## Demo

![x5Demo](x5demo.gif)

## Installation

### 1. Create Conda Environment

```bash
conda create -n HAUP python=3.10 -y
conda activate HAUP
```

### 2. Install PyTorch

**For macOS (Apple Silicon - M1/M2/M3):**

```bash
conda install pytorch::pytorch torchvision torchaudio -c pytorch -y
```

**For NVIDIA GPU (CUDA 12.1):**

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Run

```bash
python 3d_scene/3dscene.py
```

Open your browser at **http://localhost:8085**

## 📁 Project Structure

```
├── 3d_scene/                    # Main application
│   ├── 3dscene.py              # Backend server (aiohttp)
│   ├── web_interface.html      # 3D visualization frontend (Three.js)
│   ├── screw_sequence_tracker.py   # Screw sequence state machine
│   ├── sequence_from_distance_tool.py  # CLI monitoring tool
│   ├── distance_tool_screw.py  # Distance API client
│   ├── dope_inference.py       # DOPE 6D pose estimation
│   ├── yolo_inference.py       # YOLOv11 segmentation
│   ├── vggt_inference.py       # 3D point cloud reconstruction
│   ├── battery_fsm_module.py   # Battery tracking state machine (YOLO-based)
│   └── config/                 # Camera calibrations & DOPE config
│
├── data/
│   ├── recording_1-12/         # Multi-camera recordings (8 cameras each)
│   ├── scanned_objects/        # 3D models (case, e-screwdriver)
│   └── cams_calibrations.yml   # Camera calibration data
│
├── weights/                    # Model weights
│   ├── dope_tool.pth          # DOPE weights for screwdriver
│   ├── dope_case.pth          # DOPE weights for case
│   └── model.pt               # YOLOv11 finetuned weights
│
├── frameworks/                 # External frameworks
│   ├── dope/                  # DOPE implementation
│   └── vggt/                  # VGGT point cloud
│
└── yolov11_finetuned/         # YOLOv11 training & testing
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This project uses the following research works:

- **DOPE (Deep Object Pose Estimation)** - 6D pose estimation for object detection
  https://github.com/NVlabs/Deep_Object_Pose

- **VGGT (Visual Geometry Grounded Transformer)** - 3D scene reconstruction
  https://vgg-t.github.io/

- **YOLO (You Only Look Once, by Ultralytics)** - state-of-the-art real-time object detection
  https://github.com/ultralytics/ultralytics

---

**Course:** Practical Course - Human Activity Understanding  
**Institution:** Technical University of Munich (TUM)
