# 3D Scene Understanding Module

Real-time multi-camera 3D scene understanding system for monitoring assembly tasks (e.g., battery installation and screw tightening).

## Overview

This module provides a web-based visualization and monitoring system that fuses data from multiple cameras to:

- **Detect objects** using YOLO (batteries, tools, case components)
- **Estimate 6D poses** of objects using DOPE (Deep Object Pose Estimation)
- **Reconstruct 3D point clouds** using VGGT (multi-view reconstruction)
- **Track assembly sequences** including battery insertion order and screw tightening
- **Detect errors** in real-time using LSTM-based sequence analysis

## Key Components

| File | Description |
|------|-------------|
| `3dscene.py` | Main web server with synchronized multi-camera streaming |
| `yolo_inference.py` | YOLO object detection for batteries and components |
| `dope_inference.py` | DOPE 6D pose estimation for tools and case |
| `vggt_inference.py` | VGGT 3D point cloud reconstruction |
| `lstm_inference.py` | LSTM-based error detection with multi-camera fusion |
| `battery_fsm_module.py` | Finite state machine for battery sequence tracking |
| `screw_sequence_tracker.py` | Tracks screw tightening order (BL → TR → BR → TL) |
| `web_interface.html` | Interactive 3D visualization interface |
| `config/` | Camera calibration and model configuration files |

## Usage

```bash
# Run the server (from project root)
python 3d_scene/3dscene.py
```

Then open `http://localhost:8085` in your browser to access the visualization interface.

## Requirements

- Camera calibration data in `data/cams_calibrations.yml`
- Pre-recorded video data in `data/recording_X/` directories
- Model weights in `weights/` (DOPE, VGGT, YOLO, LSTM)
