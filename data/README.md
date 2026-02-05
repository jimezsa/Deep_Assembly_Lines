# Data Directory

This folder contains camera calibration data, multi-camera video recordings, 3D object models, and pre-generated scene reconstructions.

## Contents

### Camera Calibrations

- `cams_calibrations.yml` — Calibration parameters for all 8 cameras including:
  - **Extrinsics**: 4x4 transformation matrices (camera-to-world)
  - **Intrinsics**: Camera matrix (K) and distortion coefficients (D)
  - **Checkerboard**: Transform to checkerboard origin (for master camera)

### Video Recordings

Synchronized multi-camera recordings of assembly tasks:

| Folder | Description |
|--------|-------------|
| `recording_3/` | Assembly session 3 |
| `recording_7/` | Assembly session 7 |
| `recording_8/` | Assembly session 8 |
| `recording_10/` | Assembly session 10 |

Each recording contains 8 synchronized camera feeds named by camera serial number (e.g., `135122071615.mp4`).

### 3D Object Models

`scanned_objects/` — 3D scanned models used for DOPE 6D pose estimation:

- `case/` — Battery case model (`.obj`, `.mtl`, textures)
- `e-screw-driver/` — Electric screwdriver model (`.obj`, `.mtl`, textures)

### VGGT Demo

`VGGT_demo/` — Pre-generated 3D scene reconstructions (`.glb` files) from VGGT multi-view inference, useful for visualization and testing.

## Camera IDs

The 8 cameras used in recordings:

| Camera ID | Notes |
|-----------|-------|
| 135122071615 | Master camera (checkerboard reference) |
| 137322071489 | Primary YOLO detection camera |
| 138422075916 | — |
| 141722071426 | — |
| 141722073953 | — |
| 141722075184 | — |
| 141722079467 | — |
| 142122070087 | DOPE tool detection camera |
