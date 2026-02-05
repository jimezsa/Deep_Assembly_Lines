# Human Activity Understanding - Screw Assembly Tracking

3D scene visualization system that tracks a Batteries Screw assembly task using multi-camera recordings, DOPE pose estimation,YOLOv11 segmentation, and VGGT for 3D Scene reconstruction

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

---

**Course:** Practical Course - Human Activity Understanding
**Institution:** Technical University of Munich (TUM)
