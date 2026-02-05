# YOLOv11 Fine-tuning for Human Activity Understanding

This project contains fine-tuned YOLOv11 segmentation models for detecting and segmenting custom objects.

## 🚀 Installation

### Prerequisites

- **Anaconda** or **Miniconda** installed on your system
- **Python 3.10+**V
- **GPU support** (optional but recommended):
  - NVIDIA GPU with CUDA for Linux/Windows
  - Apple Silicon (M1/M2/M3) for macOS with MPS support

### Step 1: Create Conda Environment

If you don't have a conda environment yet, create one:

```bash
conda create -n HAUP python=3.10 -y
conda activate HAUP
```

If you're already using the `HAUP` environment, simply activate it:

```bash
conda activate HAUP
```

### Step 2: Install PyTorch

**For macOS (Apple Silicon - M1/M2/M3):**

```bash
conda install pytorch::pytorch torchvision torchaudio -c pytorch -y
```

**For CPU only:**

```bash
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
```

**For NVIDIA GPU (CUDA):**

```bash

# OR CUDA 12.1, use this
conda activate HAUP && pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

```

### Step 3: Install Project Dependencies

Navigate to the yolov11_finetuned directory and install requirements:

```bash
cd yolov11_finetuned
python -m pip install -r requirements.txt
```

### Step 4: Verify Installation

Check that PyTorch is correctly installed with GPU support:

```python
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'MPS available: {torch.backends.mps.is_available()}')"
```

Expected output:

- **NVIDIA GPU**: `CUDA available: True`
- **Apple Silicon**: `MPS available: True`
- **CPU only**: Both will show `False`

**Note:** The model will automatically load `best.pt` when running the test script.

## 🎮 Usage

### Running Video Inference

To run the segmentation model on a test video:

```bash
python test_yolov11n-seg.py
```

**Controls:**

- Press `q` to quit
- Press `p` to pause/resume the video

## 📦 Model Weights

The fine-tuned model weights are located in:

```
runs/segment/yolov11n_seg_custom/weights/
├── best.pt      # Best model weights (use this for inference)
├── best.onnx    # ONNX format
├── best.engine  # TensorRT engine (for Jetson deployment)
└── last.pt      # Last epoch weights
```

### Test Videos

Test videos are located in the `testdata/` directory:

- `rec7-89.mp4`
- `rec7-15.mp4`

To change the test video, modify line 131 in `test_yolov11n-seg.py`:

```python
video_path = os.path.join('testdata', 'your_video.mp4')
```

### Running Fine-tuning (Jupyter Notebook)

To fine-tune the model with your own dataset, it is recommended to open the notebook in **Google Colab** with a **white A100 GPU** for best performance. You can upload and run the notebook directly in Colab by clicking "Open in Colab" or by uploading `finetuning_yolov11m-seg.ipynb` to Colab yourself.

Alternatively, you can run the notebook locally with:

```bash
jupyter notebook finetuning_yolov11m-seg.ipynb
```

## 📁 File Structure

```
yolov11_finetuned/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies (includes WebRTC)
├── test_yolov11n-seg.py               # Main inference script
├── test_yolov11n-seg_webrtc.py        # WebRTC streaming version
├── export_to_engine.py                # TensorRT export utility
├── finetuning_yolov11m-seg.ipynb      # Fine-tuning notebook
├── test_finetunned_yolov11m-seg.ipynb # Testing notebook
├── webrtc_client.html                 # WebRTC client interface
├── WEBRTC_SETUP.md                    # WebRTC setup guide
├── QUICKSTART_WEBRTC.md               # WebRTC quick start
├── testdata/                          # Test videos
│   ├── rec7-89.mp4
│   └── rec7-15.mp4
├── labeled data example/              # Example labeled data
│   ├── rec11-89.mp4
│   └── rec4-89.mp4
└── runs/                              # Training outputs
    └── segment/
        └── yolov11n_seg_custom/
            ├── weights/
            │   ├── best.pt            # 🎯 Use this for inference
            │   └── ...
            └── ...
```
