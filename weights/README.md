## 📥 Downloading Model Weights from Google Drive

The required model files (`dope_tool.pth` and `dope_case.pth`) are 192MB each and need to be downloaded from Google Drive.

### Method 1: Manual Download (Recommended)

Simply click these links and download through your browser:

- **dope_tool.pth** (192MB): [Download from Google Drive](https://drive.google.com/file/d/1HMy5vxv6p16uUAizrhmbJ9WUcKdfwIT-/view?usp=sharing)
- **dope_case.pth** (192MB): [Download from Google Drive](https://drive.google.com/file/d/1JG4Q1yHxV2wXCAzWjFcfA8et9e-ZU6LY/view?usp=sharing)

After downloading, place both files in the `weights/` directory.

### Method 2: Using gdown (Command-line)

For automated downloads, use `gdown` (a Python tool designed for Google Drive):

```bash
# Install gdown
pip install gdown

# Navigate to weights directory
cd weights

# Download dope_tool.pth
gdown 1HMy5vxv6p16uUAizrhmbJ9WUcKdfwIT-

# Download dope_case.pth
gdown 1JG4Q1yHxV2wXCAzWjFcfA8et9e-ZU6LY

# Download vggt.pt
gdown 1VWe6BaP8IcKT45JRxjEb3XZJnIACd1BC
```

**Note**: Regular `curl` or `wget` commands don't work well with large Google Drive files due to virus scan warnings. Use `gdown` or download manually.
