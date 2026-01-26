"""
Extract and undistort the first frame from each camera video using intrinsic calibration.
"""

import cv2
import yaml
import numpy as np
from pathlib import Path


def load_calibrations(calib_path: Path) -> dict:
    """Load camera calibrations from YAML file."""
    with open(calib_path, 'r') as f:
        return yaml.safe_load(f)


def undistort_frame(frame: np.ndarray, K: np.ndarray, D: np.ndarray) -> np.ndarray:
    """
    Apply intrinsic calibration to undistort a frame.
    
    Args:
        frame: Input image (BGR)
        K: 3x3 camera intrinsic matrix
        D: Distortion coefficients (k1, k2, p1, p2, k3)
    
    Returns:
        Undistorted frame
    """
    h, w = frame.shape[:2]
    
    # Get optimal new camera matrix
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 1, (w, h))
    
    # Undistort the image
    undistorted = cv2.undistort(frame, K, D, None, new_K)
    
    # Crop to ROI if valid
    x, y, w_roi, h_roi = roi
    if w_roi > 0 and h_roi > 0:
        undistorted = undistorted[y:y+h_roi, x:x+w_roi]
    
    return undistorted


def extract_first_frame(video_path: Path) -> np.ndarray | None:
    """Extract the first frame from a video file."""
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print(f"Error: Could not read frame from {video_path}")
        return None
    
    return frame


def main():
    # Setup paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    recording_dir = project_root / "data" / "recording_3"
    calib_path = project_root / "data" / "cams_calibrations.yml"
    output_dir = script_dir / "calibrated_frames"
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    # Load calibrations
    print(f"Loading calibrations from {calib_path}")
    calibrations = load_calibrations(calib_path)
    
    # Process each camera folder
    camera_folders = sorted([d for d in recording_dir.iterdir() if d.is_dir() and not d.name.startswith('.')])
    
    print(f"Found {len(camera_folders)} camera folders")
    
    for cam_folder in camera_folders:
        cam_serial = cam_folder.name
        
        # Find video file (same name as folder)
        video_path = cam_folder / f"{cam_serial}.mp4"
        
        if not video_path.exists():
            print(f"Warning: Video not found for camera {cam_serial}")
            continue
        
        # Check if calibration exists for this camera
        if cam_serial not in calibrations:
            print(f"Warning: No calibration found for camera {cam_serial}")
            continue
        
        print(f"Processing camera {cam_serial}...")
        
        # Extract first frame
        frame = extract_first_frame(video_path)
        if frame is None:
            continue
        
        # Get intrinsic parameters
        intrinsics = calibrations[cam_serial]['intrinsics']
        K = np.array(intrinsics['K'], dtype=np.float64)
        D = np.array(intrinsics['D'], dtype=np.float64)
        
        # Apply undistortion
        corrected_frame = undistort_frame(frame, K, D)
        
        # Save corrected frame
        output_path = output_dir / f"{cam_serial}_corrected.jpg"
        cv2.imwrite(str(output_path), corrected_frame)
        print(f"  Saved: {output_path}")
    
    print(f"\nDone! Corrected frames saved to {output_dir}")


if __name__ == "__main__":
    main()
