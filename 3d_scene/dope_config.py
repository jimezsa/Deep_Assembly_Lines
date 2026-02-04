#!/usr/bin/env python3
"""
Quick configuration switcher for DOPE performance vs accuracy.

Usage:
    python 3d_scene/dope_config.py speed    # Fast mode for testing
    python 3d_scene/dope_config.py accurate # Accurate mode for production
    python 3d_scene/dope_config.py ultra    # Ultra-fast mode (minimal accuracy)
"""

import sys
import re

def update_config(mode):
    """Update 3dscene.py with appropriate DOPE settings."""
    
    configs = {
        "speed": {
            "DOPE_USE_FP16": "True",
            "DOPE_STOP_AT_STAGE": "1",
            "dope_inference_interval": "4",
            "yolo_inference_interval": "10",
            "name": "Speed Mode (1 fps target, FP16, stage 1, less accurate)"
        },
        "accurate": {
            "DOPE_USE_FP16": "False",
            "DOPE_STOP_AT_STAGE": "4",
            "dope_inference_interval": "2",
            "yolo_inference_interval": "6",
            "name": "Accurate Mode (slower, FP32, stage 4, more accurate)"
        },
        "ultra": {
            "DOPE_USE_FP16": "True",
            "DOPE_STOP_AT_STAGE": "1",
            "dope_inference_interval": "6",
            "yolo_inference_interval": "15",
            "name": "Ultra-Fast Mode (minimal accuracy, for stress testing)"
        }
    }
    
    if mode not in configs:
        print(f"Unknown mode: {mode}")
        print(f"Available modes: {', '.join(configs.keys())}")
        sys.exit(1)
    
    config = configs[mode]
    
    # Read the file
    with open("3d_scene/3dscene.py", "r") as f:
        content = f.read()
    
    # Update DOPE_USE_FP16
    content = re.sub(
        r"DOPE_USE_FP16 = (True|False)",
        f"DOPE_USE_FP16 = {config['DOPE_USE_FP16']}",
        content
    )
    
    # Update DOPE_STOP_AT_STAGE
    content = re.sub(
        r"DOPE_STOP_AT_STAGE = \d+",
        f"DOPE_STOP_AT_STAGE = {config['DOPE_STOP_AT_STAGE']}",
        content
    )
    
    # Update dope_inference_interval (first occurrence in SyncedVideoManager)
    content = re.sub(
        r"self\.dope_inference_interval = \d+  # Run DOPE every \d+ frames",
        f"self.dope_inference_interval = {config['dope_inference_interval']}  # Run DOPE every {config['dope_inference_interval']} frames",
        content
    )
    
    # Update yolo_inference_interval
    content = re.sub(
        r"self\.yolo_inference_interval = \d+  # Run YOLO every \d+ frames",
        f"self.yolo_inference_interval = {config['yolo_inference_interval']}  # Run YOLO every {config['yolo_inference_interval']} frames",
        content
    )
    
    # Write the file back
    with open("3d_scene/3dscene.py", "w") as f:
        f.write(content)
    
    print(f"\n✓ Configuration updated to: {config['name']}")
    print(f"\nSettings:")
    print(f"  DOPE_USE_FP16: {config['DOPE_USE_FP16']}")
    print(f"  DOPE_STOP_AT_STAGE: {config['DOPE_STOP_AT_STAGE']}")
    print(f"  DOPE inference interval: Every {config['dope_inference_interval']} frames")
    print(f"  YOLO inference interval: Every {config['yolo_inference_interval']} frames")
    print(f"\nRestart the server for changes to take effect:")
    print(f"  python 3d_scene/3dscene.py\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python dope_config.py [speed|accurate|ultra]")
        print("\nModes:")
        print("  speed    - Fast mode for testing (target ~1-2 fps)")
        print("  accurate - Accurate mode (slower, ~0.5 fps)")
        print("  ultra    - Ultra-fast mode (very minimal accuracy)")
        sys.exit(1)
    
    mode = sys.argv[1]
    update_config(mode)
