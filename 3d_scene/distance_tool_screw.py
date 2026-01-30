"""
Distance Tool Screw - Reads the distance from tool tip to case from the 3D scene backend.

This module provides functions to get the current distance between the e-screwdriver
tool tip and the case object, as measured by the 3D scene visualization.

Usage:
    from distance_tool_screw import get_distance, get_distance_sync
    
    # Async usage
    distance_cm = await get_distance()
    
    # Sync usage (blocking)
    distance_cm = get_distance_sync()
"""

import requests
from typing import Optional

# Backend server URL
BACKEND_URL = "http://localhost:8085"


def get_distance_sync(timeout: float = 1.0) -> Optional[float]:
    """Get the current distance from tool tip to case (synchronous/blocking).
    
    Args:
        timeout: Request timeout in seconds
        
    Returns:
        Distance in centimeters, or None if request fails
    """
    try:
        response = requests.get(
            f"{BACKEND_URL}/api/distance/tool-case",
            timeout=timeout
        )
        if response.status_code == 200:
            data = response.json()
            return data.get("distance_cm", 0.0)
        return None
    except Exception:
        return None


async def get_distance(timeout: float = 1.0) -> Optional[float]:
    """Get the current distance from tool tip to case (async).
    
    Args:
        timeout: Request timeout in seconds
        
    Returns:
        Distance in centimeters, or None if request fails
    """
    import aiohttp
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{BACKEND_URL}/api/distance/tool-case",
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("distance_cm", 0.0)
                return None
    except Exception:
        return None


def get_distance_mm_sync(timeout: float = 1.0) -> Optional[float]:
    """Get the current distance from tool tip to case in millimeters (synchronous).
    
    Args:
        timeout: Request timeout in seconds
        
    Returns:
        Distance in millimeters, or None if request fails
    """
    distance_cm = get_distance_sync(timeout)
    if distance_cm is not None:
        return distance_cm * 10.0
    return None


async def get_distance_mm(timeout: float = 1.0) -> Optional[float]:
    """Get the current distance from tool tip to case in millimeters (async).
    
    Args:
        timeout: Request timeout in seconds
        
    Returns:
        Distance in millimeters, or None if request fails
    """
    distance_cm = await get_distance(timeout)
    if distance_cm is not None:
        return distance_cm * 10.0
    return None


# Example usage / test
if __name__ == "__main__":
    import time
    
    print("Distance Tool Screw - Reading distance from 3D scene backend")
    print(f"Backend URL: {BACKEND_URL}")
    print("-" * 50)
    
    # Continuous polling example
    try:
        while True:
            distance_mm = get_distance_sync()
            if distance_mm is not None:
                print(f"Distance: {distance_mm:.1f} mm")
            else:
                print("Failed to get distance (backend not running?)")
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopped.")
