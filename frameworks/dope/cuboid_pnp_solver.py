# Copyright (c) 2018 NVIDIA Corporation. All rights reserved.
# This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
# https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode

"""
Optimized CuboidPNPSolver for 6D pose estimation.

Optimizations:
- Cached 3D vertices as numpy array (avoid repeated conversions)
- Pre-allocated arrays for 2D/3D point collection
- Replaced pyrr.Quaternion with direct numpy computation
- Use faster PnP algorithms (SQPNP when available)
- Vectorized quaternion conversion
"""

import cv2
import numpy as np
from cuboid import CuboidVertexType


class CuboidPNPSolver(object):
    """
    This class is used to find the 6-DoF pose of a cuboid given its projected vertices.
    Runs perspective-n-point (PNP) algorithm.
    
    Optimized version with:
    - Cached numpy arrays
    - Fast quaternion conversion (no pyrr dependency in hot path)
    - Better PnP algorithm selection
    """

    # Class variables - computed once
    cv2version = cv2.__version__.split(".")
    cv2majorversion = int(cv2version[0])
    cv2minorversion = int(cv2version[1]) if len(cv2version) > 1 else 0
    
    # Check for SQPNP availability (OpenCV 4.5+)
    HAS_SQPNP = cv2majorversion > 4 or (cv2majorversion == 4 and cv2minorversion >= 5)
    
    # Total vertex count cached
    TOTAL_VERTEX_COUNT = int(CuboidVertexType.TotalVertexCount)

    def __init__(
        self,
        object_name="",
        camera_intrinsic_matrix=None,
        cuboid3d=None,
        dist_coeffs=None,
    ):
        self.object_name = object_name
        
        # Camera parameters
        if camera_intrinsic_matrix is not None:
            self._camera_intrinsic_matrix = np.asarray(camera_intrinsic_matrix, dtype=np.float64)
        else:
            self._camera_intrinsic_matrix = np.zeros((3, 3), dtype=np.float64)
        
        if dist_coeffs is not None:
            self._dist_coeffs = np.asarray(dist_coeffs, dtype=np.float64)
        else:
            self._dist_coeffs = np.zeros((4, 1), dtype=np.float64)
        
        # Cache cuboid 3D points as contiguous numpy array
        self._cuboid3d = cuboid3d
        if cuboid3d is not None:
            self._cuboid3d_points = np.ascontiguousarray(
                cuboid3d.get_vertices(), dtype=np.float64
            )
        else:
            self._cuboid3d_points = None
        
        # Pre-allocate arrays for point collection (max 9 points)
        self._obj_2d_buffer = np.zeros((self.TOTAL_VERTEX_COUNT, 2), dtype=np.float64)
        self._obj_3d_buffer = np.zeros((self.TOTAL_VERTEX_COUNT, 3), dtype=np.float64)
        
        # Select best PnP algorithm
        self._default_pnp_algorithm = self._select_best_algorithm()
    
    def _select_best_algorithm(self):
        """Select the best available PnP algorithm."""
        # SQPNP is fastest and most accurate (OpenCV 4.5+)
        if self.HAS_SQPNP:
            return cv2.SOLVEPNP_SQPNP
        # Fall back to EPNP (faster than iterative)
        return cv2.SOLVEPNP_EPNP

    def set_camera_intrinsic_matrix(self, new_intrinsic_matrix):
        """Sets the camera intrinsic matrix."""
        self._camera_intrinsic_matrix = np.asarray(new_intrinsic_matrix, dtype=np.float64)

    def set_dist_coeffs(self, dist_coeffs):
        """Sets the distortion coefficients."""
        self._dist_coeffs = np.asarray(dist_coeffs, dtype=np.float64)

    def solve_pnp(self, cuboid2d_points, pnp_algorithm=None):
        """
        Detects the rotation and translation of a cuboid object 
        from its vertices' 2D location in the image.
        
        Optimized version with pre-allocated buffers and fast quaternion conversion.
        
        Args:
            cuboid2d_points: List of 9 2D points (can contain None for missing points)
            pnp_algorithm: Optional cv2 PnP algorithm flag
            
        Returns:
            tuple: (location, quaternion, projected_points)
        """
        if pnp_algorithm is None:
            pnp_algorithm = self._default_pnp_algorithm

        location = None
        quaternion = None
        projected_points = cuboid2d_points

        # Use cached 3D points
        cuboid3d_points = self._cuboid3d_points
        
        # Collect valid 2D-3D point correspondences into pre-allocated buffers
        valid_count = 0
        for i in range(self.TOTAL_VERTEX_COUNT):
            point_2d = cuboid2d_points[i]
            if point_2d is not None:
                self._obj_2d_buffer[valid_count, 0] = point_2d[0]
                self._obj_2d_buffer[valid_count, 1] = point_2d[1]
                self._obj_3d_buffer[valid_count] = cuboid3d_points[i]
                valid_count += 1

        # Need at least 4 points for PnP
        if valid_count >= 4:
            # Use views of pre-allocated buffers (no allocation)
            obj_2d_points = self._obj_2d_buffer[:valid_count]
            obj_3d_points = self._obj_3d_buffer[:valid_count]
            
            ret, rvec, tvec = cv2.solvePnP(
                obj_3d_points,
                obj_2d_points,
                self._camera_intrinsic_matrix,
                self._dist_coeffs,
                flags=pnp_algorithm,
            )

            if ret:
                # Extract location (flatten tvec)
                location = [tvec[0, 0], tvec[1, 0], tvec[2, 0]]
                
                # Convert rotation vector to quaternion (optimized, no pyrr)
                quaternion = self._rvec_to_quaternion_fast(rvec)
                
                # Project all 3D points
                projected_points, _ = cv2.projectPoints(
                    cuboid3d_points,
                    rvec,
                    tvec,
                    self._camera_intrinsic_matrix,
                    self._dist_coeffs,
                )
                projected_points = np.squeeze(projected_points)

                # Handle case where object is behind camera (Z < 0)
                if location[2] < 0:
                    location = [-location[0], -location[1], -location[2]]
                    # Rotate quaternion by 180 degrees around the location axis
                    quaternion = self._rotate_quaternion_180(quaternion, location)

        return location, quaternion, projected_points

    def _rvec_to_quaternion_fast(self, rvec):
        """
        Convert rotation vector to quaternion using direct numpy computation.
        Much faster than using pyrr library.
        
        Args:
            rvec: Rotation vector (3x1 or 1x3)
            
        Returns:
            Quaternion as pyrr.Quaternion compatible object (has x,y,z,w attributes)
        """
        # Flatten rvec
        rv = rvec.flatten()
        
        # Compute angle (theta)
        theta = np.sqrt(rv[0]*rv[0] + rv[1]*rv[1] + rv[2]*rv[2])
        
        if theta < 1e-10:
            # Near-zero rotation - return identity quaternion
            return _FastQuaternion(0.0, 0.0, 0.0, 1.0)
        
        # Normalize axis
        axis = rv / theta
        
        # Convert axis-angle to quaternion
        half_theta = theta * 0.5
        sin_half = np.sin(half_theta)
        cos_half = np.cos(half_theta)
        
        return _FastQuaternion(
            axis[0] * sin_half,
            axis[1] * sin_half,
            axis[2] * sin_half,
            cos_half
        )
    
    def _rotate_quaternion_180(self, quat, axis):
        """
        Rotate quaternion by 180 degrees around given axis.
        Replaces pyrr Quaternion.from_axis_rotation and cross product.
        
        Args:
            quat: Original quaternion (_FastQuaternion)
            axis: Rotation axis [x, y, z]
            
        Returns:
            New rotated quaternion
        """
        # Normalize axis
        axis = np.array(axis, dtype=np.float64)
        norm = np.sqrt(axis[0]*axis[0] + axis[1]*axis[1] + axis[2]*axis[2])
        if norm < 1e-10:
            return quat
        axis = axis / norm
        
        # 180 degree rotation quaternion (sin(90°)=1, cos(90°)=0)
        rot_q = _FastQuaternion(axis[0], axis[1], axis[2], 0.0)
        
        # Quaternion multiplication (Hamilton product)
        # q1 * q2
        x1, y1, z1, w1 = rot_q.x, rot_q.y, rot_q.z, rot_q.w
        x2, y2, z2, w2 = quat.x, quat.y, quat.z, quat.w
        
        return _FastQuaternion(
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
            w1*w2 - x1*x2 - y1*y2 - z1*z2
        )

    def convert_rvec_to_quaternion(self, rvec):
        """Legacy method - calls optimized version."""
        return self._rvec_to_quaternion_fast(rvec)

    def project_points(self, rvec, tvec):
        """Project points from model onto image using rotation, translation."""
        output_points, _ = cv2.projectPoints(
            self._cuboid3d_points,
            rvec,
            tvec,
            self._camera_intrinsic_matrix,
            self._dist_coeffs,
        )
        return np.squeeze(output_points)


class _FastQuaternion:
    """
    Minimal quaternion class compatible with pyrr.Quaternion interface.
    Used internally for fast quaternion operations without pyrr overhead.
    """
    __slots__ = ('x', 'y', 'z', 'w')
    
    def __init__(self, x, y, z, w):
        self.x = x
        self.y = y
        self.z = z
        self.w = w
    
    def __iter__(self):
        """Allow unpacking as (x, y, z, w)."""
        return iter([self.x, self.y, self.z, self.w])
    
    def __getitem__(self, index):
        """Allow indexing [0]=x, [1]=y, [2]=z, [3]=w."""
        return (self.x, self.y, self.z, self.w)[index]
