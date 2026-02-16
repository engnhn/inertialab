import numpy as np
from .quaternions import q_normalize

def quat_to_euler(q):
    """
    Convert quaternion to Euler angles (Roll, Pitch, Yaw) ZYX convention.
    Returns: [roll, pitch, yaw] in radians
    """
    q = q_normalize(q)
    w, x, y, z = np.moveaxis(q, -1, 0)
    
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = np.clip(2 * (w * y - z * x), -1.0, 1.0)
    pitch = np.arcsin(sinp)
    
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    return np.stack([roll, pitch, yaw], axis=-1)

def euler_to_quat(euler):
    """
    Convert Euler angles (Roll, Pitch, Yaw) to Quaternion [w, x, y, z].
    """
    roll, pitch, yaw = np.moveaxis(euler, -1, 0)
    
    cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
    cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
    cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
    
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    
    return np.stack([w, x, y, z], axis=-1)

def quat_to_rotmat(q):
    """
    Convert quaternion to 3x3 rotation matrix.
    v_global = R * v_local
    """
    q = q_normalize(q)
    w, x, y, z = np.moveaxis(q, -1, 0)
    
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    
    r00 = 1 - 2 * (yy + zz)
    r01 = 2 * (xy - wz)
    r02 = 2 * (xz + wy)
    
    r10 = 2 * (xy + wz)
    r11 = 1 - 2 * (xx + zz)
    r12 = 2 * (yz - wx)
    
    r20 = 2 * (xz - wy)
    r21 = 2 * (yz + wx)
    r22 = 1 - 2 * (xx + yy)
    
    return np.stack([
        np.stack([r00, r01, r02], axis=-1),
        np.stack([r10, r11, r12], axis=-1),
        np.stack([r20, r21, r22], axis=-1)
    ], axis=-2)
