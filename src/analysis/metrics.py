import numpy as np

def angle_difference(a, b):
    """
    Compute smallest difference between two angles in radians.
    diff = (a - b + pi) % (2*pi) - pi
    Range: [-pi, pi]
    """
    diff = (a - b + np.pi) % (2 * np.pi) - np.pi
    return diff

def compute_rmse(estimated, ground_truth):
    """
    Compute Root Mean Square Error between estimated and ground truth.
    Supports both Euler angles (N, 3) and general vectors.
    Handles angle wrapping for Euler angles.
    
    Args:
        estimated: (N, D) array
        ground_truth: (N, D) array
        
    Returns:
        rmse: (D,) array of RMSE errors per dimension
    """
    # Check if we should use angle difference (heuristic: if max val <= pi)
    # But usually passed explicitly. Let's assume input IS Euler angles for this function.
    # Or just generic deviation.
    # For robust orientation error, 2*arccos(|q_est . q_true|) is better for quats.
    
    # Let's do Euler RMSE with wrapping.
    err = angle_difference(estimated, ground_truth)
    return np.sqrt(np.mean(err**2, axis=0))

def compute_quaternion_error(q_est, q_gt):
    """
    Compute orientation error in degrees using quaternion dot product.
    error = 2 * arccos( |<q1, q2>| )
    
    Args:
        q_est: (N, 4)
        q_gt: (N, 4)
        
    Returns:
        mean_error_deg: scalar
        max_error_deg: scalar
    """
    # Dot product
    dot = np.sum(q_est * q_gt, axis=1)
    # Clamp for numerical stability
    dot = np.clip(dot, -1.0, 1.0)
    # Angle
    angle = 2 * np.arccos(np.abs(dot))
    angle_deg = np.degrees(angle)
    
    return np.mean(angle_deg), np.max(angle_deg)
