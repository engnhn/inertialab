import numpy as np

def q_mult(q1, q2):
    """
    Multiply two quaternions (or arrays of quaternions).
    Format: [w, x, y, z]
    """
    w1, x1, y1, z1 = np.moveaxis(q1, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(q2, -1, 0)
    
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    
    return np.stack([w, x, y, z], axis=-1)

def q_conj(q):
    """Compute conjugate of quaternion(s) [w, -x, -y, -z]."""
    return q * np.array([1, -1, -1, -1])

def q_norm(q):
    """Compute L2 norm of quaternion(s)."""
    return np.linalg.norm(q, axis=-1, keepdims=True)

def q_normalize(q):
    """Normalize quaternion(s) to unit length."""
    n = q_norm(q)
    n[n < 1e-8] = 1.0
    return q / n

def q_inv(q):
    """Compute inverse quaternion."""
    return q_conj(q) / (q_norm(q)**2)

def q_rot_vec(q, v):
    """
    Rotate vector(s) v by quaternion(s) q.
    v_rot = q * v * q_conj
    """
    v_q = np.zeros(v.shape[:-1] + (4,))
    v_q[..., 1:] = v
    
    rotated_q = q_mult(q_mult(q, v_q), q_conj(q))
    return rotated_q[..., 1:]

def integrate_quat(q, omega, dt):
    """
    Integrate quaternion given angular velocity.
    dq/dt = 0.5 * q * omega
    """
    omega_q = np.zeros(omega.shape[:-1] + (4,))
    omega_q[..., 1:] = omega
    
    q_dot = 0.5 * q_mult(q, omega_q)
    return q_normalize(q + q_dot * dt)
