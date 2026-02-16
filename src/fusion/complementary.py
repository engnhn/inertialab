import numpy as np
from ..math_core.quaternions import q_mult, q_normalize, integrate_quat, q_rot_vec

class ComplementaryFilter:
    def __init__(self, alpha=0.98):
        self.alpha = alpha
        self.q_est = np.array([1.0, 0.0, 0.0, 0.0])
        
    def update(self, accel, gyro, mag, dt):
        """Fusion update step."""
        # 1. Prediction (Gyro Integration)
        q_pred = integrate_quat(self.q_est, gyro, dt)
        
        # 2. Correction (Accel/Mag)
        # Simplified: Alignment of Gravity and Mag vectors
        a_norm = np.linalg.norm(accel)
        m_norm = np.linalg.norm(mag)
        
        if a_norm == 0 or m_norm == 0:
            self.q_est = q_pred
            return self.q_est
            
        accel = accel / a_norm
        mag = mag / m_norm
        
        # Tilt correction: Align accel to [0, 0, 1] (Global Up)
        q_tilt = self._q_from_vectors(accel, np.array([0, 0, 1]))
        
        # Yaw correction
        # Project mag to horizontal plane
        m_h = q_rot_vec(q_tilt, mag)
        yaw_angle = np.arctan2(m_h[1], m_h[0])
        
        # Yaw correction quaternion (negate yaw to align with X)
        sa, ca = np.sin(-yaw_angle/2), np.cos(-yaw_angle/2)
        q_yaw = np.array([ca, 0, 0, sa])
        
        q_obs = q_mult(q_yaw, q_tilt)
        
        # 3. Interpolation
        if np.dot(q_pred, q_obs) < 0:
            q_obs = -q_obs
            
        self.q_est = q_normalize(self.alpha * q_pred + (1.0 - self.alpha) * q_obs)
        return self.q_est
        
    def _q_from_vectors(self, v1, v2):
        """Compute quaternion rotating v1 to v2."""
        cross = np.cross(v1, v2)
        dot = np.dot(v1, v2)
        s = np.sqrt((1 + dot) * 2)
        
        if s < 1e-6: return np.array([0.0, 1.0, 0.0, 0.0])
        
        w = s * 0.5
        xyz = cross / s
        return np.array([w, xyz[0], xyz[1], xyz[2]])

    def run(self, accels, gyros, mags, dt):
        n = len(accels)
        qs = np.zeros((n, 4))
        qs[0] = self.q_est
        for i in range(1, n):
            qs[i] = self.update(accels[i], gyros[i], mags[i], dt)
        return qs
