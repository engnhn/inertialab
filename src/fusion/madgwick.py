import numpy as np
from ..math_core.quaternions import q_mult, q_norm, q_normalize, q_conj, q_rot_vec

class Madgwick:
    def __init__(self, beta=0.1):
        self.beta = beta
        self.q_est = np.array([1.0, 0.0, 0.0, 0.0])
        
    def update(self, accel, gyro, mag, dt):
        """Madgwick AHRS Update with Gradient Descent."""
        q = self.q_est
        
        # Gyro update (Prediction)
        w_q = np.array([0, gyro[0], gyro[1], gyro[2]])
        q_dot = 0.5 * q_mult(q, w_q) # Initial q_dot from gyro
        
        # Normalize measurements
        a_norm = np.linalg.norm(accel)
        m_norm = np.linalg.norm(mag)
        
        # If no accel/mag, just integrate gyro (Dead Reckoning)
        if a_norm == 0 or m_norm == 0:
            self.q_est = q_normalize(q + q_dot * dt)
            return self.q_est
            
        a = accel / a_norm
        m = mag / m_norm
        
        # --- Gradient Descent (Gravity) ---
        qw, qx, qy, qz = q
        
        # Estimated Gravity (Body Frame), Ref=[0,0,1]
        half_est_g = np.array([qx*qz - qw*qy, qy*qz + qw*qx, 0.5 - qx**2 - qy**2])
        est_g = 2.0 * half_est_g
        f_g = est_g - a
        
        # Jacobian Gravity
        J_g = np.array([
            [-2*qy,  2*qz, -2*qw,  2*qx],
            [ 2*qx,  2*qw,  2*qz,  2*qy],
            [    0, -4*qx, -4*qy,     0]
        ])
        grad_g = J_g.T @ f_g
        
        # --- Gradient Descent (Magnetometer) ---
        m_rot = q_rot_vec(q, m)
        b_x = np.sqrt(m_rot[0]**2 + m_rot[1]**2)
        b_z = m_rot[2]
        
        est_m_x = 2*b_x*(0.5 - qy**2 - qz**2) + 2*b_z*(qx*qz - qw*qy)
        est_m_y = 2*b_x*(qx*qy - qw*qz)       + 2*b_z*(qy*qz + qw*qx)
        est_m_z = 2*b_x*(qw*qy + qx*qz)       + 2*b_z*(0.5 - qx**2 - qy**2)
        
        f_m = np.array([est_m_x, est_m_y, est_m_z]) - m
        
        # Jacobian Magnetometer
        J_row0 = np.array([
            [    0,    0, -4*qy, -4*qz],
            [ -2*qz, 2*qy,  2*qx, -2*qw],
            [  2*qy, 2*qz,  2*qw,  2*qx]
        ])
        J_row2 = J_g 
        
        J_m = b_x * J_row0 + b_z * J_row2
        grad_m = J_m.T @ f_m
        
        
        # Combined Gradient
        grad = grad_g + grad_m
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 0: grad /= grad_norm
        
        # Apply Correction
        # q_dot_estimated = q_dot_gyro - beta * grad
        q_dot -= self.beta * grad
        self.q_est = q_normalize(q + q_dot * dt)
        
        return self.q_est

    def run(self, accels, gyros, mags, dt):
        n = len(accels)
        qs = np.zeros((n, 4))
        qs[0] = self.q_est
        for i in range(1, n):
            qs[i] = self.update(accels[i], gyros[i], mags[i], dt)
        return qs
