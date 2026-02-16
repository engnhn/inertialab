import numpy as np
from ..math_core.quaternions import q_mult, q_normalize, q_rot_vec
from ..math_core.conversions import quat_to_rotmat

class ES_EKF:
    def __init__(self, 
                 init_q=np.array([1.0, 0.0, 0.0, 0.0]),
                 sigma_acc=0.01,
                 sigma_gyro=0.001,
                 sigma_acc_bias=0.0001,
                 sigma_gyro_bias=0.00001,
                 sigma_mag=0.05):
        
        self.q = init_q
        self.b_w = np.zeros(3)
        
        # Covariance (6x6): [d_theta, d_b_w]
        self.P = np.eye(6) * 0.01 
        
        # Process Noise
        self.Q_c = np.zeros((6, 6))
        self.Q_c[0:3, 0:3] = np.eye(3) * sigma_gyro**2
        self.Q_c[3:6, 3:6] = np.eye(3) * sigma_gyro_bias**2
        
        self.R_acc = np.eye(3) * sigma_acc**2
        self.R_mag = np.eye(3) * sigma_mag**2
        
    def predict(self, gyro, dt):
        """Propagate state and covariance."""
        w_unbiased = gyro - self.b_w
        
        # Nominal State Integration
        w_q = np.array([0, w_unbiased[0], w_unbiased[1], w_unbiased[2]])
        q_dot = 0.5 * q_mult(self.q, w_q)
        self.q = q_normalize(self.q + q_dot * dt)
        
        # Jacobian F_x
        wx, wy, wz = w_unbiased
        W_x = np.array([[0, -wz, wy], [wz, 0, -wx], [-wy, wx, 0]])
        
        F_x = np.eye(6)
        F_x[0:3, 0:3] = np.eye(3) - W_x * dt
        F_x[0:3, 3:6] = -np.eye(3) * dt
        
        # Covariance
        Q_d = self.Q_c * dt
        self.P = F_x @ self.P @ F_x.T + Q_d
        self.P = 0.5 * (self.P + self.P.T)
        
    def update_accel(self, accel):
        """Correction with Accelerometer."""
        a_norm = np.linalg.norm(accel)
        if a_norm == 0: return
        a_meas = accel / a_norm
        
        # Prediction (Gravity [0,0,1] in Body)
        R = quat_to_rotmat(self.q)
        est_gravity = R.T @ np.array([0, 0, 1])
        z = a_meas - est_gravity
        
        # Jacobian H
        gx, gy, gz = est_gravity
        G_skew = np.array([[0, -gz, gy], [gz, 0, -gx], [-gy, gx, 0]])
        
        H = np.zeros((3, 6))
        H[0:3, 0:3] = G_skew
        
        self._kf_update(z, H, self.R_acc)
        
    def update_mag(self, mag):
        """Correction with Magnetometer."""
        m_norm = np.linalg.norm(mag)
        if m_norm == 0: return
        m_meas = mag / m_norm
        
        # Rotate measurement to Nav frame to estimate ref vector
        m_nav = q_rot_vec(self.q, m_meas)
        b_x = np.sqrt(m_nav[0]**2 + m_nav[1]**2)
        b_z = m_nav[2]
        m_ref = np.array([b_x, 0, b_z])
        
        # Prediction
        R = quat_to_rotmat(self.q)
        m_est = R.T @ m_ref
        z = m_meas - m_est
        
        # Jacobian H
        mx, my, mz = m_est
        M_skew = np.array([[0, -mz, my], [mz, 0, -mx], [-my, mx, 0]])
        
        H = np.zeros((3, 6))
        H[0:3, 0:3] = M_skew
        
        self._kf_update(z, H, self.R_mag)

    def _kf_update(self, z, H, R_cov):
        """Generic Kalman Filter Update."""
        S = H @ self.P @ H.T + R_cov
        K = self.P @ H.T @ np.linalg.inv(S)
        dx = K @ z
        
        self._inject_error(dx)
        
        I = np.eye(6)
        self.P = (I - K @ H) @ self.P
        self.P = 0.5 * (self.P + self.P.T)

    def _inject_error(self, dx):
        """Inject error into nominal state."""
        d_theta = dx[0:3]
        d_b_w = dx[3:6]
        
        self.b_w += d_b_w
        
        theta_norm = np.linalg.norm(d_theta)
        if theta_norm > 1e-6:
            axis = d_theta / theta_norm
            angle = theta_norm
            w = np.cos(angle/2)
            xyz = axis * np.sin(angle/2)
            dq = np.array([w, xyz[0], xyz[1], xyz[2]])
        else:
            dq = np.array([1.0, 0.5*d_theta[0], 0.5*d_theta[1], 0.5*d_theta[2]])
            
        self.q = q_mult(self.q, dq)
        self.q = q_normalize(self.q)
    
    def run(self, accels, gyros, mags, dt):
        n = len(accels)
        qs = np.zeros((n, 4))
        biases = np.zeros((n, 3))
        covs = np.zeros((n, 6))
        
        qs[0] = self.q
        
        for i in range(1, n):
            self.predict(gyros[i], dt)
            self.update_accel(accels[i])
            self.update_mag(mags[i])
            
            qs[i] = self.q
            biases[i] = self.b_w
            covs[i] = np.diag(self.P)
            
        return qs, biases, covs
