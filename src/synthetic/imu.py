import numpy as np
from ..math_core.quaternions import q_rot_vec, q_inv

class IMU:
    def __init__(self, 
                 accel_noise_std=0.01, 
                 gyro_noise_std=0.001, 
                 mag_noise_std=0.05,
                 accel_bias_instability=0.0001,
                 gyro_bias_instability=0.00001,
                 gravity=np.array([0, 0, -9.81]), 
                 magnetic_field=np.array([1.0, 0.0, 0.0]),
                 mag_hard_iron_std=0.0): # Standard deviation for random hard-iron bias generation
        self.acc_noise = accel_noise_std
        self.gyro_noise = gyro_noise_std
        self.mag_noise = mag_noise_std
        self.acc_bias_instability = accel_bias_instability
        self.gyro_bias_instability = gyro_bias_instability
        
        self.g = gravity
        self.mag_field = magnetic_field
        
        # Initial Turn-on Bias
        self.acc_bias_turn_on = np.random.uniform(-0.1, 0.1, 3)
        self.gyro_bias_turn_on = np.random.uniform(-0.01, 0.01, 3)
        
        # Hard-Iron Bias (Constant in Body Frame)
        self.mag_hard_iron = np.random.uniform(-mag_hard_iron_std, mag_hard_iron_std, 3) if mag_hard_iron_std > 0 else np.zeros(3)
        
        self.dt = 0.01

    def generate(self, trajectory):
        """
        Generate IMU readings with noise and bias instability.
        Returns: accel, gyro, mag, bias_ground_truth
        """
        q_nb = trajectory['orientation'] 
        time = trajectory['time']
        if len(time) > 1: self.dt = time[1] - time[0]
        n = len(q_nb)
        
        # Random Walk Bias
        acc_bias = self.acc_bias_turn_on + np.cumsum(np.random.normal(0, self.acc_bias_instability, (n, 3)), axis=0)
        gyro_bias = self.gyro_bias_turn_on + np.cumsum(np.random.normal(0, self.gyro_bias_instability, (n, 3)), axis=0)
        
        # Ideal Kinematics
        a_n = trajectory['acceleration']
        f_n = a_n - self.g[np.newaxis, :] # Specific Force: f = a - g
        q_bn = q_inv(q_nb)
        
        accel_clean = q_rot_vec(q_bn, f_n)
        gyro_clean = trajectory['angular_velocity']
        mag_clean = q_rot_vec(q_bn, np.broadcast_to(self.mag_field, (n, 3)))
        
        # Measurements
        accel = accel_clean + acc_bias + np.random.normal(0, self.acc_noise, (n, 3))
        gyro = gyro_clean + gyro_bias + np.random.normal(0, self.gyro_noise, (n, 3))
        
        # Magnetometer with Hard-Iron Bias
        mag = mag_clean + self.mag_hard_iron + np.random.normal(0, self.mag_noise, (n, 3))
        
        return accel, gyro, mag, {'accel': acc_bias, 'gyro': gyro_bias, 'mag_hard_iron': self.mag_hard_iron}
