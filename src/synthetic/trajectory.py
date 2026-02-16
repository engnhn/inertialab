import numpy as np
from ..math_core.conversions import euler_to_quat

class TrajectoryGenerator:
    def __init__(self, duration=10.0, dt=0.01):
        self.duration = duration
        self.dt = dt
        self.time = np.arange(0, duration, dt)
        
    def generate_sinusoidal(self, freqs, amps, phases=None):
        """
        Generate trajectory based on sinusoidal Euler angles.
        Returns dict with: time, euler, orientation, angular_velocity, acceleration.
        """
        if phases is None: phases = np.zeros(3)
        t = self.time
        n = len(t)
        
        omega_gen = 2 * np.pi * np.array(freqs)
        
        # Euler Angles and Derivatives
        roll = amps[0] * np.sin(omega_gen[0] * t + phases[0])
        pitch = amps[1] * np.sin(omega_gen[1] * t + phases[1])
        yaw = amps[2] * np.sin(omega_gen[2] * t + phases[2])
        euler = np.stack([roll, pitch, yaw], axis=-1)
        
        roll_dot = amps[0] * omega_gen[0] * np.cos(omega_gen[0] * t + phases[0])
        pitch_dot = amps[1] * omega_gen[1] * np.cos(omega_gen[1] * t + phases[1])
        yaw_dot = amps[2] * omega_gen[2] * np.cos(omega_gen[2] * t + phases[2])
        
        # Orientation
        orientation = euler_to_quat(euler)
        
        # Angular Velocity (Body Frame)
        sr, cr = np.sin(roll), np.cos(roll)
        st, ct = np.sin(pitch), np.cos(pitch)
        
        p = roll_dot - st * yaw_dot
        q = cr * pitch_dot + sr * ct * yaw_dot
        r = -sr * pitch_dot + cr * ct * yaw_dot
        
        angular_velocity = np.stack([p, q, r], axis=-1)
        
        # Linear Acceleration (Inertial Frame) - Zero for pure rotation
        acceleration = np.zeros((n, 3))
        
        return {
            "time": t,
            "euler": euler,
            "orientation": orientation,
            "angular_velocity": angular_velocity,
            "acceleration": acceleration
        }
