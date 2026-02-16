import numpy as np
import sys
import os

# Ensure src is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from src.math_core.conversions import quat_to_euler
    from src.synthetic.trajectory import TrajectoryGenerator
    from src.synthetic.imu import IMU
    from src.fusion.complementary import ComplementaryFilter
    from src.fusion.madgwick import Madgwick
    from src.fusion.es_ekf import ES_EKF
    from src.analysis.metrics import compute_rmse, compute_quaternion_error
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

def main():
    print("InertiaLab: Sensor Fusion Framework")
    print("===================================")
    
    # Configuration
    dt = 0.01
    duration = 30.0
    
    # Trajectory Generation
    print(f"Generating {duration}s of synthetic motion...")
    traj_gen = TrajectoryGenerator(duration=duration, dt=dt)
    
    # Complex sinusoidal motion
    freqs = [0.2, 0.1, 0.05] 
    amps = [np.radians(30), np.radians(20), np.radians(90)]
    phases = [0, np.pi/4, 0]
    
    traj_data = traj_gen.generate_sinusoidal(freqs, amps, phases)
    time = traj_data['time']
    euler_gt = traj_data['euler']
    q_gt = traj_data['orientation']
    
    # IMU Simulation
    print("Simulating IMU with Bias Instability...")
    imu = IMU(
        accel_noise_std=0.02, 
        gyro_noise_std=0.002, 
        mag_noise_std=0.05,
        accel_bias_instability=0.0005,
        gyro_bias_instability=0.00005
    )
    accels, gyros, mags, true_biases = imu.generate(traj_data)
    print(f"Simulated Gyro Bias (Mean): {np.mean(true_biases['gyro'], axis=0)}")
    
    # Sensor Fusion
    print("\nRunning Fusion Algorithms...")
    
    # 1. Complementary Filter
    comp_filter = ComplementaryFilter(alpha=0.98)
    q_comp = comp_filter.run(accels, gyros, mags, dt)
    
    # 2. Madgwick Filter
    madgwick = Madgwick(beta=0.1)
    q_madg = madgwick.run(accels, gyros, mags, dt)
    
    # 3. Error-State Kalman Filter (ES-EKF)
    es_ekf = ES_EKF(
        sigma_acc=0.02,
        sigma_gyro=0.002,
        sigma_mag=0.05,
        sigma_gyro_bias=0.00005
    )
    q_ekf, est_biases, covs = es_ekf.run(accels, gyros, mags, dt)
    
    # Metrics
    euler_comp = quat_to_euler(q_comp)
    euler_madg = quat_to_euler(q_madg)
    euler_ekf = quat_to_euler(q_ekf)
    print("\n--- Performance Analysis ---")
    
    # RMSE
    rmse_comp = compute_rmse(euler_comp, euler_gt)
    rmse_madg = compute_rmse(euler_madg, euler_gt)
    rmse_ekf = compute_rmse(euler_ekf, euler_gt)
    
    print(f"Complementary RMSE (deg): Roll={np.degrees(rmse_comp[0]):.2f}, Pitch={np.degrees(rmse_comp[1]):.2f}, Yaw={np.degrees(rmse_comp[2]):.2f}")
    print(f"Madgwick RMSE (deg):      Roll={np.degrees(rmse_madg[0]):.2f}, Pitch={np.degrees(rmse_madg[1]):.2f}, Yaw={np.degrees(rmse_madg[2]):.2f}")
    print(f"ES-EKF RMSE (deg):        Roll={np.degrees(rmse_ekf[0]):.2f}, Pitch={np.degrees(rmse_ekf[1]):.2f}, Yaw={np.degrees(rmse_ekf[2]):.2f}")
    
    # Mean Orientation Error
    err_comp_mean, _ = compute_quaternion_error(q_comp, q_gt)
    err_madg_mean, _ = compute_quaternion_error(q_madg, q_gt)
    err_ekf_mean, _ = compute_quaternion_error(q_ekf, q_gt)
    
    print(f"\nMean Orientation Error (deg):")
    print(f"Complementary: {err_comp_mean:.2f}")
    print(f"Madgwick:      {err_madg_mean:.2f}")
    print(f"ES-EKF:        {err_ekf_mean:.2f}")
    
    # Visualization
    plot_results(time, euler_gt, euler_comp, euler_madg, euler_ekf, true_biases, est_biases, covs)

def plot_results(time, euler_gt, euler_comp, euler_madg, euler_ekf, true_biases, est_biases, covs):
    try:
        import matplotlib.pyplot as plt
        from src.analysis.plotting import plot_euler_comparison
        
        print("\nGenerating plots...")
        plot_euler_comparison(time, euler_gt, euler_comp, title="Complementary Filter")
        plot_euler_comparison(time, euler_gt, euler_madg, title="Madgwick Filter")
        plot_euler_comparison(time, euler_gt, euler_ekf, title="ES-EKF (Bias Estimation)")
        
        # Bias Tracking Plot
        fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        labels = ['X', 'Y', 'Z']
        for i in range(3):
            ax[i].plot(time, true_biases['gyro'][:, i], 'k--', label='True Bias')
            ax[i].plot(time, est_biases[:, i], 'b-', label='Estimated')
            
            # 3-sigma bounds
            sigma = np.sqrt(covs[:, 3+i])
            ax[i].fill_between(time, est_biases[:, i]-3*sigma, est_biases[:, i]+3*sigma, color='b', alpha=0.2, label='3-sigma')
            
            ax[i].set_ylabel(f'Bias {labels[i]} (rad/s)')
            ax[i].grid(True)
            if i==0: ax[i].legend()
            
        ax[2].set_xlabel('Time (s)')
        ax[0].set_title('Gyro Bias Estimation')
        plt.tight_layout()
        plt.savefig('bias_estimation.png')
        print("Saved bias_estimation.png")
        plt.close(fig)
        
    except ImportError:
        print("Matplotlib not found. Skipping plotting.")

if __name__ == "__main__":
    main()
