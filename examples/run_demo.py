import numpy as np
import sys
import os
import matplotlib.pyplot as plt

# Ensure src is in path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.synthetic.trajectory import TrajectoryGenerator
from src.synthetic.imu import IMU
from src.fusion.complementary import ComplementaryFilter
from src.fusion.madgwick import Madgwick
from src.math_core.conversions import quat_to_euler
from src.analysis.metrics import compute_rmse

def run_scenario(name, duration, config):
    print(f"\n--- Running Scenario: {name} ---")
    
    # 1. Trajectory
    dt = 0.01
    traj_gen = TrajectoryGenerator(duration=duration, dt=dt)
    traj_data = traj_gen.generate_sinusoidal(
        freqs=[0.2, 0.1, 0.05], 
        amps=[np.radians(20), np.radians(20), np.radians(45)]
    )
    
    # 2. IMU Simulation
    imu = IMU(**config)
    accels, gyros, mags, truths = imu.generate(traj_data)
    
    # 3. Validation Checks
    print("* Physics Validation Checks:")
    
    # Check 1: Gyro Range (Approx)
    gyro_norm = np.linalg.norm(gyros, axis=1)
    print(f"  - Gyro Mean Norm: {np.mean(gyro_norm):.4f} rad/s")
    
    # Check 2: Accel Norm (Should be ~g = 9.81)
    accel_norm = np.linalg.norm(accels, axis=1)
    mean_acc = np.mean(accel_norm)
    print(f"  - Accel Mean Norm: {mean_acc:.4f} m/s^2 (Target: 9.81)")
    if abs(mean_acc - 9.81) > 0.5:
        print("    [WARNING] Accel norm deviates significantly from gravity!")
    else:
        print("    [PASS] Accel norm consistent with gravity.")

    # Check 3: Mag Norm (Should be constant if field is constant + hard iron)
    # Mag norm changes if hard-iron is large relative to field, but standard deviation should be low (noise only)
    mag_norm = np.linalg.norm(mags, axis=1)
    mag_std = np.std(mag_norm)
    print(f"  - Mag Norm Std Dev: {mag_std:.4f} (Should be low, dominated by noise)")
    
    # 4. Fusion
    # Complementary
    comp = ComplementaryFilter(alpha=0.98)
    q_comp = comp.run(accels, gyros, mags, dt)
    
    # Madgwick
    madg = Madgwick(beta=0.1)
    q_madg = madg.run(accels, gyros, mags, dt)
    
    # 5. Analysis
    e_gt = traj_data['euler']
    e_comp = quat_to_euler(q_comp)
    e_madg = quat_to_euler(q_madg)
    
    rmse_c = compute_rmse(e_comp, e_gt)
    rmse_m = compute_rmse(e_madg, e_gt)
    
    print(f"* Fusion Results (RMSE deg):")
    print(f"  - Complementary: R={np.degrees(rmse_c[0]):.2f}, P={np.degrees(rmse_c[1]):.2f}, Y={np.degrees(rmse_c[2]):.2f}")
    print(f"  - Madgwick:      R={np.degrees(rmse_m[0]):.2f}, P={np.degrees(rmse_m[1]):.2f}, Y={np.degrees(rmse_m[2]):.2f}")
    
    # Plotting
    plot_results(traj_data['time'], e_gt, e_comp, e_madg, name)
    
    return q_comp, q_madg, e_gt

def run_mag_off_test():
    print(f"\n--- Running Scenario: Mag-Off Drift Test ---")
    duration = 30.0
    dt = 0.01
    
    traj_gen = TrajectoryGenerator(duration=duration, dt=dt)
    traj_data = traj_gen.generate_sinusoidal(freqs=[0.1, 0.1, 0.0], amps=[0, 0, np.radians(180)]) # Pure Yaw rotation
    
    imu = IMU(gyro_bias_instability=1e-4, mag_hard_iron_std=0.0)
    accels, gyros, mags, _ = imu.generate(traj_data)
    
    # Artificial Bias Injection to cause drift
    gyros += np.array([0, 0, 0.05]) # 0.05 rad/s bias on Z
    
    # Run Madgwick with Mag (Standard)
    madg_on = Madgwick(beta=0.1)
    q_on = madg_on.run(accels, gyros, mags, dt)
    
    # Run Madgwick without Mag (Pass zeros or disable)
    madg_off = Madgwick(beta=0.1)
    mags_off = np.zeros_like(mags) # Zero mag
    q_off = madg_off.run(accels, gyros, mags_off, dt)
    
    # Compare Yaw Error
    e_gt = traj_data['euler']
    e_on = quat_to_euler(q_on)
    e_off = quat_to_euler(q_off)
    
    rmse_on = compute_rmse(e_on, e_gt)
    rmse_off = compute_rmse(e_off, e_gt)
    
    print(f"Yaw RMSE (Mag ON):  {np.degrees(rmse_on[2]):.2f} deg")
    print(f"Yaw RMSE (Mag OFF): {np.degrees(rmse_off[2]):.2f} deg")
    
    if rmse_off[2] > rmse_on[2] * 2:
        print("    [PASS] Drift detected when Mag is OFF.")
    else:
        print("    [WARNING] Drift distinction not clear.")
        
    # Plot Yaw Comparison
    plt.figure(figsize=(10, 5))
    plt.plot(traj_data['time'], np.degrees(e_gt[:, 2]), 'k--', label='Truth')
    plt.plot(traj_data['time'], np.degrees(e_on[:, 2]), 'g-', label='Mag ON')
    plt.plot(traj_data['time'], np.degrees(e_off[:, 2]), 'r-', label='Mag OFF')
    plt.title("Yaw Drift: Mag ON vs OFF")
    plt.legend()
    plt.grid()
    plt.savefig('mag_drift_test.png')
    plt.close()

def plot_results(time, gt, est1, est2, name):
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    labels = ['Roll', 'Pitch', 'Yaw']
    
    for i in range(3):
        axs[i].plot(time, np.degrees(gt[:, i]), 'k--', label='Truth')
        axs[i].plot(time, np.degrees(est1[:, i]), 'b-', alpha=0.7, label='Complementary')
        axs[i].plot(time, np.degrees(est2[:, i]), 'r-', alpha=0.7, label='Madgwick')
        axs[i].set_ylabel(f'{labels[i]} (deg)')
        axs[i].grid(True)
        if i == 0: axs[i].legend()
        
    axs[2].set_xlabel('Time (s)')
    axs[0].set_title(f'Scenario: {name}')
    plt.tight_layout()
    plt.savefig(f'demo_{name.lower().replace(" ", "_")}.png')
    plt.close()

if __name__ == "__main__":
    # Scenario 1: Clean (Validation)
    config_clean = {
        'accel_noise_std': 0.005,
        'gyro_noise_std': 0.001,
        'mag_noise_std': 0.01,
        'mag_hard_iron_std': 0.0
    }
    run_scenario("Clean Baseline", 10.0, config_clean)
    
    # Scenario 2: Dirty (Stress Test)
    config_dirty = {
        'accel_noise_std': 0.1,
        'gyro_noise_std': 0.05,
        'mag_noise_std': 0.2,
        'mag_hard_iron_std': 0.5, # Significant hard iron
        'gyro_bias_instability': 1e-4
    }
    run_scenario("High Noise & Bias", 10.0, config_dirty)
    
    # Scenario 3: Mag-Off Drift
    run_mag_off_test()
