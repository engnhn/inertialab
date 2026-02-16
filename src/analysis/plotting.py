import matplotlib.pyplot as plt
import numpy as np

def plot_euler_comparison(time, euler_gt, euler_est, title="Orientation Estimate"):
    """
    Plot Ground Truth vs Estimated Euler angles.
    
    Args:
        time: (N,)
        euler_gt: (N, 3) [roll, pitch, yaw] in radians
        euler_est: (N, 3)
        title: Plot title
    """
    # Convert to degrees for plotting
    gt_deg = np.degrees(euler_gt)
    est_deg = np.degrees(euler_est)
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    labels = ['Roll', 'Pitch', 'Yaw']
    
    for i in range(3):
        axes[i].plot(time, gt_deg[:, i], 'k--', label='Ground Truth', linewidth=1.5)
        axes[i].plot(time, est_deg[:, i], 'r-', label='Estimated', linewidth=1.0)
        axes[i].set_ylabel(f'{labels[i]} (deg)')
        axes[i].grid(True)
        if i == 0:
            axes[i].legend()
            axes[i].set_title(title)
            
    axes[2].set_xlabel('Time (s)')
    plt.tight_layout()
    
    # Save instead of show, as this might be headless
    filename = title.lower().replace(" ", "_") + ".png"
    plt.savefig(filename)
    print(f"Saved plot to {filename}")
    plt.close(fig)

def plot_error(time, error, title="Estimation Error"):
    """
    Plot error over time.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(time, np.degrees(error), label=['Roll Err', 'Pitch Err', 'Yaw Err'])
    ax.set_title(title)
    ax.set_ylabel('Error (deg)')
    ax.set_xlabel('Time (s)')
    ax.legend()
    ax.grid(True)
    
    filename = title.lower().replace(" ", "_") + ".png"
    plt.savefig(filename)
    print(f"Saved error plot to {filename}")
    plt.close(fig)
