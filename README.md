# InertiaLab

![Language](https://img.shields.io/badge/language-Python_3.10+-3776AB.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Stability](https://img.shields.io/badge/stability-prototype--research-orange.svg)
![Dependencies](https://img.shields.io/badge/dependencies-numpy_|_matplotlib-green.svg)

> "Navigation is the art of knowing where you are, by knowing where you've been and how you got there."

**InertiaLab** is a high-fidelity simulation and sensor fusion framework for 9-DoF Inertial Measurement Units (IMU). It serves as a rigorous preliminary study for **TEKNOFEST** autonomous systems competitions, providing a "glass-box" environment to master orientation estimation algorithms before deployment on physical hardware.

## The Context

This project is currently a **preliminary research study**. 
Its primary goal is to validate mathematical models and fusion algorithms in a controlled environment. 
Upon passing the TEKNOFEST pre-application stage, this codebase will evolve into the core guidance system for a physical platform equipped with real-world accelerometers, gyroscopes, and magnetometers.

## The Problem

Real-world sensors are messy. 
*   **Gyroscopes** drift over time due to bias instability.
*   **Accelerometers** are noisy and confuse gravity with linear acceleration.
*   **Magnetometers** are distorted by hard-iron and soft-iron magnetic anomalies.

Throwing raw sensor data at a PID controller results in crashes. You need robust sensor fusion to verify "where is down" and "where is north" reliably.

## The Solution

InertiaLab provides a deterministic, physics-based playground:

1.  **High-Fidelity Physics Engine**: Simulates realistic sensor errors, including Random Walk Bias (instability), Turn-on Bias, White Noise, and Hard-Iron Magnetic distortions.
2.  **State-of-the-Art Algorithms**:
    *   **Complementary Filter**: Simple, effective frequency-domain fusion.
    *   **Madgwick Filter**: Gradient-descent based fusion for MARG arrays.
    *   **Error-State Kalman Filter (ES-EKF)**: The gold standard for navigation, estimating not just orientation but also sensor biases in real-time.
3.  **Automated Validation**: A "CI for physics" that ensures the simulation obeys Newtonian mechanics (accel norm ~g, mag stability) and that filters actually correct drift.

## Architecture

We favor explicit mathematics over hidden libraries.

*   `src/math_core`: Quaternion algebra implementation from scratch (no `scipy.spatial` dependency).
*   `src/synthetic`: Trajectory generation and IMU error modeling.
*   `src/fusion`: Pure Python implementations of Complementary, Madgwick, and ES-EKF algorithms.
*   `src/analysis`: RMSE metrics, variation analysis, and plotting tools.

## Installation

```bash
# Clone the repository
git clone https://github.com/engnhn/inertialab.git
cd inertialab

# No strict dependency hell - just modern Python
# Standard system python is fine, or use a venv
pip install numpy matplotlib
```

## Usage

### 1. Run the Verification Demo
See the algorithms compete in different scenarios (Clean, High Noise, Mag-Off Drift).

```bash
python3 examples/run_demo.py
```
This generates plots in the root directory comparing ground truth vs. estimated orientation.

### 2. Main Simulation
Run the core simulation loop with configurable parameters:

```bash
python3 main.py
```

## Roadmap

*   **Phase 1 (Current)**: Python Prototype & Simulation Verification.
*   **Phase 2 (TEKNOFEST Prep)**: C++ Porting for embedded deployment.
*   **Phase 3 (Hardware)**: Integration with real sensors (MPU9250 / BNO055) via I2C/SPI.
*   **Phase 4 (Field Test)**: Real-time telemetry and tuning on the physical vehicle.

## Contributing

This is a research project. If you find a mathematical inconsistency or a better way to vectorize a calculation, open an issue.

## License

MIT
