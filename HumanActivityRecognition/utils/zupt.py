import numpy as np
import matplotlib.pyplot as plt

# --- Sensor Noise Parameters (from datasheet, image1) ---
# Accelerometer
acc_noise_density = 120e-6 * 9.81  # [μg/√Hz] to [m/s^2/√Hz]
acc_bandwidth = 324  # Hz (use largest axis)
acc_rms_noise = acc_noise_density * np.sqrt(acc_bandwidth)  # [m/s^2]
R_acc = np.eye(3) * (acc_rms_noise ** 2)  # (m/s^2)^2, for velocity update

# Gyroscope (for orientation update, not used in this basic ZUPT example)
gyro_noise_density = 0.007 * np.pi / 180  # [deg/s/√Hz] to [rad/s/√Hz]
gyro_bandwidth = 255  # Hz
gyro_rms_noise = gyro_noise_density * np.sqrt(gyro_bandwidth)  # [rad/s]
R_gyro = np.eye(3) * (gyro_rms_noise ** 2)  # (rad/s)^2

# --- ZUPT Detection ---
def zupt_detect(accel, gyro, fs: int = 120, acc_thresh: float = 0.2, gyro_thresh: float = 0.05, window_size:int = 20, min_segment_len_seconds: float = None):
    N = accel.shape[0]
    zupt = np.zeros(N, dtype=bool)
    acc_mag = np.linalg.norm(accel, axis=1)
    gyro_mag = np.linalg.norm(gyro, axis=1)
    for i in range(N - window_size + 1):
        if np.std(acc_mag[i:i+window_size]) < acc_thresh and np.std(gyro_mag[i:i+window_size]) < gyro_thresh:
            zupt[i:i+window_size] = True

    if min_segment_len_seconds:
        min_samples = int(min_segment_len_seconds * fs)
        # Apply morphological operations to clean up ZUPT segments
        from scipy import ndimage
        zupt_cleaned = ndimage.binary_closing(zupt, structure=np.ones(min_samples))
        return zupt_cleaned

    return zupt

def get_state_from_zupt_kalman(fs=120, acc=None, zupt=None, dt=1/120):
    """
    Function to extract the state from the ZUPT Kalman filter.
    Returns positions, velocities, and biases.
    """
    dt=1/fs  # Sampling time in seconds
    # Kalman filter
    process_noise_cov = np.eye(9) * 1e-8  # Can be tuned
    kf = ZUPTKalman(
        dt=dt,
        process_noise_cov=process_noise_cov,
        measurement_noise_cov=R_acc  # Use R_acc for ZUPT velocity update
    )

    positions = []
    velocities = []
    biases = []
    if acc is not None and zupt is not None:
        for i in range(len(acc)):
            # Predict step
            kf.predict(acc[i])
            # Correct step with ZUPT
            kf.correct(zupt[i])
            # Store the state
            state = kf.get_state()
            positions.append(state[0:3].flatten())
            velocities.append(state[3:6].flatten())
            biases.append(state[6:9].flatten())

    positions = np.array(positions)
    velocities = np.array(velocities)
    biases = np.array(biases)
    return positions, velocities, biases

# --- ZUPT Kalman Filter Class ---
class ZUPTKalman:
    def __init__(self, dt, process_noise_cov, measurement_noise_cov):
        self.dt = dt
        # State: [px, py, pz, vx, vy, vz, bax, bay, baz]
        self.x = np.zeros((9, 1))
        self.P = np.eye(9) * 1e-3
        self.F = np.eye(9)
        for i in range(3):
            self.F[i, i+3] = dt
        self.Q = process_noise_cov
        self.R = measurement_noise_cov
        self.H = np.zeros((3, 9))
        self.H[0, 3] = 1
        self.H[1, 4] = 1
        self.H[2, 5] = 1

    def predict(self, accel):
        """Predict step with acceleration input (subtracting bias)."""
        a = accel.reshape(3, 1) - self.x[6:9]
        # Position update
        self.x[0:3] += self.x[3:6] * self.dt + 0.5 * a * self.dt ** 2
        # Velocity update
        self.x[3:6] += a * self.dt
        # Covariance update
        self.P = self.F @ self.P @ self.F.T + self.Q

    def correct(self, zupt=True):
        """Correct step during zero-velocity interval."""
        if not zupt:
            return
        # Measurement: measured velocity is zero
        z = np.zeros((3, 1))
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(9) - K @ self.H) @ self.P

    def get_state(self):
        """Return current [position, velocity, bias]."""
        return self.x.copy()

# --- Example Usage ---
if __name__ == "__main__":
    # Simulate some IMU data: still, then motion, then still
    N = 1000
    dt = 0.01  # Sampling rate: 100 Hz
    np.random.seed(42)
    accel = np.random.normal(0, 0.01, (N, 3))
    gyro = np.random.normal(0, 0.005, (N, 3))
    # Simulate motion in the middle
    accel[400:600] += np.random.normal(0, 0.5, (200, 3))
    gyro[400:600] += np.random.normal(0, 0.1, (200, 3))

    # Add Gaussian noise to the toy data (to test robustness)
    # accel += np.random.normal(0, 0.02, accel.shape)  # More noise
    # gyro += np.random.normal(0, 0.01, gyro.shape)    # More noise

    # ZUPT detection
    zupt = zupt_detect(accel, gyro)

    # Kalman filter
    process_noise_cov = np.eye(9) * 1e-4  # Can be tuned
    kf = ZUPTKalman(
        dt=dt,
        process_noise_cov=process_noise_cov,
        measurement_noise_cov=R_acc  # Use R_acc for ZUPT velocity update
    )

    positions = []
    velocities = []
    biases = []
    for i in range(N):
        kf.predict(accel[i])
        kf.correct(zupt[i])
        state = kf.get_state()
        positions.append(state[0:3].flatten())
        velocities.append(state[3:6].flatten())
        biases.append(state[6:9].flatten())

    positions = np.array(positions)
    velocities = np.array(velocities)
    biases = np.array(biases)

    # Plot original acceleration and gyroscope data
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(accel[:, 0], label='Accel X')
    plt.plot(accel[:, 1], label='Accel Y')
    plt.plot(accel[:, 2], label='Accel Z')
    plt.title("Simulated Accelerometer Data")
    plt.xlabel("Sample")
    plt.ylabel("Acceleration (m/s²)")
    plt.legend()
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(gyro[:, 0], label='Gyro X')
    plt.plot(gyro[:, 1], label='Gyro Y')
    plt.plot(gyro[:, 2], label='Gyro Z')
    plt.title("Simulated Gyroscope Data")
    plt.xlabel("Sample")
    plt.ylabel("Angular Velocity (rad/s)")
    plt.legend()
    plt.grid()

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(positions[:, 0], label='x')
    plt.plot(positions[:, 1], label='y')
    plt.plot(positions[:, 2], label='z')
    plt.title("Estimated Position with ZUPT-Kalman (Gaussian Noise Added)")
    plt.xlabel("Sample")
    plt.ylabel("Position (m)")
    plt.legend()
    plt.grid()

    plt.figure(figsize=(10, 3))
    plt.plot(zupt.astype(int), label="ZUPT Detected (1=True)")
    plt.title("Zero-Velocity Detection")
    plt.ylabel("ZUPT")
    plt.xlabel("Sample")
    plt.legend()
    plt.grid()

    plt.show()