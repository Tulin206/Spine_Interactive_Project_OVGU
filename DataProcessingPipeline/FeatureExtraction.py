"""
This python file takes segmented gait cycles returned from GaitProcessing.py and extracts features from each cycle
Possible features : velocity, Cadence, stride length, step length, stride time, swing time etc.
run() function combines parameters of each gait cycle and cognitive scores. Afterwards it returns
ultimate tabular data representation (n_cycles x n_features) and class labels.
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import cumulative_trapezoid as cumtrapz    ## ISRAT
#from scipy.integrate import cumtrapz   # Ayse
import scipy.integrate as integrate
from scipy.spatial.transform import Rotation as R
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from scipy.signal import find_peaks

from DataProcessingPipeline.FLEXIBILITY_features import FLEXIBILITY_features
from DataProcessingPipeline.GAIT_features import GAIT_features
from DataProcessingPipeline.STS_features import STS_features
from DataProcessingPipeline.TUG_features import TUG_features


class FeatureExtraction:

    def __init__(self, acc, gyr, events, fs, unproc_acc):
        """
        :param data: n_sub x n_sig&events
        :param cog: n_sub x n_cog_features cognitive features of all subjects
        """
        self.acc = acc
        self.mag_acc = np.sqrt(
            np.sum([np.power(acc[0, :], 2), np.power(acc[1, :], 2), np.power(acc[2, :], 2)], axis=0))
        self.gyr = gyr
        self.mag_gyr = np.sqrt(
            np.sum([np.power(gyr[0, :], 2), np.power(gyr[1, :], 2), np.power(gyr[2, :], 2)], axis=0))
        self.events = events
        self.fs = fs
        self.unproc_acc = unproc_acc
        # self.cog = cog

    # Calculates how much x values deviate from the average
    def _get_deviation(self, x):
        avg_x = np.mean(x)
        dev = np.abs(x - avg_x)
        sd = np.sqrt(np.mean(np.power(dev, 2)))
        return sd

    # Public wrapper method
    def get_deviation(self, x):
        return self._get_deviation(x)

    # Checks how acceleration changes between the first and last movements.
    def _get_decrease(self, x, rep, sit2stand, stand2sit):
        # Decrease in acceleration - difference between first 3 rep and last 3 rep
        f_three_rep = 0
        l_three_rep = 0
        for i in range(3):
            l = rep - 3
            f_three_rep += np.mean(x[sit2stand[i]:stand2sit[i]])
            l_three_rep += np.mean(x[sit2stand[l + i]:stand2sit[l + i]])
        f_three_rep = f_three_rep / 3
        l_three_rep = l_three_rep / 3
        decrease = (np.abs(f_three_rep - l_three_rep) / f_three_rep) * 100
        return decrease

    # public wrapper method
    def get_decrease(self, x, rep, sit2stand, stand2sit):
        return self._get_decrease(x, rep, sit2stand, stand2sit)

    # Computes orientation (angle) using gyroscope data.
    def _get_orientation(self, gyr):
        dt = 1.0 / self.fs
        orientations = []
        current_orientation = R.from_euler('xyz', [0, 0, 0])
        gyr = gyr.T
        for i in range(len(gyr)):
            # Açısal hızı açı olarak entegre et
            delta_angle = gyr[i] * dt  # rad
            delta_rotation = R.from_rotvec(delta_angle)
            current_orientation = current_orientation * delta_rotation
            orientations.append(current_orientation)

        return orientations

    # public wrapper method
    def get_orientation(self, gyr):
        return self._get_orientation(gyr)

    # Converts acceleration data from local to global coordinates.
    def _transform_acc(self, acc, orientations):
        acc_global = []
        acc = acc.T
        for i in range(len(acc)):
            # Lokal ivme vektörünü global koordinat sistemine dönüştür
            global_acc = orientations[i].apply(acc[i])
            acc_global.append(global_acc)
        return np.array(acc_global)

    # public wrapper method
    def transform_acc(self, acc, orientations):
        return self._transform_acc(acc, orientations)

    # Removes Earth's gravity (9.81 m/s²) from acceleration data.
    def _remove_gravity(self, acc_global):
        acc_corrected = []
        gravity = np.array([0, 0, 9.81])  # Yerçekimi vektörü (m/s²)

        for i in range(len(acc_global)):
            acc_corrected.append(acc_global[i] - gravity)
        return np.array(acc_corrected)

    # public wrapper method
    def remove_gravity(self, acc_global):
        return self._remove_gravity(acc_global)

    # Calculates velocity by integrating acceleration over time.
    # Uses numerical integration (cumtrapz) to compute velocity from acceleration.
    def _compute_velocity(self, acc_corrected):
        dt = 1.0 / self.fs
        velocity = cumtrapz(acc_corrected, dx=dt, axis=0, initial=0)
        return velocity

    # public wrapper method
    def compute_velocity(self, acc_corrected):
        return self._compute_velocity(acc_corrected)

    # Computes the average walking speed from the velocity magnitude.
    def _get_avg_vel(self, velocity):
        velocity_magnitude = np.linalg.norm(velocity, axis=1)
        avg_velocity = np.mean(velocity_magnitude)
        return avg_velocity, velocity_magnitude

    # public wrapper method
    def get_avg_vel(self, velocity):
        return self._get_avg_vel(velocity)

    # Computes velocity step by step.
    def _get_velocity(self, x):
        velocity = [0]
        time = 1 / self.fs
        for i in range(len(x)):
            velocity.append(velocity[-1] + x[i] * time)
        del velocity[0]
        # avg_vel = np.mean(velocity)
        return velocity

    # public wrapper method
    def get_velocity(self, x):
        return self._get_velocity(x)

    # Measures total angular movement of the body.
    # Uses numerical integration (simpson()) on gyroscope data to estimate angular displacement.
    def _get_angular_displacement(self, gyr, acc):
        # Total angular displacement of the trunk -
        # shape of x: signals(x,y,z) x time
        t = np.arange(0, gyr.shape[1], 1)
        # angular position
        ang_pos_x = integrate.simpson(gyr[0, :], t)
        ang_pos_y = integrate.simpson(gyr[1, :], t)
        ang_pos_z = integrate.simpson(gyr[2, :], t)
        total_disp = np.sqrt(np.power(ang_pos_x, 2) + np.power(ang_pos_y, 2) + np.power(ang_pos_z, 2))
        return total_disp

    # public wrapper method
    def get_angular_displacement(self, gyr, acc):
        return self._get_angular_displacement(gyr, acc)

    def _count_steps(self, signal_segment):                             # ISRAT
        """Count steps in a signal segment using peak detection"""
        if len(signal_segment) == 0:
            return 0

        # Smooth the signal using moving average
        window_size = int(0.25 * self.fs)  # 250ms window
        if window_size == 0:
            window_size = 1
        smoothed = np.convolve(signal_segment, np.ones(window_size) / window_size, mode='same')

        # Find peaks with minimum height and distance requirements
        min_peak_height = np.percentile(smoothed, 75)  # 75th percentile as threshold
        min_peak_distance = int(0.4 * self.fs)  # 400ms between steps

        peaks, _ = find_peaks(smoothed,
                              height=min_peak_height,
                              distance=min_peak_distance)
        return len(peaks)

    # public wrapper method
    def count_steps(self, signal_segment):
        return self._count_steps(signal_segment)

    # Improved Angular Velocity Calculation ------------------------------------------
    def _get_hip_angular_velocity(self, phase_start, phase_end):
        """Calculate mean hip angular velocity in deg/s for a movement phase"""
        # 1. Extract gyroscope data (assumed in rad/s)
        gyr_segment = self.gyr[:, phase_start:phase_end]

        # 2. Focus on sagittal plane (X-axis for flexion/extension)
        # Adjust axis index based on your sensor's coordinate system
        flexion_axis = 0  # Typically X-axis for hip flexion

        # 3. Convert rad/s to deg/s and calculate mean
        return np.mean(np.degrees(gyr_segment[flexion_axis, :]))

    # public wrapper method
    def get_hip_angular_velocity(self, phase_start, phase_end):
        return self._get_hip_angular_velocity(phase_start, phase_end)

    def _calculate_jerk(self, start_idx, end_idx):
        """
        Calculate normalized jerk metric for a movement phase
        :param start_idx: Start index of the phase
        :param end_idx: End index of the phase
        :return: Normalized jerk (dimensionless smoothness metric)
        """
        if end_idx <= start_idx:
            return np.nan

        # 1. Get filtered acceleration data (using existing gravity-removed data)
        phase_acc = self.acc[:, start_idx:end_idx]  # (3, N)

        # 2. Calculate jerk (derivative of acceleration)
        dt = 1 / self.fs
        jerk = np.gradient(phase_acc, dt, axis=1)  # (3, N)

        # 3. Calculate jerk magnitude
        jerk_magnitude = np.linalg.norm(jerk, axis=0)  # (N,)

        # 4. Normalize by movement duration and mean speed (paper-based method)
        movement_duration = (end_idx - start_idx) / self.fs
        mean_acc = np.mean(np.linalg.norm(phase_acc, axis=0))

        # Avoid division by zero
        if movement_duration < 0.1 or mean_acc < 1e-6:
            return np.nan

        # 5. Calculate normalized jerk metric
        normalized_jerk = np.sqrt(
            integrate.simpson(jerk_magnitude ** 2, dx=dt)
        ) * (movement_duration ** 2.5) / mean_acc

        return normalized_jerk

    # public wrapper method
    def calculate_jerk(self, start_idx, end_idx):
        return self._calculate_jerk(start_idx, end_idx)

    # Uses Kalman Filter to smooth velocity estimates from acceleration data.
    def _apply_Kalman_Filter(self, acc_data):
        # Define Kalman filter parameters
        dt = 1.0 / self.fs  # Time step
        x = np.array([0, 0])  # Initial state (velocity, acc)
        P = np.eye(2) * 100  # Initial covariance matrix
        F = np.array([[1, dt], [0, 1]])  # State transition matrix
        H = np.array([[0, 1]])  # Measurement matrix (measuring acc)
        Q = Q_discrete_white_noise(dim=2, dt=dt, var=0.1)  # Process noise covariance
        R = 1  # Measurement noise covariance
        num_samples = len(acc_data)
        # Kalman Filter loop over acceleration measurements
        velocity_estimates = []

        for z in acc_data:
            # Prediction step
            x = np.dot(F, x)
            P = np.dot(F, np.dot(P, F.T)) + Q

            # Measurement update step
            y = z - np.dot(H, x)  # Measurement residual
            S = np.dot(H, np.dot(P, H.T)) + R  # Residual covariance
            K = np.dot(P, np.dot(H.T, np.linalg.inv(S)))  # Kalman gain

            x = x + np.dot(K, y)  # Updated state estimate
            P = P - np.dot(K, np.dot(H, P))  # Updated state covariance

            # Store the estimated velocity
            velocity_estimates.append(x[0])

        # Output the velocity estimates
        velocity_estimates = np.array(velocity_estimates)

        return velocity_estimates

    # public wrapper method
    def apply_Kalman_Filter(self, acc_data):
        return self._apply_Kalman_Filter(acc_data)

    def run(self, test, task):
        print(f"Extracting features...\n")
        feat = []
        feat_list = []
        if test == "gait" and task in ["2min", "10m"]:
            gait_obj = GAIT_features(self)
            feat, feat_list = gait_obj.get_gait_features(task=task)

        elif test == "sts" and task in ["1min", "5rep"]:
            sts_obj = STS_features(self)
            feat, feat_list = sts_obj.get_STS_features(task=task)

        elif test == "tug" and task in ["single", "dual"]:
            tug_obj = TUG_features(self)
            # feat, feat_list = tug_obj.get_TUG_features(task=task)
            # Single-task processing
            single_features, single_features_list = tug_obj.get_TUG_features(task='single')
            # Calculate DTC
            dtc_values, dtc_names = tug_obj.calculate_DTC(single_features, single_features_list)
            return dtc_values, dtc_names

        elif test == "flex":
            flex_obj = FLEXIBILITY_features(self)
            feat, feat_list = flex_obj.get_flexibility_features()

        print("\nNumeric feature value: \n", feat)
        print("\nname of the feature list: \n", feat_list)
        print("\n")

        return feat, feat_list

    """
        if test == "gait" and task == "2min":
            gait_obj = GAIT_features(self)
            feat, feat_list = gait_obj.get_gait_features(task=task)
        elif test == "gait" and task == "10m":
            gait_obj = GAIT_features(self)
            feat, feat_list = gait_obj.get_gait_features(task=task)
        if test == "sts" and task == "1min":
            sts_obj = STS_features(self)
            feat, feat_list = sts_obj.get_STS_features(task=task)
        if test == "sts" and task == "5rep":
            sts_obj = STS_features(self)
            feat, feat_list = sts_obj.get_STS_features(task=task)
        if test == "tug" and task == "single":
            tug_obj = TUG_features(self)
            feat, feat_list = tug_obj.get_TUG_features(task=task)
        if test == "tug" and task == "dual":
            tug_obj = TUG_features(self)
            feat, feat_list = tug_obj.get_TUG_features(task=task)
    """