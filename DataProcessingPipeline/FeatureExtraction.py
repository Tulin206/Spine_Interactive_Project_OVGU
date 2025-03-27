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
from filterpy.kalman import KalmanFilter


class FeatureExtraction:

    def __init__(self, acc, gyr, events, fs, unproc_acc):
        """
        :param data: n_sub x n_sig&events
        :param cog: n_sub x n_cog_features cognitive features of all subjects
        """
        self.acc = acc
        self.mag_acc = np.sqrt(
            np.sum([np.power(acc[0, :], 2), np.power(acc[1, :], 2), np.power(acc[2, :], 2)], axis=0))
        #print("mag_acc\n", self.mag_acc)
        self.gyr = gyr
        self.mag_gyr = np.sqrt(
            np.sum([np.power(gyr[0, :], 2), np.power(gyr[1, :], 2), np.power(gyr[2, :], 2)], axis=0))
        #print("mag_gyr\n", self.mag_gyr)
        self.events = events
        self.fs = fs
        self.unproc_acc = unproc_acc
        # self.cog = cog

<<<<<<< HEAD
    def calculate_mad(self, data):
        """Calculate Median Absolute Deviation (MAD)"""
        median = np.median(data)
        deviations = np.abs(data - median)
        mad = np.median(deviations)
        return mad

    def adaptive_threshold_mad(self, data, threshold_factor=5.3):
        """Calculate an adaptive threshold using MAD method"""
        mad = self.calculate_mad(data)
        median = np.median(data)
        threshold = median + threshold_factor * mad
        return threshold

    def calculate_adaptive_thresholds(self, accel_mag, gyro_mag):
        """Calculate separate adaptive thresholds for acceleration and gyroscope"""
        accel_thresh = self.adaptive_threshold_mad(accel_mag)
        gyro_thresh = self.adaptive_threshold_mad(gyro_mag)
        return accel_thresh, gyro_thresh

    # def calculate_mad(self, data):
    #     """Calculate Median Absolute Deviation (MAD)"""
    #     median = np.median(data)
    #     deviations = np.abs(data - median)
    #     mad = np.median(deviations)
    #     return mad
    #
    # def calculate_std(self, data):
    #     """Calculate Standard Deviation (STD)"""
    #     return np.std(data)
    #
    # def adaptive_threshold_mad(self, data, threshold_factor=3.0, use_std=False, window_size=50):
    #     """Calculate an adaptive threshold using MAD or Standard Deviation"""
    #     if use_std:
    #         # Use Standard Deviation based dynamic thresholding
    #         mean = np.mean(data)
    #         std_dev = self.calculate_std(data)
    #         threshold = mean + threshold_factor * std_dev
    #     else:
    #         # Use MAD for thresholding
    #         mad = self.calculate_mad(data)
    #         median = np.median(data)
    #         threshold = median + threshold_factor * mad
    #
    #     return threshold
    #
    # def adaptive_threshold_moving_avg(self, data, window_size=50, threshold_factor=3.0, use_std=False):
    #     """Calculate a dynamic threshold based on the moving average and MAD or STD"""
    #     if len(data) < window_size:
    #         window_size = len(data)
    #
    #     # Compute moving averages and thresholds over sliding windows
    #     thresholds = []
    #     for i in range(window_size, len(data)):
    #         window_data = data[i - window_size:i]
    #         threshold = self.adaptive_threshold_mad(window_data, threshold_factor, use_std)
    #         thresholds.append(threshold)
    #
    #     return thresholds
    #
    # def calculate_adaptive_thresholds(self, accel_mag, gyro_mag, use_std=False, window_size=50):
    #     """Calculate adaptive thresholds for acceleration and gyroscope using moving average method"""
    #     accel_thresh = self.adaptive_threshold_moving_avg(accel_mag, window_size, use_std=use_std)
    #     gyro_thresh = self.adaptive_threshold_moving_avg(gyro_mag, window_size, use_std=use_std)
    #     return accel_thresh, gyro_thresh

    def detect_turns(self, accel_mag, gyro_mag, accel_thresh=None, gyro_thresh=None):
        # Calculate thresholds automatically if not provided
        if accel_thresh is None or gyro_thresh is None:
            accel_thresh, gyro_thresh = self.calculate_adaptive_thresholds(accel_mag, gyro_mag)

=======
    def detect_turns(self, accel_mag, gyro_mag, accel_thresh=0.53, gyro_thresh=0.79):
>>>>>>> e3cc860018b71efc0bcfccc6b34dc06905a8f9b9
        walking_times = []
        turning_times = []

        is_turning = False
        turn_start = None
        walk_start = None

        for i in range(len(accel_mag)):
            if accel_mag[i] < accel_thresh and gyro_mag[i] > gyro_thresh:  # Turning phase
                if not is_turning:  # If we are switching to a turn
                    is_turning = True
                    if walk_start is not None:  # If a walking phase just ended
                        walking_times.append(i - walk_start)
                    turn_start = i
            else:  # Walking phase
                if is_turning:  # If we were turning and now switching back to walking
                    is_turning = False
                    turning_times.append(i - turn_start)
                    walk_start = i  # Start a new walking phase

        # Edge case: If the last phase was walking or turning, add it
        if is_turning and turn_start is not None:
            turning_times.append(len(accel_mag) - turn_start)
        elif not is_turning and walk_start is not None:
            walking_times.append(len(accel_mag) - walk_start)

        print("Walking times (samples):", walking_times)
        print("Turning times (samples):", turning_times)

        # Convert to seconds
        sampling_rate = self.fs

        walking_times = [t / sampling_rate for t in walking_times]
        turning_times = [t / sampling_rate for t in turning_times]

        print("Walking Times (seconds):", walking_times)
        print("Turning Times (seconds):", turning_times)

        return walking_times, turning_times

    def estimate_total_distance_v2(self, total_time, t_straight, t_turn):
        # Calculate the duration of one full cycle (T_straight + T_turn)
        cycle_duration = t_straight + t_turn

        # Calculate the total number of full cycles completed during the total time
        num_cycles = total_time / cycle_duration

<<<<<<< HEAD
        # Total distance per cycle (10m forward, 1.04m turn, 10m backward, 1.04m turn)
        cycle_distance = 10 + 1.04 + 10 + 1.04  # 22.08 meters
=======
        # Total distance per cycle (6m forward, 1.04m turn, 6m backward, 1.04m turn)
        cycle_distance = 10 + 1.04 + 10 + 1.04  # 14.08 meters
>>>>>>> e3cc860018b71efc0bcfccc6b34dc06905a8f9b9

        # Calculate the total distance walked
        total_distance = num_cycles * cycle_distance

<<<<<<< HEAD
        return total_distance

    import matplotlib.pyplot as plt

    def plot_phases(self, accel_mag, gyro_mag, walking_times_samples, turning_times_samples):
        plt.figure(figsize=(12, 6))

        # Plot raw data
        plt.plot(accel_mag, label='Accel Mag')
        plt.plot(gyro_mag, label='Gyro Mag')

        # Mark detected phases
        walk_mask = np.zeros_like(accel_mag)
        turn_mask = np.zeros_like(accel_mag)

        # Reconstruct phase timeline
        # (You'll need to implement this based on your walking_times/turning_times arrays)

        plt.plot(walk_mask, 'g', alpha=0.3, label='Walking')
        plt.plot(turn_mask, 'r', alpha=0.3, label='Turning')
        plt.legend()
        plt.show()

    """
    # Function to detect turns and straight walking based on threshold
    def detect_turns(self, accel_mag, gyro_mag, accel_thresh = 0.2, gyro_thresh = 0.65):
        # Create lists to store times for walking and turning
        walking_times = []
        turning_times = []

        is_turning = False
        turn_start = 0
        walk_start = 0

        for i in range(len(accel_mag)):
            if accel_mag[i] < accel_thresh and gyro_mag[i] > gyro_thresh:  # Turn detected
                if not is_turning:     ## If is_turning == False
                    is_turning = True
                    turn_start = i
                if i == len(accel_mag) - 1:  # End of data
                    turning_times.append(i - turn_start)
            else:  # Walking detected
                if is_turning:
                    is_turning = False
                    turning_times.append(i - turn_start)
                if walk_start == 0 or i - walk_start > 1:  # Ensure walking phase is separated
                    walk_start = i
                    walking_times.append(i - walk_start)
        print("Walking_Time: ", walking_times)
        print("Turning_Time: ", turning_times)
        return walking_times, turning_times

    # Function to calculate total distance walked based on straight walking and turning times
    def estimate_total_distance_v2(self, total_time, t_straight, t_turn):
        # Calculate the duration of one full cycle (T_straight + T_turn)
        cycle_duration = t_straight + t_turn

        # Calculate the total number of full cycles completed during the total time
        num_cycles = total_time / cycle_duration

        # Total distance per cycle (6m forward, 1.04m turn, 6m backward, 1.04m turn)
        cycle_distance = 6 + 1.04 + 6 + 1.04  # 14.08 meters

        # Calculate the total distance walked
        total_distance = num_cycles * cycle_distance
=======
        # Account for incomplete cycle if the last one is unfinished
        remaining_time = total_time % cycle_duration
        if remaining_time > 0:
            # Check if the remaining time includes a turning phase or not
            if remaining_time > t_straight:
                # Add the remaining distance from the last incomplete cycle
                total_distance += (remaining_time - t_straight) / t_turn * 1.04  # Proportional distance based on remaining turn time
            else:
                total_distance += (remaining_time / t_straight) * 10  # Add the remaining walking distance
>>>>>>> e3cc860018b71efc0bcfccc6b34dc06905a8f9b9

        return total_distance

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

    def moving_average(self, data, window_size=5):
        return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

    def _kalman_filter(self, data):
        kf = KalmanFilter(dim_x=1, dim_z=1)
        kf.x = np.array([data[0]])  # Initial state
        kf.P = np.eye(1) * 1000  # Large initial uncertainty
        kf.F = np.array([[1]])  # Transition matrix
        kf.H = np.array([[1]])  # Measurement matrix
        kf.R = 0.1  # Measurement noise
        kf.Q = 1e-5  # Process noise

        smoothed_data = []
        for i in range(len(data)):
            kf.predict()
            kf.update(data[i])
            smoothed_data.append(kf.x[0])

        return smoothed_data                             ## ISRAT

    # public wrapper method
    def kalman_filter(self, data):
        return self._kalman_filter(data)

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

    def Kalman_Filter(self, acc_data):
        """
        Uses a Kalman Filter to estimate velocity from acceleration data.
        """

        dt = 1.0 / self.fs  # Time step
        kf = KalmanFilter(dim_x=1, dim_z=1)  # 1D Kalman Filter (Velocity estimation)

        # State transition matrix (Velocity update: v_new = v_old + a*dt)
        kf.F = np.array([[1]])  # Velocity remains unless changed by acceleration

        # Measurement function (Direct acceleration measurement)
        kf.H = np.array([[1]])  # We observe acceleration directly

        # Process noise (assumed small uncertainty in acceleration)
        kf.Q = np.array([[0.01]])  # Process noise covariance (tune if needed)

        # Measurement noise (sensor noise in acceleration)
        kf.R = np.array([[1]])  # Measurement noise covariance

        # Initial state estimate (starting velocity)
        kf.x = np.array([[0]])  # Initial velocity assumed 0

        # Initial covariance matrix
        kf.P = np.array([[1]])

        velocity_estimates = []
        velocity = 0  # Start with zero velocity

        for acc in acc_data:
            # Predict step (velocity update with acceleration integration)
            velocity += acc * dt  # Integrating acceleration to get velocity

            # Kalman filter update
            kf.predict()
            kf.update(np.array([[velocity]]))  # Feeding estimated velocity

            # Store the velocity estimate
            velocity_estimates.append(kf.x[0, 0])

        return np.array(velocity_estimates)

    def run(self, test, task):
        from DataProcessingPipeline.FLEXIBILITY_features import FLEXIBILITY_features
        from DataProcessingPipeline.GAIT_features import GAIT_features
        from DataProcessingPipeline.STS_features import STS_features
        from DataProcessingPipeline.TUG_features import TUG_features
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