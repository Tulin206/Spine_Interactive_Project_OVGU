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

    # Sit-to-Stand Feature Extraction
    # Identifies sit-to-stand (sit2stand) and stand-to-sit (stand2sit) events.
    """
    Computes:
        Total time for task completion.
        Time taken per repetition.
        Peak acceleration and gyroscope values per repetition.
        Variability (standard deviation) in movement patterns.
        Mean and maximum acceleration values.
        Acceleration and gyroscope differences between transitions.
    """
    def get_STS_features(self, task="5rep"):
        """
        :param task: ["5rep", "1min"]
        :return:
        """
        test_name = "STS_" + str(task)
        sts_features = []
        sts_feature_list = []

        # two events - sit2stand and stand2sit
        sit2stand_all = np.where(self.events == 1)[0]
        stand2sit_all = np.where(self.events == 2)[0]
        rep = np.min([len(sit2stand_all), len(stand2sit_all)])
        print("repetition: ", rep)
        sit2stand = sit2stand_all[0:rep]
        stand2sit = stand2sit_all[0:rep]

        # Total time taken to perform the 5 reps
        total_t = (stand2sit[-1] - sit2stand[0]) / self.fs
        sts_features.append(total_t)
        sts_feature_list.append(str(test_name) + "_total_time")
        print(f"\nTotal time for 5 repetitions: {total_t:.2f} seconds")

        # Duration of each repetition
        t_rep = np.subtract(stand2sit, sit2stand) / self.fs
        print(f"\nDuration of each repetition: {t_rep}")
        mean_t_rep = np.mean(t_rep)
        print(f"\nMean duration of repetitions: {mean_t_rep:.2f} seconds")
        sts_features.append(mean_t_rep)
        sts_feature_list.append(str(test_name) + "_avg_time_rep")
        peak_acc_rep = []
        peak_gyr_rep = []

        # List to store ROM values for each repetition       # ISRAT
        rom_acc_rep = []
        rom_gyr_rep = []

        # List to store symmetry for each repetition         # ISRAT
        sym_acc_rep = []
        sym_gyr_rep = []

        # List to store acceleration for sit to stand and stand to sit for each repetition
        sitts_acc_rep = []
        standts_acc_rep =[]

        # List to store angular velocity for sit to stand and stand to sit for each repetition
        sitts_gyr_rep = []
        standts_gyr_rep = []

        for r in range(rep):
            peak_acc_rep.append(np.max(self.mag_acc[sit2stand[r]:stand2sit[r]]))
            peak_gyr_rep.append(np.max(self.mag_gyr[sit2stand[r]:stand2sit[r]]))

            # Compute mean acceleration for this repetition         # ISRAT
            mean_sitts_acc_rep = np.mean(self.mag_acc[sit2stand[r]:stand2sit[r]])
            sitts_acc_rep.append(mean_sitts_acc_rep)
            # Ensure valid range for the next repetition (r + 1 should not exceed rep - 1)
            if r < rep - 1:
                mean_standts_acc_rep = np.nanmean(self.mag_acc[stand2sit[r + 1]:sit2stand[r + 1]])  # Handle NaN values
            else:
                mean_standts_acc_rep = np.nan

            # mean_standts_acc_rep = np.mean(self.mag_acc[stand2sit[r + 1]:sit2stand[r + 1]]) if r < rep - 1 else np.nan
            standts_acc_rep.append(mean_standts_acc_rep)

            # Check symmetry for acceleration for this repetition       # ISRAT
            sym_acc = 1
            if mean_standts_acc_rep and (mean_sitts_acc_rep / mean_standts_acc_rep) > 0.1:
                sym_acc = 0

            # # Append symmetry (asymmetry) for acceleration
            sym_acc_rep.append(sym_acc)
            #sts_features.append(sym_acc)
            #sts_feature_list.append(f"{test_name}_sym_acc_rep{r + 1}")

            # Compute mean angular velocity for this repetition         # ISRAT
            mean_sitts_gyr_rep = np.mean(self.mag_gyr[stand2sit[r]] - self.mag_gyr[sit2stand[r]])
            sitts_gyr_rep.append(mean_sitts_gyr_rep)
            mean_standts_gyr_rep = (
                np.mean(self.mag_gyr[stand2sit[r + 1]] - self.mag_gyr[sit2stand[r + 1]])
                if r < rep - 1 else np.nan
            )
            standts_gyr_rep.append(mean_standts_gyr_rep)

            # Check symmetry for angular velocity        # ISRAT
            sym_gyr = 1
            if mean_standts_gyr_rep and (mean_sitts_gyr_rep / mean_standts_gyr_rep) > 0.1:
                sym_gyr = 0

            # Append symmetry (asymmetry) for angular velocity
            sym_gyr_rep.append(sym_gyr)
            #sts_features.append(sym_gyr)
            #sts_feature_list.append(f"{test_name}_sym_gyr_rep{r + 1}")

            # Extract the acceleration and gyroscope data for each repetition    # ISRAT
            acc_data = self.mag_acc[sit2stand[r]:stand2sit[r]]
            gyr_data = self.mag_gyr[sit2stand[r]:stand2sit[r]]

            # Calculate ROM for acceleration and gyroscope data    # ISRAT
            rom_acc = np.max(acc_data) - np.min(acc_data)
            rom_gyr = np.max(gyr_data) - np.min(gyr_data)

            # Append the ROM values to the lists    # ISRAT
            rom_acc_rep.append(rom_acc)
            rom_gyr_rep.append(rom_gyr)

        # Convert the peak lists to numpy arrays
        peak_acc_rep = np.array(peak_acc_rep)
        peak_gyr_rep = np.array(peak_gyr_rep)

        # Convert the acceleration and angular velocity lists to numpy arrays
        sitts_acc_rep = np.array(sitts_acc_rep)
        standts_acc_rep = np.array(standts_acc_rep)
        sitts_gyr_rep = np.array(sitts_gyr_rep)
        standts_gyr_rep = np.array(standts_gyr_rep)

        # Append the overall peak acceleration and gyroscope values for all repetitions
        sts_features.append(peak_acc_rep)
        sts_feature_list.append(str(test_name) + "_peak_acc")
        sts_features.append(peak_gyr_rep)
        sts_feature_list.append(str(test_name) + "_peak_gyr")

        # Append the overall acceleration and angular velocity values for all repetitions
        sts_features.append(sitts_acc_rep)
        sts_feature_list.append(str(test_name) + "_sitts_acc_rep")
        sts_features.append(standts_acc_rep)
        sts_feature_list.append(str(test_name) + "_standts_acc_rep")
        sts_features.append(sitts_gyr_rep)
        sts_feature_list.append(str(test_name) + "_sitts_gyr_rep")
        sts_features.append(standts_gyr_rep)
        sts_feature_list.append(str(test_name) + "_standts_gyr_rep")

        # Convert the symmetry lists to numpy array     # ISRAT
        sym_acc_rep = np.array(sym_acc_rep)
        sts_feature_list.append(str(test_name) + "_sym_acc_rep")
        sym_gyr_rep = np.array(sym_gyr_rep)
        sts_feature_list.append(str(test_name) + "_sym_gyr_rep")

        # Append the overall symmetry for acceleration and gyroscope values for all repetitions        # ISRAT
        sts_features.append(sym_acc_rep)
        sts_features.append(sym_gyr_rep)
        
        # Convert the ROM lists to numpy arrays    # ISRAT
        rom_acc_rep = np.array(rom_acc_rep)
        rom_gyr_rep = np.array(rom_gyr_rep)

        # Append the ROM features to the feature list    # ISRAT
        sts_features.append(rom_acc_rep)
        sts_feature_list.append(str(test_name) + "_rom_acc")
        sts_features.append(rom_gyr_rep)
        sts_feature_list.append(str(test_name) + "_rom_gyr")

        # Variability in repetitions [Rep-to-rep deviation (time)]
        # Calculate the standard deviation and coefficient of variation for time, acceleration, and gyroscope
        t_sd = self._get_deviation(t_rep)
        t_cv = (t_sd / mean_t_rep) * 100  # Coefficient of variation    # ISRAT
        peak_acc_sd = self._get_deviation(peak_acc_rep)
        peak_acc_cv = (peak_acc_sd / np.mean(peak_acc_rep)) * 100  # Coefficient of variation   ## ISRAT
        peak_gyr_sd = self._get_deviation(peak_gyr_rep)
        peak_gyr_cv = (peak_gyr_sd / np.mean(peak_gyr_rep)) * 100  # Coefficient of variation   ## ISRAT

        # Append the variability features
        sts_features.append(t_sd)
        sts_feature_list.append(str(test_name) + "_sd_time")
        sts_features.append(t_cv)
        sts_feature_list.append(str(test_name) + "_cv_time")
        sts_features.append(peak_acc_sd)
        sts_feature_list.append(str(test_name) + "_sd_acc")
        sts_features.append(peak_acc_cv)                      ## ISRAT
        sts_feature_list.append(str(test_name) + "_cv_acc")   ## ISRAT
        sts_features.append(peak_gyr_sd)
        sts_feature_list.append(str(test_name) + "_sd_gyr")
        sts_features.append(peak_gyr_cv)                      ## ISRAT
        sts_feature_list.append(str(test_name) + "_cv_gyr")   ## ISRAT

        # mean acc, max acc
        mean_acc = np.mean(self.mag_acc)
        max_acc = np.max(self.mag_acc)
        sts_features.append(mean_acc)
        sts_feature_list.append(str(test_name) + "_mean_acc")
        sts_features.append(max_acc)
        sts_feature_list.append(str(test_name) + "_max_acc")

        # mean acc and mean gyr of sittstand and standtsit
        mean_sitts_acc = np.mean(self.mag_acc[stand2sit] - self.mag_acc[sit2stand])
        mean_standts_acc = np.mean(self.mag_acc[stand2sit[1:]] - self.mag_acc[sit2stand[:-1]])
        mean_sitts_gyr = np.mean(self.mag_gyr[stand2sit] - self.mag_gyr[sit2stand])
        mean_standts_gyr = np.mean(self.mag_gyr[stand2sit[1:]] - self.mag_gyr[sit2stand[:-1]])
        sts_features.append(mean_sitts_acc)
        sts_feature_list.append(str(test_name) + "_mean_sitts_acc")
        sts_features.append(mean_standts_acc)
        sts_feature_list.append(str(test_name) + "_mean_standts_acc")
        sts_features.append(mean_sitts_gyr)
        sts_feature_list.append(str(test_name) + "_mean_sitts_gyr")
        sts_features.append(mean_standts_gyr)
        sts_feature_list.append(str(test_name) + "_mean_standts_gyr")

        # Symmetry between Sit-to-Stand and Stand-to-Sit Movements
        # Symmetry in acceleration (Hip Movement)
        sym = 1
        if (mean_sitts_acc / mean_standts_acc) > 0.1:
            # if sym=0, then asymmetric
            sym = 0
        sts_features.append(sym)
        sts_feature_list.append(str(test_name) + "_sym")

        # symmetry in angular velocity                          # ISRAT
        # Mean angular velocity during sit-to-stand and stand-to-sit
        mean_sitts_gyr = np.mean(self.mag_gyr[stand2sit] - self.mag_gyr[sit2stand])
        mean_standts_gyr = np.mean(self.mag_gyr[stand2sit[1:]] - self.mag_gyr[sit2stand[:-1]])

        # Check symmetry for angular velocity (similar to the acceleration symmetry check)       # ISRAT
        sym_gyr = 1
        if (mean_sitts_gyr / mean_standts_gyr) > 0.1:
            # if the ratio is > 10%, consider it asymmetric
            sym_gyr = 0

        # Append symmetry (asymmetry) for angular velocity         # ISRAT
        sts_features.append(sym_gyr)
        sts_feature_list.append(str(test_name) + "_sym_gyr")

        # angular displacement
        ang_disp = self._get_angular_displacement(self.gyr, self.acc)
        f_rep_disp = self._get_angular_displacement(self.gyr[:, sit2stand[0]:stand2sit[0]],
                                                    self.acc[:, sit2stand[0]:stand2sit[0]])
        l_rep_disp = self._get_angular_displacement(self.gyr[:, sit2stand[-1]:stand2sit[-1]],
                                                    self.acc[:, sit2stand[-1]:stand2sit[-1]])
        sts_features.append(ang_disp)
        sts_feature_list.append(str(test_name) + "_ang_disp")
        sts_features.append(f_rep_disp)
        sts_feature_list.append(str(test_name) + "_f_rep_disp")
        sts_features.append(l_rep_disp)
        sts_feature_list.append(str(test_name) + "_l_rep_disp")

        # power calculation
        mean_power_acc = np.mean(
            (np.power(self.acc[0, :], 2) + np.power(self.acc[1, :], 2) + np.power(self.acc[2, :], 2)) / len(
                self.mag_acc))
        mean_power_gyr = np.mean(
            (np.power(self.gyr[0, :], 2) + np.power(self.gyr[1, :], 2) + np.power(self.gyr[2, :], 2)) / len(
                self.mag_gyr))

        # Store power in features
        sts_features.append(mean_power_acc)
        sts_feature_list.append(str(test_name) + "_mean_power_acc")
        sts_features.append(mean_power_gyr)
        sts_feature_list.append(str(test_name) + "_mean_power_gyr")

        # **New Feature**: Differences between the first and last repetition for fatigue detection  # ISRAT
        decrease = self._get_decrease(self.mag_acc, rep, sit2stand, stand2sit)
        sts_features.append(decrease)
        sts_feature_list.append(str(test_name) + "_fatigue")

        if task == "1min":
            # Decrease in acceleration - difference between first 3 rep and last 3 rep
            dec_acc = self._get_decrease(self.mag_acc, rep, sit2stand, stand2sit)
            dec_gyr = self._get_decrease(self.mag_gyr, rep, sit2stand, stand2sit)
            sts_features.append(dec_acc)
            sts_feature_list.append(str(test_name) + "_dec_acc")
            sts_features.append(dec_gyr)
            sts_feature_list.append(str(test_name) + "_dec_gyr")

        return sts_features, sts_feature_list

    def get_TUG_features(self, task='single'):
        """
        :param task: ["single", "dual"]
        :return:
        """
        test_name = "TUG_" + str(task)
        tug_features = []
        tug_feature_list = []

        # seven events - sit2stand (1-2), mid_rotation-left/right (3-4), final_rotation 5, stand2sit(6-7)
        # Total time taken to perform the test
        event1 = np.where(self.events == 1)[0][0]
        event2 = np.where(self.events == 2)[0][0]
        event3 = np.where(self.events == 3)[0][0]
        event4 = np.where(self.events == 4)[0][0]
        event5 = np.where(self.events == 5)[0][0]
        event6 = np.where(self.events == 6)[0][0]
        event7 = np.where(self.events == 7)[0][0]

        # Total test time (event1 to event7)
        total_t = (event7 - event1) / self.fs
        tug_features.append(total_t)
        tug_feature_list.append(str(test_name) + "_total_time")

        # mean acc, peak acc
        mean_acc = np.mean(self.mag_acc)
        max_acc = np.max(self.mag_acc)
        tug_features.append(mean_acc)
        tug_feature_list.append(str(test_name) + "_mean_acc")
        tug_features.append(max_acc)
        tug_feature_list.append(str(test_name) + "_max_acc")

        # sit2stand and stand2sit time, mean acc, mean gyr
        t_sit2stand = (event2 - event1) / self.fs
        t_stand2sit = (event7 - event6) / self.fs
        mean_sitts_acc = np.mean(self.mag_acc[event1:event2])
        mean_standts_acc = np.mean(self.mag_acc[event6:event7])
        mean_sitts_gyr = np.mean(self.mag_gyr[event1:event2])
        mean_standts_gyr = np.mean(self.mag_gyr[event6:event7])

        tug_features.append(t_sit2stand)
        tug_feature_list.append(str(test_name) + "_t_sit2stand")
        tug_features.append(t_stand2sit)
        tug_feature_list.append(str(test_name) + "_t_stand2sit")
        tug_features.append(mean_sitts_acc)
        tug_feature_list.append(str(test_name) + "_mean_sitts_acc")
        tug_features.append(mean_standts_acc)
        tug_feature_list.append(str(test_name) + "_mean_standts_acc")
        tug_features.append(mean_sitts_gyr)
        tug_feature_list.append(str(test_name) + "_mean_sitts_gyr")
        tug_features.append(mean_standts_gyr)
        tug_feature_list.append(str(test_name) + "_mean_standts_gyr")

        # Sit-to-stand phase angular velocity
        ang_vel_sit2stand = self._get_hip_angular_velocity(event1, event2)

        # Stand-to-sit phase angular velocity
        ang_vel_stand2sit = self._get_hip_angular_velocity(event6, event7)

        # Append features (replacing the original implementation)
        tug_features.append(ang_vel_sit2stand)
        tug_feature_list.append(f"{test_name}_sit2stand_ang_vel")

        tug_features.append(ang_vel_stand2sit)
        tug_feature_list.append(f"{test_name}_stand2sit_ang_vel")

        # forward and return gait time and acc
        t_forward = (event3 - event2) / self.fs
        t_return = (event5 - event4) / self.fs

        # Pure walking time (event2 to event3 + event4 to event5)
        walking_time = t_forward + t_return  # Total walking time in seconds
        # walking_time = ((event3 - event2) + (event5 - event4)) / self.fs

        # TUG gait speed (Pure walking speed (excludes sit/stand/turn))
        pure_gait_speed = 6.0 / walking_time if walking_time > 0 else 0
        tug_features.append(pure_gait_speed)
        tug_feature_list.append(str(test_name) + "_gait_speed")

        # Calculate walking features -------------------------------------------------
        # Get accelerometer segments for walking phases
        forward_walk = self.mag_acc[event2:event3]
        return_walk = self.mag_acc[event4:event5]

        # Count steps in each phase
        forward_steps = self._count_steps(forward_walk)
        return_steps = self._count_steps(return_walk)
        total_steps = forward_steps + return_steps

        # Stride length (assuming total walking distance is 6 meters)
        stride_length = 6.0 / total_steps if total_steps > 0 else 0

        # Cadence (steps per minute)
        cadence = (total_steps / walking_time) * 60 if walking_time > 0 else 0

        acc_forward = np.mean(forward_walk)
        acc_return = np.mean(return_walk)

        # Append TUG features
        tug_features.append(t_forward)
        tug_feature_list.append(str(test_name) + "_t_forward")

        tug_features.append(t_return)
        tug_feature_list.append(str(test_name) + "_t_return")

        tug_features.append(acc_forward)
        tug_feature_list.append(str(test_name) + "_acc_forward")

        tug_features.append(acc_return)
        tug_feature_list.append(str(test_name) + "_acc_return")

        tug_features.append(total_steps)
        tug_feature_list.append(f"{test_name}_step_counts")

        tug_features.append(stride_length)
        tug_feature_list.append(f"{test_name}_stride_length")

        tug_features.append(cadence)
        tug_feature_list.append(f"{test_name}_cadence")

        # turn time
        turn_t = (event4 - event3) / self.fs
        mean_acc_turn = np.mean(self.mag_acc[event3:event4])
        tug_features.append(turn_t)
        tug_feature_list.append(str(test_name) + "_turn_t")
        tug_features.append(mean_acc_turn)
        tug_feature_list.append(str(test_name) + "_mean_acc_turn")

        # Calculate jerk metrics for key phases    # ISRAT
        phases = {
            "sit2stand": (event1, event2),
            "stand2sit": (event6, event7),
            "forward_walk": (event2, event3),
            "return_walk": (event4, event5)
        }

        for phase_name, (start, end) in phases.items():
            jerk = self._calculate_jerk(start, end)
            tug_features.append(jerk)
            tug_feature_list.append(f"{test_name}_{phase_name}_jerk")

        """# velocity
        t = np.arange(0,len(self.mag_acc),1)
        vel_forward = integrate.cumtrapz(self.mag_acc, t, initial=0)"""
        """# Smoothness - jerk: the derivative of acceleration
        t_range = np.arange(0, self.acc.shape[1], 1)
        j_x = FinDiff(self.acc[0, :], t_range, 1)
        j_y = FinDiff(self.acc[1, :], t_range, 1)
        j_z = FinDiff(self.acc[2, :], t_range, 1)
        jerk = np.mean(np.sqrt(j_x + j_y + j_z))
        print('jerk', jerk)"""

        """
        if task == 'dual':
            # Accuracy of the answers
            # (num_correct / total) * 100
            correct = 0
            tug_features.append(correct)
            tug_feature_list.append(str(test_name) + "_correct")
            """

        return tug_features, tug_feature_list

    def calculate_DTC(self, single_task_features, single_feature_names):         # ISRAT
        """
        Calculate Dual-Task Cost for all comparable features
        :param single_task_features: Features from single-task TUG
        :param single_feature_names: Names of single-task features
        :return: DTC features and names
        """
        dtc_features = []
        dtc_feature_list = []

        # Get dual-task features from current instance
        dual_task_features, dual_feature_names = self.get_TUG_features(task='dual')

        # Create mapping for feature lookup
        single_feat_map = {name: (idx, value) for idx, (name, value)
                           in enumerate(zip(single_feature_names, single_task_features))}

        for dual_idx, dual_name in enumerate(dual_feature_names):
            # Extract base feature name (remove "dual" identifier)
            base_name = dual_name.replace("TUGdual_", "TUG_")

            # Find matching single-task feature
            single_name = base_name.replace("TUG_", "TUGsingle_")

            if single_name in single_feat_map:
                single_idx, single_value = single_feat_map[single_name]
                dual_value = dual_task_features[dual_idx]

                # Handle zero-division and NaN cases
                if single_value == 0 or np.isnan(single_value) or np.isnan(dual_value):
                    dtc = np.nan
                else:
                    dtc = ((dual_value - single_value) / single_value) * 100

                dtc_features.append(dtc)
                dtc_feature_list.append(f"DTC_{base_name.replace('TUG_', '')}")

        return dtc_features, dtc_feature_list

    def get_gait_features(self, task="2min"):
        test_name = str(task) + "_gait"
        gait_features = []
        gait_feature_list = []

        # seven events - HS, LP, MSt, TSt, PSw, TO, MSw
        HS = np.where(self.events == 1)[0]
        MSt = np.where(self.events == 3)[0]
        TSt = np.where(self.events == 4)[0]
        PSw = np.where(self.events == 5)[0]
        TO = np.where(self.events == 6)[0]
        MSw = np.where(self.events == 7)[0]

        # num of gait cycle
        num_gc = np.min([len(HS), len(MSw)])
        gait_features.append(num_gc)
        gait_feature_list.append(str(test_name) + "_num_gc")
        mag_acc = self.mag_acc[HS[0]:MSw[-1]]
        acc = self.acc[:, HS[0]:MSw[-1]]
        gyr = self.gyr[:, HS[0]:MSw[-1]]
        duration = (MSw[-1] - HS[0]) / self.fs

        # num steps
        n_steps = num_gc * 2
        gait_features.append(n_steps)
        gait_feature_list.append(str(test_name) + "_n_steps")

        # mean acc, max acc
        mean_acc = np.mean(mag_acc)
        max_acc = np.max(mag_acc)
        gait_features.append(mean_acc)
        gait_feature_list.append(str(test_name) + "_mean_acc")
        gait_features.append(max_acc)
        gait_feature_list.append(str(test_name) + "_max_acc")
        # average velocity
        # orientations = self._get_orientation(gyr)
        # acc_global = self._transform_acc(acc, orientations)
        # acc_corrected = self._remove_gravity(acc.T)
        """t = np.arange(0, len(acc[0,:]), 1)
        velocity_x = np.power(self._get_velocity(self.acc[0, :]), 2)
        velocity_y = np.power(self._get_velocity(self.acc[1, :]), 2)
        velocity_z = np.power(self._get_velocity(self.acc[2, :]), 2)"""

        # velocity = self._compute_velocity(acc_corrected)
        velocity_x = np.power(self._apply_Kalman_Filter(self.acc[0, :]), 2)
        velocity_y = np.power(self._apply_Kalman_Filter(self.acc[1, :]), 2)
        velocity_z = np.power(self._apply_Kalman_Filter(self.acc[2, :]), 2)
        mag_vel = np.sqrt(velocity_x + velocity_y + velocity_z)
        avg_vel = np.mean(mag_vel) * 10
        # avg_vel = np.mean(mag_vel)        # ISRAT
        gait_features.append(avg_vel)
        gait_feature_list.append(str(test_name) + "_avg_vel")     ## Gait_Speed

        """plt.plot(velocity_x)
        plt.plot(velocity_y)
        plt.plot(velocity_z)
        plt.show()"""

        """#plt.plot(self.acc[2, 0:2000])
        plt.plot(self.acc[2, 0:1500])
        plt.plot(mag_vel[0:1500])
        plt.show()
        plt.plot(self._get_velocity(self.acc[0, :])[0:5000])
        plt.plot(self._get_velocity(self.acc[1, :])[0:5000])
        plt.plot(self._get_velocity(self.acc[2, :])[0:5000])
        plt.plot(self.acc[2, :])
        plt.show()"""

        # Fatigue index calculation     #### ISRAT
        fatigue_index = 0.0
        if num_gc >= 2:
            velocities_per_cycle = []
            for i in range(num_gc):
                start = HS[i] - HS[0]
                end = MSw[i] - HS[0]
                start = max(0, start)
                end = min(len(mag_vel), end)
                if start < end:
                    cycle_vel = mag_vel[start:end]
                    velocities_per_cycle.append(np.mean(cycle_vel) * 10)

            # Print velocities per cycle
            print("Velocities per cycle:", velocities_per_cycle)

            if len(velocities_per_cycle) >= 2:
                x = np.arange(len(velocities_per_cycle))
                slope, _ = np.polyfit(x, velocities_per_cycle, 1)
                fatigue_index = slope

        gait_features.append(fatigue_index)
        gait_feature_list.append(f"{test_name}_fatigue_index")

        # Additional fatigue feature: percent change in cycle velocity    ### ISRAT
        fatigue_percent_change = 0.0
        if velocities_per_cycle:
            first_cycle = velocities_per_cycle[0]
            last_cycle = velocities_per_cycle[-1]
            print(f"First cycle velocity: {first_cycle}")
            print(f"Last cycle velocity: {last_cycle}")

            if first_cycle != 0:
                fatigue_percent_change = ((last_cycle - first_cycle) / first_cycle) * 100
                print(f"Computed Fatigue Percent Change: {fatigue_percent_change}%")

        # Append the fatigue percent change feature
        gait_features.append(fatigue_percent_change)
        gait_feature_list.append(f"{test_name}_fatigue_percent_change")

        # Check for outliers in velocity data
        if len(velocities_per_cycle) > 0:
            q1 = np.percentile(velocities_per_cycle, 25)
            q3 = np.percentile(velocities_per_cycle, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            outliers = [v for v in velocities_per_cycle if v < lower_bound or v > upper_bound]
            print("Outliers in velocity data:", outliers)

        # Append fatigue percent change feature
        gait_features.append(fatigue_percent_change)
        gait_feature_list.append(f"{test_name}_fatigue_percent_change")

        # Plot velocity trend
        if velocities_per_cycle:
            plt.plot(range(len(velocities_per_cycle)), velocities_per_cycle, marker='o', linestyle='-')
            plt.xlabel("Gait Cycle Number")
            plt.ylabel("Mean Velocity per Cycle (m/s)")
            plt.title("Velocity Change Over Gait Cycles")
            plt.show()

        # Gait variability metrics     ### ISRAT
        def add_variability_metrics(values, metric_name):
            if len(values) > 0:
                std = np.std(values)
                mean = np.mean(values)
                cv = std / mean if mean != 0 else 0.0
            else:
                std = cv = 0.0
            gait_features.extend([std, cv])
            gait_feature_list.extend([f"{test_name}_{metric_name}_std",
                                      f"{test_name}_{metric_name}_cv"])

        # Stride time variability    ### ISRAT
        stride_times = (HS[1:num_gc] - HS[:num_gc - 1]) / self.fs
        add_variability_metrics(stride_times, "stride_time")

        # Step length variability    ### ISRAT
        step_lengths = ((TSt[:num_gc] - HS[:num_gc]) / self.fs * avg_vel)
        add_variability_metrics(step_lengths, "step_length")

        # Stride length variability     ### ISRAT
        stride_lengths = ((HS[1:num_gc] - HS[:num_gc - 1]) / self.fs * avg_vel)
        add_variability_metrics(stride_lengths, "stride_length")

        # distance
        distance = duration * avg_vel
        gait_features.append(distance)
        gait_feature_list.append(str(test_name) + "_distance")

        # average step length - step len = TSt - HS
        avg_step_len = np.mean(((TSt[0:num_gc] - HS[0:num_gc]) / self.fs) * avg_vel)
        gait_features.append(avg_step_len)
        gait_feature_list.append(str(test_name) + "_avg_step_len")

        # average stride length - stride len = HS - HS
        avg_stride_len = np.mean(((HS[1:num_gc] - HS[0:num_gc - 1]) / self.fs) * avg_vel)
        gait_features.append(avg_stride_len)
        gait_feature_list.append(str(test_name) + "_avg_stride_len")

        # stance
        stance = (TO[0:num_gc] - HS[0:num_gc]) / self.fs
        mean_stance = np.mean(stance)
        stance_per = (mean_stance / duration) * 100
        gait_features.append(mean_stance)
        gait_feature_list.append(str(test_name) + "_mean_stance")
        gait_features.append(stance_per)
        gait_feature_list.append(str(test_name) + "_stance_per")

        # Swing
        swing = (HS[1:num_gc] - TO[0:num_gc - 1]) / self.fs
        mean_swing = np.mean(swing)
        swing_per = (mean_swing / duration) * 100
        gait_features.append(mean_swing)
        gait_feature_list.append(str(test_name) + "_mean_swing")
        gait_features.append(swing_per)
        gait_feature_list.append(str(test_name) + "_swing_per")

        # Cadence
        cadence = n_steps / duration
        gait_features.append(cadence)
        gait_feature_list.append(str(test_name) + "_cadence")

        # linear acc
        avg_ax = np.mean(acc[0, :])
        avg_ay = np.mean(acc[1, :])
        avg_az = np.mean(acc[2, :])
        gait_features.append(avg_ax)
        gait_feature_list.append(str(test_name) + "_avg_ax")
        gait_features.append(avg_ay)
        gait_feature_list.append(str(test_name) + "_avg_ay")
        gait_features.append(avg_az)
        gait_feature_list.append(str(test_name) + "_avg_az")

        # ang vel
        avg_wx = np.mean(gyr[0, :])
        avg_wy = np.mean(gyr[1, :])
        avg_wz = np.mean(gyr[2, :])
        gait_features.append(avg_wx)
        gait_feature_list.append(str(test_name) + "_avg_wx")
        gait_features.append(avg_wy)
        gait_feature_list.append(str(test_name) + "_avg_wy")
        gait_features.append(avg_wz)
        gait_feature_list.append(str(test_name) + "_avg_wz")

        # double support time
        ds_time1 = (MSt[0:num_gc] - HS[0:num_gc]) / self.fs
        ds_time2 = (TO[0:num_gc] - TSt[0:num_gc]) / self.fs
        ds_time = ds_time1 + ds_time2
        mean_double_support = np.mean(ds_time)
        gait_features.append(mean_double_support)
        gait_feature_list.append(str(test_name) + "_mean_double_support")
        ds_per = (mean_double_support / duration) * 100
        gait_features.append(ds_per)
        gait_feature_list.append(str(test_name) + "_ds_per")

        # **Gait Speed Calculation**     ISRAT
        total_time = duration
        #total_distance = n_steps * np.mean(step_lengths)  # Total distance walked    # Ensure step_lengths is a single value
        #gait_speed = total_distance / total_time  # Speed in meters per second (m/s)
        gait_speed = distance / total_time  # Speed in meters per second (m/s)

        gait_features.append(gait_speed)
        gait_feature_list.append(str(test_name) + "_gait_speed")

        #gait_features.append(total_distance)
        #gait_feature_list.append(str(test_name) + "_total_distance")

        # ========== Symmetry Index Calculations ==========
        # Step Length Symmetry        #### ISRAT
        step_lens = ((TSt[0:num_gc] - HS[0:num_gc]) / self.fs) * avg_vel
        si_step = 0.0
        if len(step_lens) >= 2:
            step_lens = np.array(step_lens)  # Convert to numpy array
            diffs = np.abs(np.diff(step_lens))
            means = (step_lens[:-1] + step_lens[1:]) / 2
            valid = means > 0.001
            si_step = np.mean((diffs[valid] / means[valid]) * 100) if np.any(valid) else 0

        # Stride Time Symmetry
        stride_times = (HS[1:num_gc] - HS[0:num_gc - 1]) / self.fs
        si_stride_time = 0.0
        if len(stride_times) >= 2:
            stride_times = np.array(stride_times)  # Convert to numpy array
            diffs = np.abs(np.diff(stride_times))
            means = (stride_times[:-1] + stride_times[1:]) / 2
            valid = means > 0.01
            si_stride_time = np.mean((diffs[valid] / means[valid]) * 100) if np.any(valid) else 0

        # Angular Velocity Symmetry (Final Working Version)
        si_wx, si_wy, si_wz = 0.0, 0.0, 0.0
        if num_gc >= 2:
            wx_mags, wy_mags, wz_mags = [], [], []

            # 1. Convert to degrees if needed (remove if already in rad/s)
            gyr_deg = np.degrees(gyr)  # Only use if original data is in radians

            for i in range(num_gc - 1):
                start = HS[i] - HS[0]
                end = HS[i + 1] - HS[0] if (i + 1 < len(HS)) else len(gyr_deg[0])

                if 0 <= start < end <= len(gyr_deg[0]):
                    wx_mags.append(np.mean(np.abs(gyr_deg[0, start:end])))
                    wy_mags.append(np.mean(np.abs(gyr_deg[1, start:end])))
                    wz_mags.append(np.mean(np.abs(gyr_deg[2, start:end])))

            def dynamic_si(values):
                if len(values) < 2: return 0.0
                values = np.array(values)
                diffs = np.abs(np.diff(values))
                means = (values[:-1] + values[1:]) / 2

                # Dynamic threshold (25th percentile or 5° minimum)
                threshold = max(np.percentile(values, 25), 5.0)
                valid = means > threshold

                return np.mean((diffs[valid] / means[valid]) * 100) if np.any(valid) else 0.0

            si_wx = dynamic_si(wx_mags)
            si_wy = dynamic_si(wy_mags)
            si_wz = dynamic_si(wz_mags)

        # Add to results
        gait_features.extend([si_step, si_stride_time, si_wx, si_wy, si_wz])
        gait_feature_list.extend([
            f"{test_name}_step_length_symmetry",
            f"{test_name}_stride_time_symmetry",
            f"{test_name}_angular_vel_x_symmetry",
            f"{test_name}_angular_vel_y_symmetry",
            f"{test_name}_angular_vel_z_symmetry"
        ])

        # Kalman-filtered velocity computation    ## ISRAT
        velocity_x = np.power(self._apply_Kalman_Filter(self.acc[0, :]), 2)
        velocity_y = np.power(self._apply_Kalman_Filter(self.acc[1, :]), 2)
        velocity_z = np.power(self._apply_Kalman_Filter(self.acc[2, :]), 2)
        mag_vel = np.sqrt(velocity_x + velocity_y + velocity_z)

        # DEBUG: Compare with manual integration     ## ISRAT
        dt = 1 / self.fs  # time interval per sample
        v_x = np.cumsum(self.acc[0, :] * dt)
        v_y = np.cumsum(self.acc[1, :] * dt)
        v_z = np.cumsum(self.acc[2, :] * dt)
        mag_vel_manual = np.sqrt(v_x ** 2 + v_y ** 2 + v_z ** 2)
        print("Manual average velocity (m/s):", np.mean(mag_vel_manual))
        print("Kalman filter average velocity (m/s):", np.mean(mag_vel))

        time = np.arange(len(mag_vel_manual)) * dt  # Time axis based on sampling rate   ## ISRAT
        plt.plot(time, mag_vel_manual, label="Manually Integrated Velocity")
        plt.plot(time, mag_vel, label="Kalman Filtered Velocity", linestyle="dashed")
        plt.xlabel("Time (s)")
        plt.ylabel("Velocity (m/s)")
        plt.legend()
        plt.title("Velocity over Time")
        plt.show()

        fs = 100  # Sampling frequency
        acc = np.random.randn(3, 1000)
        gyr = np.random.randn(3, 1000)
        events = np.zeros(1000)
        unproc_acc = np.random.randn(3, 1000)
        fe = FeatureExtraction(acc, gyr, events, fs, unproc_acc)

        deviation = fe._get_deviation(acc[0])
        assert deviation >= 0, "Deviation should be non-negative"

        velocity = fe._compute_velocity(acc)
        assert velocity.shape == acc.shape, "Velocity shape mismatch"

        avg_vel, vel_mag = fe._get_avg_vel(velocity)
        assert avg_vel >= 0, "Average velocity should be non-negative"

        steps = fe._count_steps(acc[0])
        assert isinstance(steps, int) and steps >= 0, "Step count should be a non-negative integer"

        print("All tests passed!")

        return gait_features, gait_feature_list

    def get_flexibility_features(self):
        ...
        flex = 0
        flex_list = "max_gyr"
        return flex, flex_list

    def run(self, test, task):
        print(f"Extracting features...\n")
        feat = []
        feat_list = []
        if test == "gait" and task == "2min":
            feat, feat_list = self.get_gait_features(task=task)
        if test == "gait" and task == "10m":
            feat, feat_list = self.get_gait_features(task=task)
        if test == "sts" and task == "1min":
            feat, feat_list = self.get_STS_features(task=task)
        if test == "sts" and task == "5rep":
            feat, feat_list = self.get_STS_features(task=task)
        if test == "tug" and task == "single":
            feat, feat_list = self.get_TUG_features(task=task)
        if test == "tug" and task == "dual":
            #feat, feat_list = self.get_TUG_features(task=task)
            # Single-task processing
            single_features, single_features_list = self.get_TUG_features(task='single')
            # Calculate DTC
            dtc_values, dtc_names = self.calculate_DTC(single_features, single_features_list)
            return dtc_values, dtc_names
        if test == "flex":
            feat, feat_list = self.get_flexibility_features()
        print("\nNumeric feature value: \n", feat)
        print("\nname of the feature list: \n", feat_list)
        print("\n")
        return feat, feat_list
