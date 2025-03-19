from DataProcessingPipeline.FeatureExtraction import FeatureExtraction
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid as cumtrapz
"""
we don't need inherit FeatureExtraction class rather we can create an instance of FeatureExtraction class to access it's attributes (composition principle-> "has-a" relationship)
class TUG_features(FeatureExtraction):
    def __init__(self, acc, gyr, events, fs, unproc_acc):
        # Initialize the parent FeatureExtraction class
        super().__init__(acc, gyr, events, fs, unproc_acc)
"""
class GAIT_features:
    def __init__(self, feature_extractor: FeatureExtraction):
        self.fe = feature_extractor  # Store the FeatureExtraction instance

    def get_gait_features(self, task="2min"):
        test_name = str(task) + "_gait"
        gait_features = []
        gait_feature_list = []

        # seven events - HS, LP, MSt, TSt, PSw, TO, MSw
        HS = np.where(self.fe.events == 1)[0]
        MSt = np.where(self.fe.events == 3)[0]
        TSt = np.where(self.fe.events == 4)[0]
        PSw = np.where(self.fe.events == 5)[0]
        TO = np.where(self.fe.events == 6)[0]
        MSw = np.where(self.fe.events == 7)[0]

        # num of gait cycle
        num_gc = np.min([len(HS), len(MSw)])
        gait_features.append(num_gc)
        gait_feature_list.append(str(test_name) + "_num_gc")
        mag_acc = self.fe.mag_acc[HS[0]:MSw[-1]]
        acc = self.fe.acc[:, HS[0]:MSw[-1]]
        print(np.shape(acc))
        # mag_gyr = self.fe.mag_gyr[HS[0]:MSw[-1]]
        gyr = self.fe.gyr[:, HS[0]:MSw[-1]]
        duration = (MSw[-1] - HS[0]) / self.fe.fs
        # duration = 120
        gait_features.append(duration)
        gait_feature_list.append(str(test_name) + "_duration")

        # num steps
        n_steps = num_gc * 2
        gait_features.append(n_steps)
        gait_feature_list.append(str(test_name) + "_n_steps")

        steps_peaks = self.fe.count_steps(mag_acc)  # Use magnitude signal

        # Use agreement or weighted average
        if abs(n_steps - steps_peaks) <= 2:
            final_steps = (n_steps + steps_peaks) // 2
        else:
            final_steps = steps_peaks  # Fallback to peaks

        gait_features.append(final_steps)
        gait_feature_list.append(str(test_name) + "_final_steps")

        # Calculate total distance and gait speed
        walking_times, turning_times = self.fe.detect_turns(self.fe.mag_acc, self.fe.mag_gyr)

        t_straight = np.mean(walking_times)  # Average time spent walking straight
        t_turn = np.mean(turning_times)  # Average time spent turning

        # Check for empty lists to prevent errors
        if walking_times and turning_times:
            total_distance = self.fe.estimate_total_distance_v2(120, t_straight, t_turn)
            gait_speed = total_distance / 120  # Speed = Distance / Time
        else:
            total_distance = 0
            gait_speed = 0

        print("Total Distance Walked:", total_distance)
        print("Gait Speed:", gait_speed)

        gait_features.append(total_distance)
        gait_feature_list.append(str(test_name) + "_total_distance")

        gait_speed = total_distance / 120  # Speed = Distance / Time
        gait_features.append(gait_speed)
        gait_feature_list.append(str(test_name) + "_gait_speed")

        #print("Accel Magnitude:", self.fe.mag_acc[:30])  # First 10 values for inspection
        #print("Gyro Magnitude:", self.fe.mag_gyr[:30])  # First 10 values for inspection

        # Select a window for zooming in (e.g., 2 to 4 seconds)
        start_time = 50  # in seconds
        end_time = 110  # in seconds
        start_idx = int(start_time * self.fe.fs)
        end_idx = int(end_time * self.fe.fs)

        # Plot only the selected time window
        plt.figure(figsize=(12, 6))

        # Accelerometer Magnitude Plot
        plt.subplot(2, 1, 1)
        plt.plot(np.linspace(start_time, end_time, end_idx - start_idx), self.fe.mag_acc[start_idx:end_idx], color='b')
        plt.title(f'Magnitude of Accelerometer (Zoomed In: {start_time}-{end_time} sec)')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Magnitude')
        plt.grid()

        # Gyroscope Magnitude Plot
        plt.subplot(2, 1, 2)
        plt.plot(np.linspace(start_time, end_time, end_idx - start_idx), self.fe.mag_gyr[start_idx:end_idx], color='r')
        plt.title(f'Magnitude of Gyroscope (Zoomed In: {start_time}-{end_time} sec)')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Magnitude')
        plt.grid()

        plt.tight_layout()
        plt.show()

        # Plot the magnitude of accelerometer data
        plt.subplot(2, 1, 1)
        plt.plot(self.fe.mag_acc, label='Magnitude of Accelerometer (Acc)', color='b')
        plt.title('Magnitude of Accelerometer (Acc) with Walking and Turning Phases')
        plt.xlabel('Time (samples)')
        plt.ylabel('Magnitude')
        plt.legend()

        # Plot the magnitude of gyroscope data
        plt.subplot(2, 1, 2)
        plt.plot(self.fe.mag_gyr, label='Magnitude of Gyroscope (Gyro)', color='r')
        plt.title('Magnitude of Gyroscope (Gyro) for All Phases')
        plt.xlabel('Time (samples)')
        plt.ylabel('Magnitude')
        plt.legend()

        plt.tight_layout()
        plt.show()

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

        # Compute Velocity
        # velocity_x = np.power(self.fe.apply_Kalman_Filter(self.fe.acc[0, :]), 2)
        # velocity_y = np.power(self.fe.apply_Kalman_Filter(self.fe.acc[1, :]), 2)
        # velocity_z = np.power(self.fe.apply_Kalman_Filter(self.fe.acc[2, :]), 2)
        # mag_vel = np.sqrt(velocity_x + velocity_y + velocity_z)
        velocity_x = cumtrapz(self.fe.acc[0, :], initial=0)  # Integrate accelerometer data to get velocity
        velocity_y = cumtrapz(self.fe.acc[1, :], initial=0)
        velocity_z = cumtrapz(self.fe.acc[2, :], initial=0)
        mag_vel = np.sqrt(velocity_x**2 + velocity_y**2 + velocity_z**2) + 1e-6
        #avg_vel = np.mean(mag_vel) * 10
        avg_vel = gait_speed
        gait_features.append(avg_vel)
        gait_feature_list.append(str(test_name) + "_avg_vel")

        # Plot velocity magnitude
        plt.plot(mag_vel)
        plt.title('Velocity Magnitude')
        plt.xlabel('Time Step')
        plt.ylabel('Velocity (m/s)')
        plt.show()

        # Check ranges of mag_acc and mag_vel
        print(f"mag_acc min: {np.min(mag_acc)}, max: {np.max(mag_acc)}")
        print(f"mag_vel min: {np.min(mag_vel)}, max: {np.max(mag_vel)}")

        # Plot mag_vel to see if it makes sense
        plt.plot(mag_vel, label="Magnitude of Velocity")
        plt.xlabel('Time (samples)')
        plt.ylabel('Velocity (m/s)')
        plt.title('Magnitude of Velocity Over Time')
        plt.legend()
        plt.show()


        # Compute Power
        mag_acc = (self.fe.mag_acc - np.min(self.fe.mag_acc)) / (np.max(self.fe.mag_acc) - np.min(self.fe.mag_acc))
        mag_vel = (mag_vel - np.min(mag_vel)) / (np.max(mag_vel) - np.min(mag_vel))

        # Add a small constant to avoid multiplication by zero
        epsilon = 1e-6  # Small value to avoid multiplication by zero
        mag_vel = np.maximum(mag_vel, epsilon)  # Ensure no value is too close to zero

        print(f"mag_acc std: {np.std(mag_acc)}")
        print(f"mag_vel std: {np.std(mag_vel)}")

        #power = np.abs(mag_acc * mag_vel)
        power = (mag_acc * mag_vel) / 1e3  # Scaling down if necessary
        print("Power Values:", power[:10])  # Print the first 10 values of power for inspection
        # # Normalize power to range [0, 1] for better variation handling
        # power = (power - np.min(power)) / (np.max(power) - np.min(power))

        def moving_average(data, window_size=5):
            return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

        # Apply moving average to mag_vel
        smoothed_mag_vel = moving_average(mag_vel)

        # Now recalculate power with the smoothed mag_vel
        power = (mag_acc[:len(smoothed_mag_vel)] * smoothed_mag_vel) / 1e3

        # Check power to see its fluctuation
        plt.plot(power, label='Power')
        plt.axhline(y=np.max(power), color='r', linestyle='--', label='Peak Power')
        plt.axhline(y=np.min(power), color='b', linestyle='--', label='Min Power')
        plt.xlabel('Time (samples)')
        plt.ylabel('Power')
        plt.title('Power Over Time')
        plt.legend()
        plt.show()
        #min_length = min(len(mag_acc), len(mag_vel))
        #power = mag_acc[:min_length] * mag_vel[:min_length]
        peak_power = np.max(power)
        mean_power = np.mean(power)
        mean_power = max(mean_power, 1e-6)  # Prevent extremely small values
        std_power = np.std(power)
        cv = (std_power / mean_power) * 100  # Coefficient of variation in percentage
        fatigue_index = cv
        # # Example alternate fatigue index calculation
        # if menn_power > 0:  # Ensure we don't divide by zero
        #     fatigue_index = (peak_power - menn_power) / peak_power * 100
        # else:
        #     fatigue_index = 0  # Handle the case where min_power is zero
        # fatigue_index = ((peak_power - menn_power) / peak_power) * 100
        print(f"Peak Power: {peak_power}, Min Power: {mean_power}")
        # fatigue_index = ((peak_power - min_power) / peak_power) * 100 if peak_power != 0 else 0

        print(f"mag_acc min: {np.min(mag_acc)}, max: {np.max(mag_acc)}")
        print(f"mag_vel min: {np.min(mag_vel)}, max: {np.max(mag_vel)}")
        print(f"Power min: {np.min(power)}, max: {np.max(power)}")

        gait_features.append(fatigue_index)
        gait_feature_list.append(str(test_name) + "_fatigue_index")

        plt.figure(figsize=(10, 5))
        plt.plot(power, label='Power', color='g')
        plt.axhline(y=peak_power, color='r', linestyle='--', label='Peak Power')
        plt.axhline(y=mean_power, color='b', linestyle='--', label='Mean Power')
        plt.xlabel('Time (samples)')
        plt.ylabel('Power')
        plt.title('Power Over Time')
        plt.legend()
        plt.grid()
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
        stride_times = (HS[1:num_gc] - HS[:num_gc - 1]) / self.fe.fs
        add_variability_metrics(stride_times, "stride_time")

        # Step length variability    ### ISRAT
        step_lengths = ((TSt[:num_gc] - HS[:num_gc]) / self.fe.fs * avg_vel)
        add_variability_metrics(step_lengths, "step_length")

        # Stride length variability     ### ISRAT
        stride_lengths = ((HS[1:num_gc] - HS[:num_gc - 1]) / self.fe.fs * avg_vel)
        add_variability_metrics(stride_lengths, "stride_length")

        # distance
        #distance = duration * avg_vel
        distance = total_distance
        #gait_features.append(distance)
        #gait_feature_list.append(str(test_name) + "_distance")

        # average step length - step len = TSt - HS
        avg_step_len = np.mean(((TSt[0:num_gc] - HS[0:num_gc]) / self.fe.fs) * avg_vel)
        gait_features.append(avg_step_len)
        gait_feature_list.append(str(test_name) + "_avg_step_len")

        # average stride length - stride len = HS - HS
        avg_stride_len = np.mean(((HS[1:num_gc] - HS[0:num_gc - 1]) / self.fe.fs) * avg_vel)
        gait_features.append(avg_stride_len)
        gait_feature_list.append(str(test_name) + "_avg_stride_len")

        # stance
        stance = (TO[0:num_gc] - HS[0:num_gc]) / self.fe.fs
        mean_stance = np.mean(stance)
        stance_per = (mean_stance / duration) * 100
        gait_features.append(mean_stance)
        gait_feature_list.append(str(test_name) + "_mean_stance")
        gait_features.append(stance_per)
        gait_feature_list.append(str(test_name) + "_stance_per")

        # Swing
        swing = (HS[1:num_gc] - TO[0:num_gc - 1]) / self.fe.fs
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
        ds_time1 = (MSt[0:num_gc] - HS[0:num_gc]) / self.fe.fs
        ds_time2 = (TO[0:num_gc] - TSt[0:num_gc]) / self.fe.fs
        ds_time = ds_time1 + ds_time2
        mean_double_support = np.mean(ds_time)
        gait_features.append(mean_double_support)
        gait_feature_list.append(str(test_name) + "_mean_double_support")
        ds_per = (mean_double_support / duration) * 100
        gait_features.append(ds_per)
        gait_feature_list.append(str(test_name) + "_ds_per")

        # ========== Symmetry Index Calculations ==========
        # Step Length Symmetry        #### ISRAT
        step_lens = ((TSt[0:num_gc] - HS[0:num_gc]) / self.fe.fs) * avg_vel
        si_step = 0.0
        if len(step_lens) >= 2:
            step_lens = np.array(step_lens)  # Convert to numpy array
            diffs = np.abs(np.diff(step_lens))
            means = (step_lens[:-1] + step_lens[1:]) / 2
            valid = means > 0.001
            si_step = np.mean((diffs[valid] / means[valid]) * 100) if np.any(valid) else 0

        # Stride Time Symmetry
        stride_times = (HS[1:num_gc] - HS[0:num_gc - 1]) / self.fe.fs
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

        # # Kalman-filtered velocity computation    ## ISRAT
        # velocity_x = np.power(self.fe.apply_Kalman_Filter(self.fe.acc[0, :]), 2)
        # velocity_y = np.power(self.fe.apply_Kalman_Filter(self.fe.acc[1, :]), 2)
        # velocity_z = np.power(self.fe.apply_Kalman_Filter(self.fe.acc[2, :]), 2)
        # mag_vel = np.sqrt(velocity_x + velocity_y + velocity_z)
        #
        # # DEBUG: Compare with manual integration     ## ISRAT
        # dt = 1 / self.fe.fs  # time interval per sample
        # v_x = np.cumsum(self.fe.acc[0, :] * dt)
        # v_y = np.cumsum(self.fe.acc[1, :] * dt)
        # v_z = np.cumsum(self.fe.acc[2, :] * dt)
        # mag_vel_manual = np.sqrt(v_x ** 2 + v_y ** 2 + v_z ** 2)
        # print("Manual average velocity (m/s):", np.mean(mag_vel_manual))
        # print("Kalman filter average velocity (m/s):", np.mean(mag_vel))
        #
        # time = np.arange(len(mag_vel_manual)) * dt  # Time axis based on sampling rate   ## ISRAT
        # plt.plot(time, mag_vel_manual, label="Manually Integrated Velocity")
        # plt.plot(time, mag_vel, label="Kalman Filtered Velocity", linestyle="dashed")
        # plt.xlabel("Time (s)")
        # plt.ylabel("Velocity (m/s)")
        # plt.legend()
        # plt.title("Velocity over Time")
        # plt.show()
        #
        # fs = 100  # Sampling frequency
        # acc = np.random.randn(3, 1000)
        # gyr = np.random.randn(3, 1000)
        # events = np.zeros(1000)
        # unproc_acc = np.random.randn(3, 1000)
        # fe = FeatureExtraction(acc, gyr, events, fs, unproc_acc)
        #
        # deviation = fe.get_deviation(acc[0])
        # assert deviation >= 0, "Deviation should be non-negative"
        #
        # velocity = fe.compute_velocity(acc)
        # assert velocity.shape == acc.shape, "Velocity shape mismatch"
        #
        # avg_vel, vel_mag = fe.get_avg_vel(velocity)
        # assert avg_vel >= 0, "Average velocity should be non-negative"
        #
        # steps = fe.count_steps(acc[0])
        # assert isinstance(steps, int) and steps >= 0, "Step count should be a non-negative integer"
        #
        # print("All tests passed!")

        return gait_features, gait_feature_list