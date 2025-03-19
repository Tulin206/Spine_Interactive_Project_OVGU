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
        t_turn = np.median(turning_times)  # Average time spent turning

        # Check for empty lists to prevent errors
        if walking_times and turning_times:
            total_distance = self.fe.estimate_total_distance_v2(120, t_straight, t_turn)
            gait_speed = total_distance / 120  # Speed = Distance / Time
        else:
            total_distance = 0
            gait_speed = 0

        print("Total Distance Walked:", total_distance)
        print("Gait Speed:", gait_speed)

        avg_vel = gait_speed

        gait_features.append(total_distance)
        gait_feature_list.append(str(test_name) + "_total_distance")

        gait_speed = total_distance / 120  # Speed = Distance / Time
        gait_features.append(gait_speed)
        gait_feature_list.append(str(test_name) + "_gait_speed")

        # Apply Kalman filter to accel_mag and gyro_mag
        #smoothed_accel_mag = self.fe.moving_average(self.fe.mag_acc)
        #smoothed_gyro_mag = self.fe.moving_average(self.fe.mag_gyr)

        #print("Filtered Accel Magnitude:", smoothed_accel_mag[:10])  # First 10 values for inspection
        #print("Filtered Gyro Magnitude:", smoothed_gyro_mag[:10])  # First 10 values for inspection

        print("Accel Magnitude:", self.fe.mag_acc[:30])  # First 10 values for inspection
        print("Gyro Magnitude:", self.fe.mag_gyr[:30])  # First 10 values for inspection

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

        velocity_x = np.power(self.fe.apply_Kalman_Filter(self.fe.acc[0, :]), 2)
        velocity_y = np.power(self.fe.apply_Kalman_Filter(self.fe.acc[1, :]), 2)
        velocity_z = np.power(self.fe.apply_Kalman_Filter(self.fe.acc[2, :]), 2)
        mag_vel = np.sqrt(velocity_x + velocity_y + velocity_z)
        #avg_vel = np.mean(mag_vel) * 10
        #gait_features.append(avg_vel)
        #gait_feature_list.append(str(test_name) + "_avg_vel")


        """t = np.arange(0, len(acc[0,:]), 1)
        velocity_x = np.power(self._get_velocity(self.acc[0, :]), 2)
        velocity_y = np.power(self._get_velocity(self.acc[1, :]), 2)
        velocity_z = np.power(self._get_velocity(self.acc[2, :]), 2)"""
        """
        orientations = self.fe.get_orientation(gyr)
        acc_global = self.fe.transform_acc(acc, orientations)
        acc_corrected = self.fe.remove_gravity(acc_global)
        velocity = self.fe.compute_velocity(acc_corrected)
        avg_velocity = self.fe.get_avg_vel(velocity)
        gait_features.append(avg_velocity)
        gait_feature_list.append(str(test_name) + "_avg_velocity")  ## Gait_Speed

        print("Raw Acceleration:", acc[:5])
        print("Gravity Removed Acceleration:", acc_corrected[:5])
        print("Velocity:", velocity[:5])
        print("Average Velocity:", avg_velocity)
        print("Final Velocity:", velocity[-5:])  # Check for drift
        print("Mean Velocity:", np.mean(velocity, axis=0))  # Check expected range
        
        
         # Step 1: Apply Kalman filter to get velocity components
        velocity_x = self.fe.apply_Kalman_Filter(self.fe.acc[0, :])
        velocity_y = self.fe.apply_Kalman_Filter(self.fe.acc[1, :])
        velocity_z = self.fe.apply_Kalman_Filter(self.fe.acc[2, :])

        # Step 2: Calculate the magnitude of velocity
        mag_vel = np.sqrt(velocity_x ** 2 + velocity_y ** 2 + velocity_z ** 2)

        # Step 3: Integrate velocity to find total displacement
        dt = 1.0 / self.fe.fs  # Time step

        # Calculate total displacement using np.trapz
        try:
            total_displacement = np.trapz(mag_vel, dx=dt)
        except AttributeError:
            # If np.trapz is not available, use manual integration
            total_displacement = 0
            for i in range(1, len(mag_vel)):
                total_displacement += 0.5 * (mag_vel[i] + mag_vel[i - 1]) * dt

        # Step 4: Calculate total time (assuming consistent time step)
        total_time = len(mag_vel) * dt

        # Step 5: Calculate average velocity
        avg_vel = total_displacement / total_time

        # Store the result
        gait_features.append(avg_vel)
        gait_feature_list.append(str(test_name) + "_avg_vel")
    
        # Step 1: Compute orientation using gyroscope data
        orientations = self.fe.get_orientation(self.fe.gyr)  # Get orientations from gyro data

        # Step 2: Transform accelerometer data from local to global coordinates
        acc_global = self.fe.transform_acc(self.fe.acc, orientations)

        # Step 3: Remove gravity from global acceleration
        acc_data_no_gravity = self.fe.remove_gravity(acc_global)

        velocity = self.fe.compute_velocity(acc_data_no_gravity)
        avg_vel, _ = self.fe.get_avg_vel(velocity)
        gait_features.append(avg_vel)
        gait_feature_list.append(str(test_name) + "_avg_vel")
    """
        # Step 1: Remove gravity
        #acc_data_no_gravity_x = self.fe.remove_gravity(acc[0, :])
        #acc_data_no_gravity_y = self.fe.remove_gravity(acc[1, :])
        #acc_data_no_gravity_z = self.fe.remove_gravity(acc[2, :])

        # Step 2: Apply Kalman filter to estimate velocity (for each axis)
        #velocity_x = self.fe.apply_Kalman_Filter(acc_data_no_gravity[0, :])  # Apply Kalman filter to x-axis
        #velocity_y = self.fe.apply_Kalman_Filter(acc_data_no_gravity[1, :])  # Apply Kalman filter to y-axis
        #velocity_z = self.fe.apply_Kalman_Filter(acc_data_no_gravity[2, :])  # Apply Kalman filter to z-axis

        # Step 3: Compute the magnitude of the velocity
        #mag_vel = np.sqrt(np.power(velocity_x, 2) + np.power(velocity_y, 2) + np.power(velocity_z, 2))

        #velocity_x = np.power(self.fe.apply_Kalman_Filter(self.fe.acc[0, :]), 2)
        #velocity_y = np.power(self.fe.apply_Kalman_Filter(self.fe.acc[1, :]), 2)
        #velocity_z = np.power(self.fe.apply_Kalman_Filter(self.fe.acc[2, :]), 2)
        #mag_vel = np.sqrt(velocity_x + velocity_y + velocity_z)

        # Step 4: Compute the average velocity
        #avg_vel = np.mean(mag_vel) * 10
        #avg_vel = np.mean(mag_vel)        # ISRAT
        #gait_features.append(avg_vel)
        #gait_feature_list.append(str(test_name) + "_avg_vel")  ## Gait_Speed

        # Plot velocity magnitude
        plt.plot(mag_vel)
        plt.title('Velocity Magnitude')
        plt.xlabel('Time Step')
        plt.ylabel('Velocity (m/s)')
        plt.show()

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
                    # velocities_per_cycle.append(np.mean(cycle_vel) * 10)
                    velocities_per_cycle.append(np.mean(cycle_vel))

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
        stride_times = (HS[1:num_gc] - HS[:num_gc - 1]) / self.fe.fs
        add_variability_metrics(stride_times, "stride_time")

        # Step length variability    ### ISRAT
        step_lengths = ((TSt[:num_gc] - HS[:num_gc]) / self.fe.fs * avg_vel)
        add_variability_metrics(step_lengths, "step_length")

        # Stride length variability     ### ISRAT
        stride_lengths = ((HS[1:num_gc] - HS[:num_gc - 1]) / self.fe.fs * avg_vel)
        add_variability_metrics(stride_lengths, "stride_length")

        # distance
        distance = duration * avg_vel
        gait_features.append(distance)
        gait_feature_list.append(str(test_name) + "_distance")

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

        """
        # **Gait Speed Calculation**     ISRAT
        total_time = duration
        #total_distance = n_steps * np.mean(step_lengths)  # Total distance walked    # Ensure step_lengths is a single value
        #gait_speed = total_distance / total_time  # Speed in meters per second (m/s)
        gait_speed = distance / total_time  # Speed in meters per second (m/s)

        gait_features.append(gait_speed)
        gait_feature_list.append(str(test_name) + "_gait_speed")
        """
        """
        walking_distance = 6
        turning_distance = 1
        # Distance covered per 2 steps (gait cycle)
        distance_per_step = (walking_distance + turning_distance)/2  # in meters = 3.5

        # Total distance covered (in meters)
        #total_dist = n_steps * distance_per_step

        # Total distance covered (in meters)
        total_dist = n_steps * step_lengths

        # Time duration of the walk (in seconds)
        time_duration = 2 * 60  # 2 minutes in seconds

        # Gait speed (in meters per second)
        gait_speed_test = total_dist / time_duration

        # Append the gait speed to the features list
        gait_features.append(gait_speed_test)
        gait_feature_list.append(str(test_name) + "_gait_speed_test")
        """

        #gait_features.append(total_distance)
        #gait_feature_list.append(str(test_name) + "_total_distance")

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

        # Kalman-filtered velocity computation    ## ISRAT
        velocity_x = np.power(self.fe.apply_Kalman_Filter(self.fe.acc[0, :]), 2)
        velocity_y = np.power(self.fe.apply_Kalman_Filter(self.fe.acc[1, :]), 2)
        velocity_z = np.power(self.fe.apply_Kalman_Filter(self.fe.acc[2, :]), 2)
        mag_vel = np.sqrt(velocity_x + velocity_y + velocity_z)

        # DEBUG: Compare with manual integration     ## ISRAT
        dt = 1 / self.fe.fs  # time interval per sample
        v_x = np.cumsum(self.fe.acc[0, :] * dt)
        v_y = np.cumsum(self.fe.acc[1, :] * dt)
        v_z = np.cumsum(self.fe.acc[2, :] * dt)
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

        deviation = fe.get_deviation(acc[0])
        assert deviation >= 0, "Deviation should be non-negative"

        velocity = fe.compute_velocity(acc)
        assert velocity.shape == acc.shape, "Velocity shape mismatch"

        avg_vel, vel_mag = fe.get_avg_vel(velocity)
        assert avg_vel >= 0, "Average velocity should be non-negative"

        steps = fe.count_steps(acc[0])
        assert isinstance(steps, int) and steps >= 0, "Step count should be a non-negative integer"

        print("All tests passed!")

        return gait_features, gait_feature_list