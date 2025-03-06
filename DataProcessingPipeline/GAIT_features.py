from DataProcessingPipeline.FeatureExtraction import FeatureExtraction
import numpy as np
import matplotlib.pyplot as plt
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
        gyr = self.fe.gyr[:, HS[0]:MSw[-1]]
        duration = (MSw[-1] - HS[0]) / self.fe.fs

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
        velocity_x = np.power(self.fe.apply_Kalman_Filter(self.fe.acc[0, :]), 2)
        velocity_y = np.power(self.fe.apply_Kalman_Filter(self.fe.acc[1, :]), 2)
        velocity_z = np.power(self.fe.apply_Kalman_Filter(self.fe.acc[2, :]), 2)
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