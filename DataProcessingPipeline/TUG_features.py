from DataProcessingPipeline.FeatureExtraction import FeatureExtraction
import numpy as np
"""
we don't need inherit FeatureExtraction class rather we can create an instance of FeatureExtraction class to access it's attributes (composition principle-> "has-a" relationship)
class TUG_features(FeatureExtraction):
    def __init__(self, acc, gyr, events, fs, unproc_acc):
        # Initialize the parent FeatureExtraction class
        super().__init__(acc, gyr, events, fs, unproc_acc)
"""
class TUG_features:
    def __init__(self, feature_extractor: FeatureExtraction):
        self.fe = feature_extractor  # Store the FeatureExtraction instance

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
        event1 = np.where(self.fe.events == 1)[0][0]
        event2 = np.where(self.fe.events == 2)[0][0]
        event3 = np.where(self.fe.events == 3)[0][0]
        event4 = np.where(self.fe.events == 4)[0][0]
        event5 = np.where(self.fe.events == 5)[0][0]
        event6 = np.where(self.fe.events == 6)[0][0]
        event7 = np.where(self.fe.events == 7)[0][0]

        # Total test time (event1 to event7)
        total_t = (event7 - event1) / self.fe.fs
        tug_features.append(total_t)
        tug_feature_list.append(str(test_name) + "_total_time")

        # mean acc, peak acc
        mean_acc = np.mean(self.fe.mag_acc)
        max_acc = np.max(self.fe.mag_acc)
        tug_features.append(mean_acc)
        tug_feature_list.append(str(test_name) + "_mean_acc")
        tug_features.append(max_acc)
        tug_feature_list.append(str(test_name) + "_max_acc")

        # sit2stand and stand2sit time, mean acc, mean gyr
        t_sit2stand = (event2 - event1) / self.fe.fs
        t_stand2sit = (event7 - event6) / self.fe.fs
        mean_sitts_acc = np.mean(self.fe.mag_acc[event1:event2])
        mean_standts_acc = np.mean(self.fe.mag_acc[event6:event7])
        mean_sitts_gyr = np.mean(self.fe.mag_gyr[event1:event2])
        mean_standts_gyr = np.mean(self.fe.mag_gyr[event6:event7])

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
        ang_vel_sit2stand = self.fe.get_hip_angular_velocity(event1, event2)

        # Stand-to-sit phase angular velocity
        ang_vel_stand2sit = self.fe.get_hip_angular_velocity(event6, event7)

        # Append features (replacing the original implementation)
        tug_features.append(ang_vel_sit2stand)
        tug_feature_list.append(f"{test_name}_sit2stand_ang_vel")

        tug_features.append(ang_vel_stand2sit)
        tug_feature_list.append(f"{test_name}_stand2sit_ang_vel")

        # forward and return gait time and acc
        t_forward = (event3 - event2) / self.fe.fs
        t_return = (event5 - event4) / self.fe.fs

        # Pure walking time (event2 to event3 + event4 to event5)
        walking_time = t_forward + t_return  # Total walking time in seconds
        # walking_time = ((event3 - event2) + (event5 - event4)) / self.fs

        # TUG gait speed (Pure walking speed (excludes sit/stand/turn))
        pure_gait_speed = 6.0 / walking_time if walking_time > 0 else 0
        tug_features.append(pure_gait_speed)
        tug_feature_list.append(str(test_name) + "_gait_speed")

        # Calculate walking features -------------------------------------------------
        # Get accelerometer segments for walking phases
        forward_walk = self.fe.mag_acc[event2:event3]
        return_walk = self.fe.mag_acc[event4:event5]

        # Count steps in each phase
        forward_steps = self.fe.count_steps(forward_walk)
        return_steps = self.fe.count_steps(return_walk)
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
        turn_t = (event4 - event3) / self.fe.fs
        mean_acc_turn = np.mean(self.fe.mag_acc[event3:event4])
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
            jerk = self.fe.calculate_jerk(start, end)
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
