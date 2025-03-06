from DataProcessingPipeline.FeatureExtraction import FeatureExtraction
import numpy as np
"""
we don't need inherit FeatureExtraction class rather we can create an instance of FeatureExtraction class to access it's attributes (composition principle-> "has-a" relationship)
class TUG_features(FeatureExtraction):
    def __init__(self, acc, gyr, events, fs, unproc_acc):
        # Initialize the parent FeatureExtraction class
        super().__init__(acc, gyr, events, fs, unproc_acc)
"""
class STS_features:
    def __init__(self, feature_extractor: FeatureExtraction):
        self.fe = feature_extractor  # Store the FeatureExtraction instance

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
        sit2stand_all = np.where(self.fe.events == 1)[0]
        stand2sit_all = np.where(self.fe.events == 2)[0]
        rep = np.min([len(sit2stand_all), len(stand2sit_all)])
        print("repetition: ", rep)
        sit2stand = sit2stand_all[0:rep]
        stand2sit = stand2sit_all[0:rep]

        # Total time taken to perform the 5 reps
        total_t = (stand2sit[-1] - sit2stand[0]) / self.fe.fs
        sts_features.append(total_t)
        sts_feature_list.append(str(test_name) + "_total_time")
        print(f"\nTotal time for 5 repetitions: {total_t:.2f} seconds")

        # Duration of each repetition
        t_rep = np.subtract(stand2sit, sit2stand) / self.fe.fs
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
        standts_acc_rep = []

        # List to store angular velocity for sit to stand and stand to sit for each repetition
        sitts_gyr_rep = []
        standts_gyr_rep = []

        for r in range(rep):
            peak_acc_rep.append(np.max(self.fe.mag_acc[sit2stand[r]:stand2sit[r]]))
            peak_gyr_rep.append(np.max(self.fe.mag_gyr[sit2stand[r]:stand2sit[r]]))

            # Compute mean acceleration for this repetition         # ISRAT
            mean_sitts_acc_rep = np.mean(self.fe.mag_acc[sit2stand[r]:stand2sit[r]])
            sitts_acc_rep.append(mean_sitts_acc_rep)
            # Ensure valid range for the next repetition (r + 1 should not exceed rep - 1)
            if r < rep - 1:
                mean_standts_acc_rep = np.nanmean(self.fe.mag_acc[stand2sit[r + 1]:sit2stand[r + 1]])  # Handle NaN values
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
            # sts_features.append(sym_acc)
            # sts_feature_list.append(f"{test_name}_sym_acc_rep{r + 1}")

            # Compute mean angular velocity for this repetition         # ISRAT
            mean_sitts_gyr_rep = np.mean(self.fe.mag_gyr[stand2sit[r]] - self.fe.mag_gyr[sit2stand[r]])
            sitts_gyr_rep.append(mean_sitts_gyr_rep)
            mean_standts_gyr_rep = (
                np.mean(self.fe.mag_gyr[stand2sit[r + 1]] - self.fe.mag_gyr[sit2stand[r + 1]])
                if r < rep - 1 else np.nan
            )
            standts_gyr_rep.append(mean_standts_gyr_rep)

            # Check symmetry for angular velocity        # ISRAT
            sym_gyr = 1
            if mean_standts_gyr_rep and (mean_sitts_gyr_rep / mean_standts_gyr_rep) > 0.1:
                sym_gyr = 0

            # Append symmetry (asymmetry) for angular velocity
            sym_gyr_rep.append(sym_gyr)
            # sts_features.append(sym_gyr)
            # sts_feature_list.append(f"{test_name}_sym_gyr_rep{r + 1}")

            # Extract the acceleration and gyroscope data for each repetition    # ISRAT
            acc_data = self.fe.mag_acc[sit2stand[r]:stand2sit[r]]
            gyr_data = self.fe.mag_gyr[sit2stand[r]:stand2sit[r]]

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
        t_sd = self.fe.get_deviation(t_rep)
        t_cv = (t_sd / mean_t_rep) * 100  # Coefficient of variation    # ISRAT
        peak_acc_sd = self.fe.get_deviation(peak_acc_rep)
        peak_acc_cv = (peak_acc_sd / np.mean(peak_acc_rep)) * 100  # Coefficient of variation   ## ISRAT
        peak_gyr_sd = self.fe.get_deviation(peak_gyr_rep)
        peak_gyr_cv = (peak_gyr_sd / np.mean(peak_gyr_rep)) * 100  # Coefficient of variation   ## ISRAT

        # Append the variability features
        sts_features.append(t_sd)
        sts_feature_list.append(str(test_name) + "_sd_time")
        sts_features.append(t_cv)
        sts_feature_list.append(str(test_name) + "_cv_time")
        sts_features.append(peak_acc_sd)
        sts_feature_list.append(str(test_name) + "_sd_acc")
        sts_features.append(peak_acc_cv)  ## ISRAT
        sts_feature_list.append(str(test_name) + "_cv_acc")  ## ISRAT
        sts_features.append(peak_gyr_sd)
        sts_feature_list.append(str(test_name) + "_sd_gyr")
        sts_features.append(peak_gyr_cv)  ## ISRAT
        sts_feature_list.append(str(test_name) + "_cv_gyr")  ## ISRAT

        # mean acc, max acc
        mean_acc = np.mean(self.fe.mag_acc)
        max_acc = np.max(self.fe.mag_acc)
        sts_features.append(mean_acc)
        sts_feature_list.append(str(test_name) + "_mean_acc")
        sts_features.append(max_acc)
        sts_feature_list.append(str(test_name) + "_max_acc")

        # mean acc and mean gyr of sittstand and standtsit
        mean_sitts_acc = np.mean(self.fe.mag_acc[stand2sit] - self.fe.mag_acc[sit2stand])
        mean_standts_acc = np.mean(self.fe.mag_acc[stand2sit[1:]] - self.fe.mag_acc[sit2stand[:-1]])
        mean_sitts_gyr = np.mean(self.fe.mag_gyr[stand2sit] - self.fe.mag_gyr[sit2stand])
        mean_standts_gyr = np.mean(self.fe.mag_gyr[stand2sit[1:]] - self.fe.mag_gyr[sit2stand[:-1]])
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
        mean_sitts_gyr = np.mean(self.fe.mag_gyr[stand2sit] - self.fe.mag_gyr[sit2stand])
        mean_standts_gyr = np.mean(self.fe.mag_gyr[stand2sit[1:]] - self.fe.mag_gyr[sit2stand[:-1]])

        # Check symmetry for angular velocity (similar to the acceleration symmetry check)       # ISRAT
        sym_gyr = 1
        if (mean_sitts_gyr / mean_standts_gyr) > 0.1:
            # if the ratio is > 10%, consider it asymmetric
            sym_gyr = 0

        # Append symmetry (asymmetry) for angular velocity         # ISRAT
        sts_features.append(sym_gyr)
        sts_feature_list.append(str(test_name) + "_sym_gyr")

        # angular displacement
        ang_disp = self.fe.get_angular_displacement(self.fe.gyr, self.fe.acc)
        f_rep_disp = self.fe.get_angular_displacement(self.fe.gyr[:, sit2stand[0]:stand2sit[0]],
                                                    self.fe.acc[:, sit2stand[0]:stand2sit[0]])
        l_rep_disp = self.fe.get_angular_displacement(self.fe.gyr[:, sit2stand[-1]:stand2sit[-1]],
                                                    self.fe.acc[:, sit2stand[-1]:stand2sit[-1]])
        sts_features.append(ang_disp)
        sts_feature_list.append(str(test_name) + "_ang_disp")
        sts_features.append(f_rep_disp)
        sts_feature_list.append(str(test_name) + "_f_rep_disp")
        sts_features.append(l_rep_disp)
        sts_feature_list.append(str(test_name) + "_l_rep_disp")

        # power calculation
        mean_power_acc = np.mean(
            (np.power(self.fe.acc[0, :], 2) + np.power(self.fe.acc[1, :], 2) + np.power(self.fe.acc[2, :], 2)) / len(
                self.fe.mag_acc))
        mean_power_gyr = np.mean(
            (np.power(self.fe.gyr[0, :], 2) + np.power(self.fe.gyr[1, :], 2) + np.power(self.fe.gyr[2, :], 2)) / len(
                self.fe.mag_gyr))

        # Store power in features
        sts_features.append(mean_power_acc)
        sts_feature_list.append(str(test_name) + "_mean_power_acc")
        sts_features.append(mean_power_gyr)
        sts_feature_list.append(str(test_name) + "_mean_power_gyr")

        # **New Feature**: Differences between the first and last repetition for fatigue detection  # ISRAT
        decrease = self.fe.get_decrease(self.fe.mag_acc, rep, sit2stand, stand2sit)
        sts_features.append(decrease)
        sts_feature_list.append(str(test_name) + "_fatigue")

        if task == "1min":
            # Decrease in acceleration - difference between first 3 rep and last 3 rep
            dec_acc = self.fe.get_decrease(self.fe.mag_acc, rep, sit2stand, stand2sit)
            dec_gyr = self.fe.get_decrease(self.fe.mag_gyr, rep, sit2stand, stand2sit)
            sts_features.append(dec_acc)
            sts_feature_list.append(str(test_name) + "_dec_acc")
            sts_features.append(dec_gyr)
            sts_feature_list.append(str(test_name) + "_dec_gyr")

        return sts_features, sts_feature_list