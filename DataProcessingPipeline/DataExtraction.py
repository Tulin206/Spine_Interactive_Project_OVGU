"""
In this project we analyzed data obtained from gait signals during selected tasks and cognitive task scores
This python file reads the data from files and transforms the signals from smartphones local system
to world's global coordinate system.
Dataset is obtained from smartphone and stored in the files. What is the format of the files?
Signal Data files: contain accelerometer, gyroscope and magnetometer data
Cognitive files: tabular data
run() function returns acc and gyro signals as numpy array, cognitive task scores and class labels?
"""

from os import path
import numpy as np
import pandas as pd


class DataExtraction:
    def __init__(self, data_dir):
        self.sp_data_dir = data_dir['phone']
        self.enode_data_dir = data_dir['enode']

    def read2minData(self, dir, fname, rep, ext=".xls"):
        if rep is None:
            full_path = dir + fname + ext
        else:
            full_path = dir + fname + rep + ext

        if ext == ".csv" or ext == ".tsv":
            df = pd.read_csv(full_path, sep='\t', decimal=",")

            if 'Event' not in df:
                df['Event'] = 0

            acc = np.transpose(np.concatenate((df[["ax"]].to_numpy(), df[["ay"]].to_numpy(),
                                               df[["az"]].to_numpy(), df[["Event"]].to_numpy()), axis=1))
            gyr = 0

        else:
            df_acc = pd.read_excel(full_path, sheet_name="Linear Accelerometer")
            df_gyr = pd.read_excel(full_path, sheet_name="Gyroscope")

            if 'events' not in df_acc:
                df_acc['events'] = 0

            if 'events' not in df_gyr:
                df_gyr['events'] = 0


            print ("X_direction\n", df_acc[["X (m/s^2)"]].to_numpy())
            print("\n\nY_direction\n", df_acc[["Y (m/s^2)"]].to_numpy())
            print("\n\nZ_direction\n", df_acc[["Z (m/s^2)"]].to_numpy())
            print("\n\nevents\n", df_acc[['events']])

            acc_before_transpose = np.concatenate((df_acc[["X (m/s^2)"]].to_numpy(), df_acc[["Y (m/s^2)"]].to_numpy(),
                                               df_acc[["Z (m/s^2)"]].to_numpy(), df_acc[['events']]), axis=1)
            print("\n\nnumpy dataframe for acceleration before transposed\n", acc_before_transpose)
            print("\n\nshape of accelerometer before transposed\n", np.shape(acc_before_transpose))

            acc = np.transpose(np.concatenate((df_acc[["X (m/s^2)"]].to_numpy(), df_acc[["Y (m/s^2)"]].to_numpy(),
                                               df_acc[["Z (m/s^2)"]].to_numpy(), df_acc[['events']]), axis=1))
            print("\n\nnumpy dataframe for acceleration after transposed\n", acc)
            print("\n\nshape of accelerometer after transposed\n", np.shape(acc))

            gyr_before_transpose = np.concatenate((df_gyr[["X (rad/s)"]].to_numpy(), df_gyr[["Y (rad/s)"]].to_numpy(),
                                               df_gyr[["Z (rad/s)"]].to_numpy(), df_gyr[['events']]), axis=1)
            print("\n\nnumpy dataframe for gyroscope before transposed\n", gyr_before_transpose)
            gyr = np.transpose(np.concatenate((df_gyr[["X (rad/s)"]].to_numpy(), df_gyr[["Y (rad/s)"]].to_numpy(),
                                               df_gyr[["Z (rad/s)"]].to_numpy(), df_gyr[['events']]), axis=1))
            print("\n\nnumpy dataframe for gyroscope after transposed\n", gyr)

        return acc, gyr

    def align_events(self, ref, target, ref_fs, tgt_fs, eventLen):
        # if reference signal duration is longer than the target signal duration
        t_ref = ref.shape[1] / ref_fs
        t_tgt = target.shape[1] / tgt_fs
        if t_ref > t_tgt:
            dif = t_ref - t_tgt
            add_in = int(round(dif*tgt_fs))
            add = np.zeros(shape=(4, add_in))
            target = np.concatenate((target, add), axis=1)

        rate_fs = tgt_fs / ref_fs
        ref_event = ref[3, :]
        # find event i in the ref signal
        # map index to tgt
        for i in range(eventLen):
            events = np.argwhere(ref_event == (i + 1))
            new_index = rate_fs * events
            new_index = np.round(new_index).astype(np.int32)
            target[3, new_index] = i + 1

        print("\n\nreference matrix (Enode sensor):\n", ref)
        print("\n\ntarget matrix (SmartPhone sensor):\n", target)

        return target

    def run(self):
        print(f"\n\nReading data...")
        # this function will read experiments data for all subjects
        # read data files - X_acc, X_gyro, X_mag, y

        X_acc_s, X_gyr_s = self.read2minData(dir=self.sp_data_dir, fname='t11' + '_2mW', rep=None, ext='.xls')
        X_acc_e, X_gyr_e = self.read2minData(dir=self.enode_data_dir, fname='t11' + '_2mW' + '_labelled',
                                             rep=None, ext='.tsv')

        # X_acc_s, X_gyr_s = self.read2minData(dir=self.sp_data_dir, fname='s6' + '_2mW_1', rep=None, ext='.xls')
        # X_acc_e, X_gyr_e = self.read2minData(dir=self.enode_data_dir, fname='s6' + '_2mW_1' + '_labelled',
        #                                      rep=None, ext='.tsv')

        # ## ISRAT
        # X_acc_s, X_gyr_s = self.read2minData(dir=self.sp_data_dir, fname='s4' + '_STS5r_1', rep=None, ext='.xls')
        # X_acc_e, X_gyr_e = self.read2minData(dir=self.enode_data_dir, fname='s4' + '_STS5r_1' + '_labelled',
        #                                      rep=None, ext='.tsv')
        # ## ISRAT
        # X_acc_s, X_gyr_s = self.read2minData(dir=self.sp_data_dir, fname='s4' + '_TUGst_1', rep=None, ext='.xls')
        # X_acc_e, X_gyr_e = self.read2minData(dir=self.enode_data_dir, fname='s4' + '_TUGst_1' + '_labelled_modified',
        #                                      rep=None, ext='.tsv')
        #
        # ## ISRAT
        # X_acc_s, X_gyr_s = self.read2minData(dir=self.sp_data_dir, fname='s4' + '_2mW_1', rep=None, ext='.xls')
        # X_acc_e, X_gyr_e = self.read2minData(dir=self.enode_data_dir, fname='s4' + '_2mW_1' + '_labelled',
        #                                      rep=None, ext='.tsv')

        delay = 0
        srate_enode = 62.5
        srate_s = 100.0
        eventLen = 7

        if delay >= 0:
            pad = np.zeros((4, int(delay * srate_s)))
            X_acc_s = np.concatenate([pad, X_acc_s], axis=1)
            X_gyr_s = np.concatenate([pad, X_gyr_s], axis=1)
        else:
            s_in = int(delay * srate_s)
            X_acc_s = X_acc_s[:, -s_in:]
            X_gyr_s = X_gyr_s[:, -s_in:]

        X_acc = self.align_events(X_acc_e, X_acc_s, srate_enode, srate_s, eventLen)
        X_gyr = self.align_events(X_acc_e, X_gyr_s, srate_enode, srate_s, eventLen)

        data = {"acc": X_acc, "gyr": X_gyr, "enode": X_acc_e}
        # returns acc_data, gyr_data, cognitive scores and class labels
        return data

