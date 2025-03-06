"""
This python file takes acc and gyro signals returned from DataExtraction.py and
detects gait events from acc signal (numpy array). Afterwards, it segments signals into gait cycles
Gait events: Heel strike, Toe off, Heel off ...
Gait cycle from right leg HS to right leg HS.

run() function returns segmented acc and gyro signals (nparray): n_sub x n_cycles x acc, n_sub x n_cycles x gyro
and additionally gait events of each cycle. (It may be needed to extract features.)
"""
import matplotlib
matplotlib.use('TkAgg')  # Use WebAgg backend
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths, butter, sosfiltfilt
import numpy as np
from DataProcessingPipeline.DataExtraction import DataExtraction
from visualize.visualization import plot_signals, plot_signal_peaks

f_name = "t14_TUGst_1"


class GaitProcessing:
    def __init__(self, data, fs=None):
        """
        :param X_acc: (n_subjects x acc) Accelerometer signal of all participants
        :param X_gyro: (n_subjects x gyro)
        :param fs: sampling frequency
        """
        self.fs = 100
        self.Xacc = data["acc"]
        self.Xgyr = data["gyr"]
        self.enode = data["enode"]

    def compute_mag(self, sig):
        return np.sqrt(np.power(sig[0], 2) + np.power(sig[1], 2) + np.power(sig[2], 2))

    def apply_filter(self, sig):
        sos = butter(5, 5, btype='lowpass', fs=100, output='sos')
        filt1 = sosfiltfilt(sos, sig[0])
        filt2 = sosfiltfilt(sos, sig[1])
        filt3 = sosfiltfilt(sos, sig[2])
        return np.concatenate((np.expand_dims(filt1, axis=0), np.expand_dims(filt2, axis=0),
                               np.expand_dims(filt3, axis=0)), axis=0)

    def normalize(self, sig):
        return (sig - np.mean(sig)) / (np.max(sig) - np.min(sig))

    def apply_fft(self, sig):
        fs = 100
        n = len(sig)
        T = n / fs
        freq = np.arange(n) / T
        freq = freq[:len(freq) // 2]

        res = np.fft.fft(sig) / n
        res = res[:n // 2]

        """plt.plot(freq, abs(res))
        plt.show()"""

        ff_filtered = np.zeros_like(res)
        indices = np.where((np.abs(freq - 5) <= 1.0))[0]
        ff_filtered[indices] = res[indices]

        retrieved_sig = np.fft.ifft(ff_filtered).real

    def peak_detection(self, sig):
        peaks, prop = find_peaks(sig)
        # Identify the highest peak
        highest_peak_idx = peaks[np.argmax(sig[peaks])]
        high_peak_th = 0.6 * sig[highest_peak_idx]

        # Find all periodic peaks that are similar to the highest peak
        high_peaks, _ = find_peaks(sig, height=high_peak_th, distance=self.fs // 2)
        second_high_peaks, _ = find_peaks(sig, height=(0.2, high_peak_th))

        plt.figure(figsize=(12, 6))
        plt.plot(sig[0:1800], label='Signal')
        plt.plot(high_peaks[0:13], sig[high_peaks][0:13], 'rx', label='High Peaks')
        plt.xlabel('Time [s]')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.title('Signal with Periodic Peaks of the Highest Peak')
        plt.show()

        return peaks, prop

    def preprocess_signal(self, sig, mag=True):
        print("\n\nshape of sig before slicing event:", np.shape(sig))
        ev = sig[3, :]   # all columns of the 3rd row only
        print("\nshape of ev:", np.shape(ev))
        sig = sig[0:3, :]   # all columns of first 3 rows
        print("\nshape of sig:", np.shape(sig))
        sig = self.apply_filter(sig)
        print("\nshape of sig:", np.shape(sig))

        if mag is True:
            sig = self.compute_mag(sig)   # Compute the magnitude of the signal (x, y, z axes combined)
            sig = self.normalize(sig)     # Normalize the signal
            # remove DC component
            # sig = sig - np.mean(sig)
            sig = np.concatenate((np.expand_dims(sig, axis=0), np.expand_dims(ev, axis=0)), axis=0)
        else:
            # Normalize each row individually, and subtract the mean (DC component)
            for ax in range(3):
                sig[ax, :] = self.normalize(sig[ax, :])          # Normalize each axis
                sig[ax, :] = sig[ax, :] - np.mean(sig[ax, :])    # Subtract the mean (remove DC component)
            ev = np.reshape(ev, (1, len(ev)))          # Reshape `ev` into a 1-row vector
            print("\nshape of ev:", np.shape(ev))
            sig = np.concatenate((sig, ev), axis=0)      # Add `ev` to `sig` (concatenate along axis 0)
            print("\nshape of sig after normalizing and removing mean/DC component:", np.shape(sig))
            print(sig)
        return sig

    def plot(self, mag_enode, mag_acc, gyr, acc):
        eventLen = 7
        fs_e = 62.5
        fs_s = 100
        f_time = 14
        e_time = 21
        start_s, stop_s = f_time * fs_s, e_time * fs_s
        start_e, stop_e = int(f_time * fs_e), int(e_time * fs_e)

        plot_signals([mag_enode[0, start_e:stop_e], mag_acc[0, start_s:stop_s]],
                     [mag_enode[1, start_e:stop_e], mag_acc[1, start_s:stop_s]],
                     eventLen, ['Enode', 'Smartphone'], event=True, type="sub",
                     title="Magnitude of Acc signal - " + str(f_name) + "- (" + str(f_time) + "-" + str(e_time) + "s)")

        plot_signals([gyr[0, start_s:stop_s], gyr[1, start_s:stop_s], gyr[2, start_s:stop_s]],
                     [gyr[3, start_s:stop_s], gyr[3, start_s:stop_s], gyr[3, start_s:stop_s]],
                     eventLen, ['gyro_x', 'gyro_y', 'gyro_z'], event=True, type="sub",
                     title="Smartphone - Gyroscope signal - " + str(f_name) + " - (" + str(f_time) + "-" + str(
                         e_time) + "s)")

        plot_signals([acc[0, start_s:stop_s], acc[1, start_s:stop_s], acc[2, start_s:stop_s]],
                     [acc[3, start_s:stop_s], acc[3, start_s:stop_s], acc[3, start_s:stop_s]],
                     eventLen, ['acc_x', 'acc_y', 'acc_z'], event=True, type="sub",
                     title="Smartphone - Acc signal - " + str(f_name) + " - (" + str(f_time) + "-" + str(e_time) + "s)")

    def _get_highest_peak(self, sig, th=None):
        peaks, _ = find_peaks(sig)
        highest_peak_idx = peaks[np.argmax(sig[peaks])]
        if th is None:
            return highest_peak_idx
        else:
            high_peak_th = th * sig[highest_peak_idx]
            high_peaks, _ = find_peaks(sig, height=high_peak_th, distance=self.fs // 2)
            return high_peaks

    def _get_diff(self, sig, start=None, end=None):
        diff = []
        if start is None:
            start = 0
        if end is None:
            end = len(sig)

        for i in range(start, end, 1):
            if i + 1 <= end:
                diff.append(round(sig[i] - sig[i + 1], 2))
        return np.array(diff)

    def _get_zero_crossing(self, sig):
        zero_cross = []
        for i in range(len(sig)):
            if i == len(sig) - 1:
                break
            if np.sign(sig[i]) != np.sign(sig[i + 1]):
                zero_cross.append(i)

        return np.array(zero_cross)

    def _check_STS_events(self, events):
        passed = 0
        print("\n\nChecking if STS events are valid or not..")
        # check order of the STS events
        check_events = events[np.where(events != 0)]
        f_ev = check_events[::2]
        s_ev = check_events[1::2]
        if len(np.where(f_ev != 1)[0]) == 0 and len(np.where(s_ev != 2)[0]) == 0:
            passed = 1
        return passed

    def _check_TUG_events(self, events):
        # checks order of the events, if one is missing, then repeat the exercise
        # event should be ascending order
        passed = 0
        print("\n\nChecking if TUG events are valid or not..")
        labels = [1, 2, 3, 4, 5, 6, 7]
        labels_in = np.nonzero(events)
        labels_in = np.array(labels_in)
        if all(x in events for x in labels) and all(labels_in[:-1] <= labels_in[1:]):
            passed = 1
        return passed

    def _check_gait_events(self, events, threshold=80):
        # check each gait cycle. If one event is missing, exclude only this gait cycle
        # dont check second event because we dont detect it!
        # each gait cycle should be ascending order
        valid = 0
        labels = [1, 3, 4, 5, 6, 7]
        ev_1 = np.where(events == 1)[0]
        ev_7 = np.where(events == 7)[0]
        num_gc = min(len(ev_1), len(ev_7))
        for i in range(num_gc):
            gc = events[ev_1[i]:ev_7[i]+1]
            labels_in = np.array(np.nonzero(gc))[0]
            # if all events are not detected, remove whole gait cycle and
            if len(labels) != len(labels_in):
                # turn events 0
                events[ev_1[i]:ev_7[i]+1] = 0
            # if all events are not ordered
            if not all(labels_in[:-1] <= labels_in[1:]):
                # turn events 0
                events[ev_1[i]:ev_7[i]+1] = 0

        final_ev_1 = np.where(events == 1)[0]
        final_ev_7 = np.where(events == 7)[0]
        final_num_gc = min(len(final_ev_1), len(final_ev_7))
        rate = (final_num_gc/num_gc)*100
        if rate >= threshold:
            valid = 1
        return valid

    def get_gaitEvents(self, acc, gyr):
        # we can not detect event 2 through signal processing - loading process !!

        accX = acc[0, :]
        accY = acc[1, :]
        gyrZ = gyr[2, :]

        events = np.zeros((acc.shape[1],))
        # Mid Swing:7(olive) - from gyroZ - peak and the amplitude greater that the half of the max value
        h = np.max(gyrZ) / 2
        ms_peaks, _ = find_peaks(gyrZ, height=h)
        events[ms_peaks] = 7

        # Heel Strike:1(blue) - from accY - zero cross (neg2pos) after olive
        # hs_peaks, _ = find_peaks(accX)
        # events[hs_peaks[0]] = 1
        for ms in ms_peaks:
            forw = 1
            while (ms + forw) < len(accY):
                if accY[ms + forw] >= 0:
                    events[ms + forw] = 1
                    break
                forw += 1
        # remove first mid swing event from event array and ms_peaks array
        # if you can detect first heel strike, remove these lines
        events[ms_peaks[0]] = 0
        ms_peaks = ms_peaks[1:]

        # Pre Swing:5(cyan) - from accY - peak - highest peak between 1st and 7th event
        f_event = np.where(events == 1)[0]
        print("f_event:", f_event)
        gc = np.min([len(ms_peaks), len(f_event)])
        for c in range(gc):
            # to prevent overwrite +1/-1
            start = f_event[c] + 1
            stop = ms_peaks[c] - 1
            sig = accY[start:stop]
            ps_peak = self._get_highest_peak(sig)
            events[start + ps_peak] = 5

        # Toe-off:6(magenta) - from accX - first loc min after pre swing (5)
        # and gyro value at this point should be positive
        sig = np.power(accX, 2)
        ps_peaks = np.where(events == 5)[0]
        for ind, ps in enumerate(ps_peaks):
            # to prevent overwrite -1
            ps = ps + 1
            ms = ms_peaks[ind] - 1
            diff = self._get_diff(sig, ps, ms)
            for d, di in enumerate(diff):
                if di == 0.0 and gyrZ[ps + d] > 0:
                    events[ps + d] = 6
                    break

        # Terminal Stance:4(black) - from accY - zero cross before cyan
        for ps in ps_peaks:
            # to prevent overwrite -1
            ps = ps - 1
            back = 0
            while True:
                if accY[ps - back] <= 0:
                    events[ps - back] = 4
                    break
                back += 1

        # Mid Stance:3(red) - from gyrZ - peak under 0 or peak point between HS(1) and TS(4)
        """mst_peaks, _ = find_peaks(gyrZ, height=(None, 0))
        mst_peaks = mst_peaks[np.argmax(sig[mst_peaks])]
        events[mst_peaks] = 3"""
        ts_peaks = np.where(events == 4)[0]
        for c in range(gc):
            # to prevent overwrite +1/-1
            start = f_event[c] + 1
            stop = ts_peaks[c] - 1
            sig = gyrZ[start:stop]
            peaks, _ = find_peaks(sig)
            # signal and peaks not empty
            if len(sig) > 0 and len(peaks) > 0:
                mst_peak = peaks[np.argmax(sig[peaks])]
            else:
                # if there is no peaks then take middle point
                mst_peak = int((stop - start) / 2)
            events[start + mst_peak] = 3

        valid = self._check_gait_events(events)
        if valid == 0:
            print("\n\nPlease do the gait experiment again!")
        else:
            print("\n\nGait experiment is valid!")
        return events, valid

    def get_TUGEvents(self, acc, gyr):
        sos = butter(10, 1, btype='lowpass', fs=100, output='sos')
        gyroX = gyr[0, :]
        gyroY = gyr[1, :]
        gyroZ = gyr[2, :]
        accY = acc[1, :]
        # depending on the return direction, signal can be positive or negative
        end = len(gyroX)
        events = np.zeros((acc.shape[1],))

        # PEAKS
        """plt.plot(np.power(gyroY+gyroZ, 2))
        plt.plot(pre_gyroY)"""
        pre_gyroY = np.power(gyroY, 2)
        pre_gyroY_peaks = self._get_highest_peak(pre_gyroY, th=0.4)

        # Chair Rising Start: 1 - gyrox
        min_ind = np.argmin(gyroX)
        width_info = peak_widths(abs(gyroX), [min_ind], rel_height=0.9)
        rising_ind = int(width_info[2])
        events[rising_ind] = 1

        # beginning of trunk flexion = 6 - gyroy
        trunk_ind = pre_gyroY_peaks[-1]
        events[trunk_ind] = 6

        # Final rotation: 5 - gyroy
        zero_ind = self._get_zero_crossing(gyroY)
        fin_rot_ind = zero_ind[zero_ind < trunk_ind][-1]
        events[fin_rot_ind] = 5

        # Intermediate rotation: 3-4 - pre_gyroY
        pre_gyroY = sosfiltfilt(sos, np.power(gyroY[0:fin_rot_ind], 2))
        int_rot_peak = self._get_highest_peak(pre_gyroY)
        rot_width_info = peak_widths(pre_gyroY, [int_rot_peak], rel_height=0.9)
        left_rot = int(rot_width_info[2])
        right_rot = int(rot_width_info[3])
        events[left_rot] = 3
        events[right_rot] = 4
        """int_rot_ind = pre_gyroY_peaks[0]
        width_info = peak_widths(pre_gyroY, [int_rot_ind], rel_height=0.9)
        int_rot_ind = int(width_info[2])
        events[int_rot_ind] = 3"""

        # Chair Rising End: 2 - gyrox
        sig = gyroX[rising_ind:left_rot]
        rising_peak = self._get_highest_peak(sig)
        events[rising_ind + rising_peak] = 2

        # complete sit = 7 - gyrox
        x_peaks = self._get_highest_peak(gyroX[trunk_ind:end])
        x_peak_in = x_peaks + trunk_ind
        sit_ind = self._get_zero_crossing(gyroX[x_peak_in:end])[0]
        sit_ind = x_peak_in + sit_ind
        events[sit_ind] = 7

        valid = self._check_TUG_events(events)
        if valid == 0:
            print("\n\nPlease do the TUG experiment again!")
        else:
            print("\n\nTUG experiment is valid!")
        return events, valid

    def get_STSEvents(self, gyr):
        gyroX = gyr[0, :]
        events = np.zeros((gyr.shape[1],))
        # the rest is computed by referencing first peak
        peaks = self._get_highest_peak(gyroX, th=0.6)
        f_peak = peaks[0]
        zc_points = self._get_zero_crossing(gyroX[f_peak:])
        sit2stand_in = zc_points[2::4]
        stand2sit_in = zc_points[::4]
        events[sit2stand_in + f_peak] = 1
        events[stand2sit_in + f_peak] = 2
        # first event
        error = 0.05  # error value added to signal to get rid of negative starting points
        zc_points = self._get_zero_crossing(gyroX[0:f_peak] + error)
        events[zc_points[0]] = 1
        valid = self._check_STS_events(events)
        if valid == 0:
            print("\n\nPlease do the STS experiment again!")
        else:
            print("\n\nSTS experiment is valid!")
        return events, valid

    def mean_time_error(self, target, pred, event, fs=100):
        if event not in pred:
            print("\n\nEvent is not detected!")
        else:
            tar_times = np.divide(np.where(target == event), fs)
            pred_times = np.divide(np.where(pred == event), fs)
            l = len(tar_times[0])
            mae = np.mean(abs(np.subtract(tar_times[0, 0:l], pred_times[0, 0:l])))
            # print(abs(np.subtract(tar_times[0, 0:l], pred_times[0, 0:l])))
            print("\n\nMean Absolute Error of event", event, "is:", mae)

    def run(self, test):
        print(f"\n\nProcessing signal...")
        valid = 0
        events = 0
        unproc_acc = self.apply_filter(self.Xacc)
        # preproc signal
        mag_acc = self.preprocess_signal(self.Xacc)
        print("\n\nshape of mag_acc after pre processing:", np.shape(mag_acc))
        acc = self.preprocess_signal(self.Xacc, mag=False)
        print("\n\nshape of acc after pre processing:", np.shape(acc))
        gyr = self.preprocess_signal(self.Xgyr, mag=False)
        print("\n\nshape of gyr after pre processing:", np.shape(gyr))

        # enode magnitude is computed to remove the DC affect
        ev = self.enode[3, :]
        sig = self.enode[0:3, :]
        sig = self.compute_mag(sig)
        sig = self.normalize(sig)
        mag_enode = np.concatenate((np.expand_dims(sig, axis=0), np.expand_dims(ev, axis=0)), axis=0)
        print("\n\nshape of mag_enode:", np.shape(mag_enode))

        self.plot(mag_enode, mag_acc, gyr, acc)
        if test == "gait":
            events, valid = self.get_gaitEvents(acc, gyr)
            print("\nevents", events[180:290])
            print("\nvalid", valid)
        elif test == "sts":
            events, valid = self.get_STSEvents(gyr)
        elif test == "tug":
            events, valid = self.get_TUGEvents(acc, gyr)

        """"
        plot_signals([gyr[0, 0:2000], gyr[0, 0:2000]],
                     [mag_acc[1, 0:2000], events[0:2000]], 7,
                     ['mag_acc - actual events', 'mag_acc and predicted events'], event=True, type="sub",
                     title=str(f_name) + " - (" + str(5) + "-" + str(10) + "s)")
         
        # Evalute Event Detection
        for i in range(7):
            self.mean_time_error(mag_acc[1, :], events, i + 1)

        plot_signals([mag_acc[0, 0:1500], mag_acc[0, 0:1500]],
                     [mag_acc[1, 0:1500], events[0:1500]], 7,
                     ['mag_acc', 'mag_acc'], event=True, type="sub",
                     title="mid swing - s2_2mW_1 - (" + str(0) + "-" + str(15) + "s)")"""

        return acc[0:3, :], gyr[0:3, :], events, unproc_acc, valid
