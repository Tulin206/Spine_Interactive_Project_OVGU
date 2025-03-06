import numpy as np
from matplotlib import pyplot as plt


def plot_signals(signal_list, event_list, eventLen, signal_label=None, event=False, type=None, title=""):
    """This function plots multiple plots"""
    n_sig = len(signal_list)
    fig, ax = plt.subplots(n_sig, 1, figsize=(12, 6))
    fig.suptitle(title, fontsize=16)
    color_list = ['bo', 'go', 'ro', 'ko', 'co', 'mo', 'yo']

    for i in range(n_sig):
        signal = signal_list[i]

        if type == "sub":
            ax[i].plot(signal)
            if event is True:
                events = event_list[i]
                for ev in range(eventLen):
                    ev_inx = np.argwhere(events == (ev + 1))
                    ax[i].plot(ev_inx, signal[ev_inx], color_list[ev])

            ax[i].set_ylabel(signal_label[i])
            ax[i].axhline(y=0, color='r', linestyle='--')
            ax[i].grid(color='grey')

        else:
            plt.plot(signal)
            if event is True:
                events = event_list[i]
                for ev in range(eventLen):
                    ev_inx = np.argwhere(events == (ev + 1))
                    plt.plot(ev_inx, signal[ev_inx], 'bo')
            plt.axhline(y=0, color='r', linestyle='--')
            plt.grid(color='grey', alpha=0.2)

    #plt.figure()
    #plt.savefig("plot.png")
    plt.show()


def plot_signal_peaks(signal, peaks, title=""):
    """This function plots a signal and shows the peak values"""
    plt.plot(signal)
    plt.plot(peaks, signal[peaks], "x")
    plt.figure(figsize=(12, 6))
    plt.title = title
    plt.savefig("plot_peak.png")
    plt.show()
