import numpy as np
import matplotlib.pyplot as plt

def plot_impulse_response(sig, fs = 48000, plot_time=False):
    if(plot_time):
        time = np.arange(len(sig[:,0]))*1000/fs
    else:
        time = np.arange(len(sig[:,0]))
    mx = np.max(sig)
    mn = np.min(sig)

    fig, ((ax0, ax1), (ax2 ,ax3)) = plt.subplots(2,2,layout='constrained')
    ax0.plot(time, sig[:,0])
    ax0.set_title("hLL")
    ax0.set_ylim(mn, mx)
    ax0.set_xlabel('Samples')
    ax0.set_ylabel('Amplitude')

    ax1.plot(time, sig[:,1])
    ax1.set_title("hLR")
    ax1.set_ylim(mn, mx)
    ax1.set_xlabel('Samples')
    ax1.set_ylabel('Amplitude')

    ax2.plot(time, sig[:,2])
    ax2.set_title("hRL")
    ax2.set_ylim(mn, mx)
    ax2.set_xlabel('Samples')
    ax2.set_ylabel('Amplitude')

    ax3.plot(time, sig[:,3])
    ax3.set_title("hRR")
    ax3.set_ylim(mn, mx)
    ax3.set_xlabel('Samples')
    ax3.set_ylabel('Amplitude')
    plt.show()

def plot_playrec(sig_play, sig_rec, fs = 48000):
    

    mx = np.max((np.max(sig_play), np.max(sig_rec)))
    mn = np.min((np.min(sig_play), np.min(sig_rec)))

    maxLength = np.min((len(sig_play), len(sig_rec))) - 1

    time = np.arange(maxLength)*1/fs

    fig, (ax0, ax1) = plt.subplots(2,1,layout='constrained')
    ax0.plot(time, sig_play[0:maxLength,0], c='y', label='play')
    ax0.plot(time, sig_rec[0:maxLength,0], c='b', label='recording')
    ax0.set_title("Left Channel")
    ax0.legend()
    ax0.set_ylim(mn, mx)
    ax1.plot(time, sig_play[0:maxLength,1], c='y', label='play')
    ax1.plot(time, sig_rec[0:maxLength,1], c='b', label='recording')
    ax1.set_title("Right Channel")
    ax1.legend()
    ax1.set_ylim(mn, mx)
    plt.show()

def plot_stereo(sig, fs = 48000):
    time = np.arange(len(sig))*1/fs

    mx = np.max(sig)
    mn = np.min(sig)

    fig, (ax0, ax1) = plt.subplots(2,1,layout='constrained')
    ax0.plot(time, sig[:,0])
    ax0.set_title("Left Channel")
    ax0.set_ylim(mn, mx)
    ax1.plot(time, sig[:,1])
    ax1.set_title("Right Channel")
    ax1.set_ylim(mn, mx)