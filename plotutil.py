import numpy as np
import matplotlib.pyplot as plt

TSIZE = 16 #Title size

def plot_impulse_response(sig, title, fs = 48000, plot_time=False, size=(6.4, 4.8)):
    
    if(plot_time):
        time = np.arange(len(sig[:,0]))*1000/fs
    else:
        time = np.arange(len(sig[:,0]))

    mx = np.max(sig)
    mn = np.min(sig)
    
    padding = np.max((np.abs(mx), np.abs(mn))) * 0.1

    mx = mx + padding
    mn = mn - padding

    fig, ((ax0, ax1), (ax2 ,ax3)) = plt.subplots(2,2, figsize=size,layout='constrained')
    ax0.plot(time, sig[:,0])
    ax0.set_title("hLL", loc='left')
    ax0.set_ylim(mn, mx)

    ax1.plot(time, sig[:,1])
    ax1.set_title("hLR", loc='left')
    ax1.set_ylim(mn, mx)

    ax2.plot(time, sig[:,2])
    ax2.set_title("hRL", loc='left')
    ax2.set_ylim(mn, mx)

    ax3.plot(time, sig[:,3])
    ax3.set_title("hRR", loc='left')
    ax3.set_ylim(mn, mx)

    fig.supylabel("Amplitude")
    fig.supxlabel("Samples")
    fig.suptitle(title, fontsize=TSIZE)

    plt.show()

def plot_psd(sigs, title, fs=48000, size=(6.4, 4.8)):

    fig = plt.figure(figsize=size)

    for sig in sigs:
        plt.psd(sig[0], label=sig[1], Fs=fs)

    plt.legend()
    plt.title(title, fontsize=TSIZE)
    plt.xscale("log")
    plt.show()

def plot_playrec(sig_play, sig_rec, fs = 48000, size=(6.4, 4.8)):

    mx = np.max((np.max(sig_play), np.max(sig_rec)))
    mn = np.min((np.min(sig_play), np.min(sig_rec))) 
    
    padding = np.max((np.abs(mx), np.abs(mn))) * 0.1

    mx = mx + padding
    mn = mn - padding

    maxLength = np.min((len(sig_play), len(sig_rec))) - 1

    time = np.arange(maxLength)*1/fs

    fig, (ax0, ax1) = plt.subplots(2,1,layout='constrained', figsize=size)
    ax0.plot(time, sig_play[0:maxLength,0], c='y', label='play')
    ax0.plot(time, sig_rec[0:maxLength,0], c='b', label='recording')
    ax0.set_title("Left Channel")
    ax0.legend()
    ax0.set_ylim(mn, mx)
    ax1.plot(time, sig_play[0:maxLength,1], c='y', label='play')
    ax1.plot(time, sig_rec[0:maxLength,1], c='b', label='recording')
    ax1.set_title("Right Channel", fontsize=TSIZE)
    ax1.legend()
    ax1.set_ylim(mn, mx)
    plt.show()

def plot_stereo(sig, title, fs = 48000, size=(6.4, 4.8)):
    
    time = np.arange(len(sig))*1/fs

    mx = np.max(sig)
    mn = np.min(sig)

    padding = np.max((np.abs(mx), np.abs(mn))) * 0.1

    mx = mx + padding
    mn = mn - padding

    fig, (ax0, ax1) = plt.subplots(2,1,layout='constrained', figsize=size)
    ax0.plot(time, sig[:,0])
    ax0.set_title("Left Channel", loc="left")
    ax0.set_ylim(mn, mx)
    ax1.plot(time, sig[:,1])
    ax1.set_title("Right Channel", loc="left")
    ax1.set_ylim(mn, mx)

    fig.suptitle(title, fontsize=TSIZE)
    fig.supylabel("Amplitude")
    fig.supxlabel("Time (s)")

    plt.show()
    
def plot_comparison(sig_rec, sig_sim, title, fs=48000, normalize=False, size=(6.4, 4.8)):
    
    time = np.arange(len(sig_rec))*1/fs 

    if(normalize):
        sig_rec /= np.max(sig_rec)
        sig_sim /= np.max(sig_sim)

    mx = np.max((sig_rec, sig_sim))
    mn = np.min((sig_rec, sig_sim))

    padding = np.max((np.abs(mx), np.abs(mn))) * 0.1

    mx = mx + padding
    mn = mn - padding

    fig = plt.figure(figsize=size)
    plt.plot(time, sig_rec[:,0], label="Recorded")
    plt.plot(time, sig_sim[:,0], label="Simulated")
    plt.ylim(mn, mx)
    
    fig.suptitle(title, fontsize=TSIZE)
    fig.supylabel("Amplitude")
    fig.supxlabel("Time (s)")
    fig.legend()

    plt.show()

def plot_stereo_comparison(sig_rec, sig_sim, title, fs=48000, normalize=False, size=(6.4, 4.8)):
    
    time = np.arange(len(sig_rec))*1/fs 

    if(normalize):
        sig_rec /= np.max(sig_rec)
        sig_sim /= np.max(sig_sim)

    mx = np.max((sig_rec, sig_sim))
    mn = np.min((sig_rec, sig_sim))

    padding = np.max((np.abs(mx), np.abs(mn))) * 0.1

    mx = mx + padding
    mn = mn - padding

    fig, (ax0, ax1) = plt.subplots(2,1,layout='constrained', figsize=size)
    ax0.plot(time, sig_rec[:,0], label="Recorded")
    ax0.plot(time, sig_sim[:,0], label="Simulated")
    ax0.set_title("Left Channel", loc="left")
    ax0.set_ylim(mn, mx)

    ax1.plot(time, sig_rec[:,1], label="Recorded")
    ax1.plot(time, sig_sim[:,1], label="Simulated")
    ax1.set_title("Right Channel", loc="left")
    ax1.set_ylim(mn, mx)

    fig.suptitle(title, fontsize=TSIZE)
    fig.supylabel("Amplitude")
    fig.supxlabel("Time (s)")
    fig.legend()

    plt.show()