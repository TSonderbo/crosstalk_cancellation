import sounddevice as sd
import numpy as np
import maximumlengthsequence as mlseq
import time
import matplotlib.pyplot as plt

def measure_impulse_response(fs, plot_signal = False):

    mls, sig = mlseq.generate_mls_signal(fs, T = 10, nbits=15)

    # Left channel impulse responses
    left_sig = np.column_stack((sig, np.zeros(len(sig))))

    left_recordings = sd.playrec(left_sig, fs, channels=2)
    sd.wait()

    time.sleep(2)

    hLL, hLL_quality = mlseq.decode_mls_signal(left_recordings[:,0], mls)
    hRL, hRL_quality = mlseq.decode_mls_signal(left_recordings[:,1], mls)

    # Right channel impulse responses
    right_sig = np.column_stack((np.zeros(len(sig)), sig))

    right_recordings = sd.playrec(right_sig, fs, channels=2)
    sd.wait()

    hLR, hLR_quality = mlseq.decode_mls_signal(right_recordings[:,0], mls)
    hRR, hRR_quality = mlseq.decode_mls_signal(right_recordings[:,1], mls)
    
    out = np.column_stack((hLL, hLR, hRL, hRR))
    
    if(plot_signal == True):
        fig, (ax0, ax1, ax2 ,ax3) = plt.subplots(4,1,layout='constrained')
        ax0.plot(left_recordings[:,0])
        ax0.set_title("hLL")
        ax1.plot(left_recordings[:,1])
        ax1.set_title("hRL")
        ax2.plot(right_recordings[:,0])
        ax2.set_title("hLR")
        ax3.plot(right_recordings[:,1])
        ax3.set_title("hRR")
        plt.show()

    return out

def measure_impulse_response_with_ref(fs, plot_signal = False, pad_len = 0):

    asio_out = sd.AsioSettings(channel_selectors=[0, 1, 2])
    asio_in = sd.AsioSettings(channel_selectors=[0, 1, 2])
    sd.default.extra_settings = asio_in, asio_out

    mls, sig = mlseq.generate_mls_signal(fs, T = 10, nbits=15)

    # Left channel impulse responses
    left_sig = np.column_stack((sig, np.zeros(len(sig)), sig))

    left_recordings = sd.playrec(left_sig, fs, channels=3, device=54)
    sd.wait()

    time.sleep(2)

    hLL, _ = mlseq.decode_mls_signal(left_recordings[:,0], mls)
    hRL, _ = mlseq.decode_mls_signal(left_recordings[:,1], mls)
    h_ref, _ = mlseq.decode_mls_signal(left_recordings[:,2], mls)

    ref_idx = np.argmax(h_ref)

    hLL = hLL[ref_idx - pad_len:]
    hRL = hRL[ref_idx - pad_len:]

    # Right channel impulse responses
    right_sig = np.column_stack((np.zeros(len(sig)), sig, sig))

    right_recordings = sd.playrec(right_sig, fs, channels=3, device=54)
    sd.wait()

    hLR, _ = mlseq.decode_mls_signal(right_recordings[:,0], mls)
    hRR, _ = mlseq.decode_mls_signal(right_recordings[:,1], mls)
    h_ref, _ = mlseq.decode_mls_signal(right_recordings[:,2], mls)

    ref_idx = np.argmax(h_ref)

    hLR = hLR[ref_idx - pad_len:]
    hRR = hRR[ref_idx - pad_len:]
    
    out = np.column_stack((hLL, hLR, hRL, hRR))

    return out