import sounddevice as sd
import numpy as np
import maximumlengthsequence as mlseq
import time
import plotutil


def measure_impulse_response(fs):

    mls, sig = mlseq.generate_mls_signal(fs, T = 10, nbits=15)

    # Left channel impulse responses
    left_sig = np.column_stack((sig, np.zeros(len(sig))))

    left_recordings = sd.playrec(left_sig, fs, channels=2, dtype='float64')
    sd.wait()

    time.sleep(2)

    hLL, hLL_quality = mlseq.decode_mls_signal(left_recordings[:,0], mls)
    hRL, hRL_quality = mlseq.decode_mls_signal(left_recordings[:,1], mls)

    # Right channel impulse responses
    right_sig = np.column_stack((np.zeros(len(sig)), sig))

    right_recordings = sd.playrec(right_sig, fs, channels=2, dtype='float64')
    sd.wait()

    hLR, hLR_quality = mlseq.decode_mls_signal(right_recordings[:,0], mls)
    hRR, hRR_quality = mlseq.decode_mls_signal(right_recordings[:,1], mls)
    
    out = np.column_stack((hLL, hLR, hRL, hRR))
    
    
    # fig, (ax0, ax1, ax2 ,ax3) = plt.subplots(4,1,layout='constrained')
    # ax0.plot(left_recordings[:,0])
    # ax0.set_title("hLL")
    # ax1.plot(left_recordings[:,1])
    # ax1.set_title("hRL")
    # ax2.plot(right_recordings[:,0])
    # ax2.set_title("hLR")
    # ax3.plot(right_recordings[:,1])
    # ax3.set_title("hRR")
    # plt.show()

    #plotutil.plot_impulse_response(np.column_stack((left_recordings, right_recordings)))

    return out
