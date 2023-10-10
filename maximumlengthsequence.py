import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

# def generate_mls(pmls = 8, nmls = 0):

#     ppt_table=np.array([1], [2, 1], [3, 1], [4, 1], [5, 2], [6, 1], [7, 1], [8, 4, 3, 2], [9, 4], [10, 3], [11, 2], [12, 6, 4, 1], [13, 4, 3, 1] ,[14, 5, 3, 1] ,[15, 1] ,[16, 5, 3, 2] ,[17, 3], [18, 5, 2, 1] [19, 5, 2, 1])

#     if(nmls == 0):
#         nmls = (2^pmls)-1

#     ppt = ppt_table[pmls]

#     s = np.ones(pmls,1)
#     mls = np.zeros(nmls,1)
#     for j in range(nmls):
#         s0 = s[0]
#         for k in range(1,len(ppt)):
#             s0 = xor(bool(s0),bool(s[ppt[k]]))
#         s[1:pmls]=s[0:pmls-1]
#         s[0] = s0
#         mls[j]=s0
    
#     mls = [2*x-1 for x in mls]

#     return mls

def generate_mls_signal(fs, T = 5, nbits = 13):
    """ This generates an mls signal for measuring impulse responses in sound devices

        Parameters
        ----------
        T : int
            Duration of the mls signal in seconds.
        Fs : int
            Samplerate in Hz.
        nbits : int, optional
            Number of bits to use for the mls - anything greater than 16 can take a while.
        """
    mls = signal.max_len_seq(nbits)[0]

    mls = np.array([(x * 2 - 1)*0.5 for x in mls])

    n = int(T*fs/len(mls))

    y = np.tile(mls, n)

    return mls, y


def decode_mls_signal(sig, mls):
    """ Decode a impulse response recording with the mls used for recording

        Parameters
        ----------
        sig : array
            The recorded signal.
        mls : array
            The mls signal used for recording.
        """

    sig = np.array(sig)
    mls = np.array(mls)

    filtsig = signal.oaconvolve(np.flip(mls), sig) #overlap add method

    Q = 2
    T = 0

    i = len(mls)
    start = Q*len(mls)
    stop = len(filtsig)-Q*len(mls)
    l = len(np.arange(start, stop, i))

    seg = np.zeros((l,i))

    for n in range(start, stop, i):
        seg[T,:] = filtsig[range(n, n+i)]
        T += 1

    decodedData = np.mean(seg, axis=0)
    std = [np.std(x) for x in seg.T]
    quality=1-np.mean([x**2 for x in std])/np.mean([x**2 for x in decodedData])

    return decodedData, quality
