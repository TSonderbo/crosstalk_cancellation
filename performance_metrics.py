import numpy as np
import scipy as sp
import scipy.fft as fft

def channel_separation(filt, Nc):

    R = np.empty((2,2,Nc), dtype='complex_')

    for i in range(0,2):
        for j in range(0,2):
            R[i,j,:] = fft.fft(filt.h[i,j,:], Nc) * fft.fft(filt.A[i,j,:], Nc)

    CHSP1 = R[0,1,:] / R[0,0,:]
    CHSP2 = R[1,0,:] / R[1,1,:]



    return out

def performance_error(filt, Nc):

    

    return out