import numpy as np
import scipy as sp
import scipy.fft as fft
import scipy.signal as signal
import scipy.linalg as linalg
import math
from numpy.linalg import inv

class LS_cancellation_filter_FD:

    def __init__(self, h, h_full, Nc, beta):

        self.Nc = Nc

        self.h_full = np.array([[h_full[:,0], h_full[:,1]],
                       [h_full[:,2], h_full[:,3]]])
        
        self.h = np.array([[h[:,0], h[:,1]],
                       [h[:,2], h[:,3]]])

        H = np.array([[fft.fft(h[:,0], Nc), fft.fft(h[:,1], Nc)],
                       [fft.fft(h[:,2], Nc), fft.fft(h[:,3], Nc)]])
        
        self.H = H

        #Build delta function
        dlength = np.size(self.h,2)
        delta = np.zeros(dlength)
        delta[dlength - 1] = 1

        #Target filter array - i.e delta functions on the diagonals
        d = np.array([[fft.fft(delta, Nc), fft.fft(np.zeros(dlength), Nc)],
                       [fft.fft(np.zeros(dlength), Nc), fft.fft(delta, Nc)]])

        # Complex Conjugate Transpose
        H_T = np.empty((2,2,Nc), dtype='complex_')
        for k in range(0, Nc):
            H_T[:,:,k] = H[:,:,k].conj().T

        I = np.identity(2) # 2x2 Identity Matrix

        m = Nc//2 #Delay
        
        A = np.empty((2,2,Nc), dtype='complex_')
        for k in range(1, Nc):
            A[:,:,k] = np.linalg.inv(np.matrix(H_T[:,:,k], dtype='complex_') * np.matrix(H[:,:,k], dtype='complex_') * beta * I) \
            * np.matrix(H_T[:,:,k], dtype='complex_') * np.matrix(d[:,:,k], dtype='complex_')

        h0 = np.array([[fft.ifft(A[0,0,:], Nc), fft.ifft(A[0,1,:], Nc)],
                       [fft.ifft(A[1,0,:], Nc), fft.ifft(A[1,1,:], Nc)]])

        h0 = np.roll(h0, m, 2)
        
        self.A = h0

class LS_Cancellation_filter_TD:

    def __init__(self, h, h_full, Nc, beta, b):

        #Impulse response array shaped by
        #microphones on row and speakers on columns
        #depth is in samples
        self.h = np.array([[h[:,0], h[:,1]],
                       [h[:,2], h[:,3]]])

        Nh = np.size(h,0)

        #Build delta function
        delta = np.zeros(Nh)
        delta[Nh - 1] = 1

        #Target filter array - i.e delta functions on the diagonals
        d = np.array([[delta, np.zeros(Nh)],
                       [np.zeros(Nh), delta]])


        Nhc = Nh + Nc - 1

        delay = np.zeros(Nc)

        t = int(np.ceil(Nc/2))

        delay[t] = 1

        dm = np.zeros((2, 2, np.size(d,2) + np.size(delay, 0) - 1))

        delay = np.matrix(delay)

        for i in range(0, 2): #Row
            for j in range(0, 2): #Col
                cmtx = np.matrix(linalg.convolution_matrix(d[j,i,:], Nc))
                cmtx_d = np.array(cmtx*delay.T).flatten()
                dm[j,i,:] = cmtx_d
        
        H_t = np.zeros((2*Nhc, Nc*2))

        for i in range(0, 2): #Row
            for j in range(0, 2): #Col
                H_t[j*Nhc:(j+1)*Nhc, i*Nc:(i+1)*Nc] = linalg.convolution_matrix(self.h[j,i,:], Nc)

        A = np.empty((2,2,Nc))

        B_t = np.zeros((2*Nc, 2*Nc))

        for i in range(0, 2): #Row
            for j in range(0, 2): #Col
                if(i == j):
                    B_t[j*Nc:(j+1)*Nc, i*Nc:(i+1)*Nc] = linalg.convolution_matrix(b, Nc, 'same')
                else:
                    B_t[j*Nc:(j+1)*Nc, i*Nc:(i+1)*Nc] = np.zeros((Nc, Nc))

        H_t = np.matrix(H_t)
        B_t = np.matrix(B_t)


        for i in range(0,2):
            hh = np.linalg.inv(H_t.T * H_t + beta * B_t.T * B_t) * H_t.T * np.matrix((dm[:,i,:].flatten())).T
            A[:,i,:] = np.reshape(hh, (2,Nc))
        

        self.A = A
        