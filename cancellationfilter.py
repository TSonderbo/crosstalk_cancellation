import numpy as np
import scipy.signal as signal
import scipy.linalg as linalg

class Cancellation_filter:

    def __init__(self, h, filterlength):
        """Instantiate a cross talk cancellatio filter.

        Parameters
        ----------
        h : array_like
            An Nx4 array of HRIR with the columns corresponding to hLL, hLR, hRL, hRR. 
        filterlength : _type_
            The desired length of the cancellation filter. Note that increasing this to large numbers significantly increases the computation time
        """
        #if(len(h) <= filterlength):
        #    raise Exception("")
        self.filterLength = filterlength

        self.A = self.__build_filter(h, filterlength)

    def __build_filter(self, h, filterlength):


        #Impulse response array shaped by
        #microphones on row and speakers on columns
        #depth is in samples
        self.h = np.array([[h[:,0], h[:,1]],
                       [h[:,2], h[:,3]]])

        #Build delta function
        dlength = np.size(self.h,2)
        delta = np.zeros(dlength)
        delta[dlength - 1] = 1

        #Target filter array - i.e delta functions on the diagonals
        d = np.array([[delta, np.zeros(dlength)],
                       [np.zeros(dlength), delta]])

        self.nrMics = np.size(self.h, 0) #Nr of microphones
        self.nrSpk = np.size(self.h, 1) #Nr of speakers
        self.nrSamp = np.size(self.h, 2) #Length of impulse responses / number of samples

        lsm = self.nrSamp + filterlength - 1

        delay = np.zeros(filterlength)

        t = int(np.ceil(filterlength/2))

        delay[t] = 1

        cz = np.zeros((self.nrMics, self.nrSpk, np.size(d,2) + np.size(delay, 0) - 1))
        
        delay = np.matrix(delay)

        for i in range(0, self.nrMics): #Row
            for j in range(0, self.nrSpk): #Col
                cmtx = np.matrix(linalg.convolution_matrix(d[i,j,:], filterlength))
                cmtx_d = np.array(cmtx*delay.T).flatten()
                cz[i,j,:] = cmtx_d
        
        cm = np.zeros((self.nrSpk*lsm, filterlength*self.nrMics))

        for i in range(0, self.nrMics): #Row
            for j in range(0, self.nrSpk): #Col
                cm[j*lsm:(j+1)*lsm, i*filterlength:(i+1)*filterlength] = linalg.convolution_matrix(self.h[i,j,:], filterlength)

        H = np.empty((self.nrMics,self.nrSpk,filterlength))

        for i in range(0, self.nrSpk):
            hh = np.array(np.matrix(linalg.pinv(cm, None)) * np.matrix((cz[:,i,:].flatten())).T)
            hh = np.reshape(hh, (self.nrMics,filterlength))
            H[:,i,:] = hh
            
        return H

    def filter_stereo(self, sig):
        """Filters the input with the cancellation filters of the filter object.

        Parameters
        ----------
        sig : array_like
            An Nx2 array of N audio samples where the columns correspond to the left and right channel. 

        Returns
        -------
        out : array_like
            An Nx2 array of filtered audio samples where the columns correspond to the left and right channel.
        """

        outLength = np.size(sig,0) + self.filterLength - 1

        out = np.zeros((outLength, 2))

        for i in range(0, self.nrMics):
            for j in range(0, self.nrSpk):
                out[:,i] += signal.fftconvolve(sig[:,j], self.A[i,j,:])

        # b = signal.firwin(16, 0.66)
        # out[:,0] = signal.filtfilt(b, 1, out[:,0], padlen=150)    
        # out[:,1] = signal.filtfilt(b, 1, out[:,1], padlen=150)    
        out /= np.max(np.abs(out))
        return out

    # def filter_left(self, sig):

    #     if(len(np.shape(sig)) > 1):
    #         sig = sig[:,0]

    #     outLength = np.size(sig,0) + self.filterLength - 1

    #     out = np.zeros((outLength, 2))

    #     for i in range(0, self.nrMics):
    #         out[:,i] = signal.fftconvolve(sig[:,0], self.A[i,0,:])

    #     return out

    # def filter_right(self, sig):
        
    #     if(len(np.shape(sig)) > 1):
    #         sig = sig[:,1]

    #     outLength = np.size(sig,0) + self.filterLength - 1

    #     out = np.zeros((outLength, 2))

    #     for i in range(0, self.nrMics):
    #         out[:,i] = signal.fftconvolve(sig[:,1], self.A[i,1,:])

    #     max = np.amax(out)

    #     out = (out/max)*0.5

    #     return out
        
    def filter_reference(self, sig):
        """Filters the input signal with the cancellations filters of the object as well as the corresponding HRIRs. 
        The resulting output indicates the theoretical preassures at the ears of the listener.

        Parameters
        ----------
        sig : array_like
            An Nx2 array of N audio samples where the columns correspond to the left and right channel. 

        Returns
        -------
        out : array_like
            An Nx2 array of N filtered audio samples representing the theoretical preassured at the ears of the listener. 
            The columns correspond to left and right channel. 
        """

        outLength_a = np.size(sig,0) + self.filterLength - 1

        out_a = np.zeros((outLength_a, 2))

        for i in range(0, self.nrMics):
            for j in range(0, self.nrSpk):
                out_a[:,i] += signal.fftconvolve(sig[:,j], self.A[i,j,:])

        outLength_h = np.size(out_a, 0) + self.nrSamp - 1

        out_h = np.zeros((outLength_h, 2))
        for i in range(0, self.nrMics):
            for j in range(0, self.nrSpk):
                out_h[:,i] += signal.fftconvolve(out_a[:,j], self.h[i,j,:])

        return out_h

def CHSP(sig, sig_rec):
    """Calculate the Channel Separation index.

    Parameters
    ----------
    sig : array_like
        The dry signal.
    sig_rec : array_like
        The recorded filtered signal.

    Returns
    -------
    out : int
        Channel Separation Index in dB.
    """



    return None
def PE(sig, sig_rec):
    """Calculate the Performance Error Index.

    Parameters
    ----------
    sig : array_like
        The dry signal.
    sig_rec : array_like
        The recorded filtered signal.

    Returns
    -------
    out : int
        Performance Error Index in dB.
    """

    

    return None