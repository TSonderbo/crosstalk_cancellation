import numpy as np
import scipy as sp
import math

fs = 48000 #Sample rate
T = 1/fs #Sampling period

c = 330 #wave velocity

a = 0 #free numerical parameter
b = 0 #free numerical parameter

lambdaSq = 1**2
X = c / (fs * np.sqrt(lambdaSq))#Grid spacing
d1 = lambdaSq * (1 - 4 * a + 4 * b)
d2 = lambdaSq * (a - 2 * b)
d3 = lambdaSq * b
d4 = 2 * (1 - 3 * lambdaSq + 6 * lambdaSq * a - 4 * b * lambdaSq)


pPrev = np.zeros(())
p = np.zeros(())
pNext = np.zeros(())

for l in range():
    for m in range():
        for i in range():
            pNext[l,m,i] = d1 * (p[l+1,m,i] + p[l-1,m,i] + p[l,m+1,i] + p[l,m-1,i] + p[l,m,i+1] + p[l,m,i-1]) \
                + d2 * (p[l+1,m+1,i] + p[l+1,m-1,i] + p[l+1,m,i+1] + p[l+1,m,i-1] + p[l,m+1,i+1] + p[l,m+1,i-1] \
                + p[l,m-1,i+1] + p[l,m-1,i-1] + p[l-1,m+1,i] + p[l-1,m-1,i] + p[l-1,m,i+1] + p[l-1,m,i-1]) \
                + d3 * (p[l+1,m+1,i+1] + p[l+1,m-1,i+1] + p[l+1,m+1,i-1] + p[l+1,m-1,i-1] + p[l-1,m+1,i+1] \
                + p[l-1,m-1,i+1] + p[l-1,m+1,i-1] + p[l-1,m-1,i-1]) \
                + d4 * p[l,m,i] - pPrev[l,m,i]