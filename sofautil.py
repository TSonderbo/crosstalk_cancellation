import numpy as np
import scipy as sp

def get_nearest_coord(HRTF, az, el):

    source_positions = HRTF.Source.Position.get_values(system="spherical")

    origin = []

    #Find index which is closest to az
    azimuths = np.abs(source_positions[:,0] - az)
    i = np.where(azimuths == azimuths.min())
    
    #Find index which is closest to el
    el += 90
    elevations = np.array(source_positions[:,1]) + 90
    elevations = np.abs(elevations - el)
    j = np.where(elevations == elevations.min())

    k = np.intersect1d(i, j)[0]

    return k, source_positions[k,:]