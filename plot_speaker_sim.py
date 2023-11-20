import numpy as np
import scipy as sp
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.animation as ani

def plot_sim(coords, f, phi):

    Rspeaker = coords[2,:] #Speaker Radius (m) - i.e distance from equilibrium
    margin = 0.5 #Padding for the plot
    c = 340 #Wavespeed
    fs = 48000 #Samplerate
    t = c/fs #Timestep
    k = 2 * math.pi * f / c
    speakerAngle = coords[0,:] #List of speaker angles to calculate for
    n = 0 #Index
    start = -1 * np.max(Rspeaker)-margin
    stop = np.max(Rspeaker)+margin
    xy = np.linspace(start, stop, round(2*(np.max(Rspeaker)-margin)/t)) #X and Y axis
    P = np.empty((len(xy),len(xy),len(speakerAngle))) #Output array
    p0 = np.empty((2,2)) #Calculated coordinates for speakers

    for spk in speakerAngle:
        p0[n,:] = Rspeaker[n]*np.array([np.cos(spk/360*2*math.pi), np.sin(spk/360*2*math.pi)]) # Coords
        xx = np.ones((round(2*(Rspeaker[n]-margin)/t),1))*xy
        yy = np.flipud(xx.T)
        xones = np.ones((np.size(xx,0),1))
        p0x = p0[n,0]*xones
        p0y = p0[n,1]*xones

        R = np.sqrt((p0x - xx)**2 + (p0y - yy)**2)
        P[:,:,n] = 1 * np.sin(k*R + phi[n])/R 

        n+=1

    plt.imshow(np.sum(P,2), extent=[-200, 200, -200, 200])
    plt.xlabel("Distance (cm)")
    plt.ylabel("Distance (cm)")
    #cmap = mpl.cm.cool
    norm = mpl.colors.Normalize(vmin=-1, vmax=1)
    plt.colorbar(mpl.cm.ScalarMappable(norm=norm), orientation='vertical', label='Amplitude')

    plt.show()

def animate_sim(coords, f, phi):

    Rspeaker = coords[2,:] #Speaker Radius (m) - i.e distance from equilibrium
    margin = 0.5 #Padding for the plot
    c = 340 #Wavespeed
    fs = 48000 #Samplerate
    t = c/fs #Timestep
    k = 2 * math.pi * f / c
    speakerAngle = coords[0,:] #List of speaker angles to calculate for
    n = 0 #Index
    start = -1 * np.max(Rspeaker)-margin
    stop = np.max(Rspeaker)+margin
    xy = np.linspace(start, stop, round(2*(np.max(Rspeaker)-margin)/t)) #X and Y axis
    P = np.empty((len(xy),len(xy),len(speakerAngle))) #Output array
    p0 = np.empty((2,2)) #Calculated coordinates for speakers

    fig, ax = plt.subplots()

    for spk in speakerAngle:
        p0[n,:] = Rspeaker[n]*np.array([np.cos(spk/360*2*math.pi), np.sin(spk/360*2*math.pi)]) # Coords
        xx = np.ones((round(2*(Rspeaker[n]-margin)/t),1))*xy
        yy = np.flipud(xx.T)
        xones = np.ones((np.size(xx,0),1))
        p0x = p0[n,0]*xones
        p0y = p0[n,1]*xones

        R = np.sqrt((p0x - xx)**2 + (p0y - yy)**2)
        P[:,:,n] = 1 * np.sin(k*R + phi[n])/R 

        n+=1

    axesImage = ax.imshow(np.sum(P,2), extent=[-200, 200, -200, 200], animated=True)

    def animate(i):
        n = 0
        for spk in speakerAngle:
            p0[n,:] = Rspeaker[n]*np.array([np.cos(spk/360*2*math.pi), np.sin(spk/360*2*math.pi)]) # Coords
            xx = np.ones((round(2*(Rspeaker[n]-margin)/t),1))*xy
            yy = np.flipud(xx.T)
            xones = np.ones((np.size(xx,0),1))
            p0x = p0[n,0]*xones
            p0y = p0[n,1]*xones

            R = np.sqrt((p0x - xx)**2 + (p0y - yy)**2)
            P[:,:,n] = 1 * np.sin(k*R + phi[n] - (i/10))/R 

            n+=1
        n = 0
        axesImage.set_data(np.sum(P,2))
        return [axesImage]

    animation = ani.FuncAnimation(fig, animate, interval=20, frames=200, blit = True)
    
    ax.set_xlabel("Distance (cm)")
    ax.set_ylabel("Distance (cm)")
    #norm = mpl.colors.Normalize(vmin=-1, vmax=1)
    #ax.colorbar(mpl.cm.ScalarMappable(norm=norm), orientation='vertical', label='Amplitude')

    plt.show()

    animation.save('basic_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])