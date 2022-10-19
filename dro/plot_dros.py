#!/usr/bin/env python

import numpy as np
import sys
from matplotlib import pyplot as plt, rc, ticker
from mpl_toolkits.mplot3d import Axes3D
from astropy import constants as const

def plot_dros(task):

    plt.clf()
    rc('font', size=12)
    fig =   plt.figure()
    fig.set_figwidth(5.5)
    fig.set_figheight(5)
    fig.subplots_adjust(left=0.15, right=0.97, top=0.97, bottom=0.1)
    ax  =   plt.gca()
    ax.set_aspect('equal')
    
    for stn in task.stns:
        
#        if stn.cb != 'Earth':
#            print('Skip non Earth crs...')
#            continue

        if stn.type != 'dro':
            continue
# as a demonstration, we only plot the separation with the Earth
        p   =   stn.p_rot
#        print(p)
        plt.plot(p[:, 0], p[:, 1], ls='-', label='orbit %s' % (stn.ob_id))

    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    rlim    =   1.05
    plt.xlim(1-rlim, 1+rlim)
    plt.ylim(-rlim, rlim)
    plt.xlabel('X')
    plt.ylabel('Y')

#    plt.legend(ncol=5, loc='upper right')
    name    =   'dros.png'
    plt.savefig(name)

def rad2mas(rad):
    return rad / np.pi * 180. * 3600E3

def rad2min(rad):
    return rad / np.pi * 180. * 60.

def rad2deg(rad):
    return rad / np.pi * 180.

def mas2rad(mas):
    return mas / 1E3 / 3600 / 180. * np.pi
