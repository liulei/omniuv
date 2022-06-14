#!/usr/bin/env python

import numpy as np
import sys
from matplotlib import pyplot as plt, rc
from mpl_toolkits.mplot3d import Axes3D
from astropy import constants as const

def upsample(arr0, nup, nc1):
    
    print('Upsampling: %d' % (nup))

    s0  =   arr0.shape
    assert s0[0] == s0[1]
    nc0 =   s0[0]
    
    nc0_crop    =   nc1 // nup
    _arr0   =   arr0.copy()
    if nc0 > nc0_crop:
        print('img 0: size %d, expected %d' % (nc0, nc0_crop))
        dn  =   nc0 - nc0_crop
        nc0 =   nc0_crop
        if dn // 2 != 0:
            dn  +=  1
        print('dn: %d' % (dn))
        i0 =   dn // 2
        if i0 == 0:
            i0  =   1
        _arr0   =   _arr0[i0:-i0, i0:-i0]
    
    _arr1   =   _arr0.repeat(nup, axis=0).repeat(nup, axis=1)
    if nc1 == _arr1.shape[0]:
        return _arr1

    arr1    =   np.zeros((nc1, nc1), dtype=float)
    i1      =   (nc1 - _arr1.shape[0])//2
    arr1[i1:-i1, i1:-i1]    =   _arr1[:, :]

    print(arr1.shape)
    return arr1

def lonlat_deg2xyz(lon, lat):

    lon *=  (np.pi/180.)
    lat *=  (np.pi/180.)

    R   =   const.R_earth.value
    x   =   R * np.cos(lat) * np.cos(lon)
    y   =   R * np.cos(lat) * np.sin(lon)
    z   =   R * np.sin(lat)

    return x, y, z


def plot_el_sep(task):

    plt.clf()
    fig =   plt.figure()
    fig.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)
    
    for stn in task.stns:
        
#        if stn.cb != 'Earth':
#            print('Skip non Earth crs...')
#            continue

        if stn.type == 'fixed':
            angle   =   stn.el[0]
        elif stn.type == 'orbit':
# as a demonstration, we only plot the separation with the Earth
            if not 'Earth' in stn.sep[0].keys():
                continue
            angle   =   stn.sep[0]['Earth']

        angle   *=  (180./np.pi)
            
        idt =   stn.idt_avail[0]
        plt.plot(task.ts[idt]/3600., angle[idt], ls='-', label=stn.name)
    plt.xlim(0, 24)
    plt.ylim(0, 180.)
    plt.xlabel('Time [Hour]')
    plt.ylabel('Elevation/Seperation [degree]')

    plt.legend(ncol=5, loc='upper right')
    plt.savefig('el_sep.png')

# Input:
# uvw:      in units of wave number, 2-D array
# cellsize: in mas
# nc:       grid number along one axis
# name:     saved figure file name
def plot_uv(uvw, cellsize, nc, name):

    uv  =   uvw[:, :2]

    urange  =   1. / mas2rad(cellsize)
    umin    =   -urange * 0.5  

    sc  =   1E6
    plt.clf()
    fig =   plt.figure()
    ax  =   fig.add_subplot()
    ax.set_aspect('equal')
    plt.plot(uv[:, 0]/sc, uv[:, 1]/sc, marker = '.', c = 'steelblue', \
            ls = 'none', ms = 2)
    ax.set_xlabel('$u [10^6\lambda]$')
    ax.set_ylabel('$v [10^6\lambda]$')
    ax.set_xlim(umin/sc, -umin/sc) 
    ax.set_ylim(umin/sc, -umin/sc) 
    plt.savefig(name)

# Input:
# beam:     2-D array
# cellsize: in mas
# nc:       grid number along one axis
# name:     saved figure file name
def plot_beam(beam, cellsize, nc, name):

    cellsize    /=  60E3

    hcs     =   cellsize * 0.5
    s       =   cellsize * nc * 0.5
    vmin    =   np.min(beam)
    vmax    =   np.max(beam)

    plt.clf()
    fig =   plt.figure()
    ax  =   fig.add_subplot()
    im  =   ax.imshow(beam, vmin = vmin, vmax = vmax, \
                origin = 'lower', cmap = plt.get_cmap('rainbow'), \
                extent = (-s, s, -s, s))
    ax.set_xlabel('X [arcmin]')
    ax.set_ylabel('Y [arcmin]')
#    cb  =   plt.colorbar(im, ax = ax)
#    cb.ax.set_ylabel('Flux [Jy]', rotation=90, va='bottom')
    cb  =   plt.colorbar(im, orientation='vertical')
    cb.ax.set_ylabel('Strength')

    plt.savefig(name)

# Input:
# image:    2-D array
# cellsize: in mas
# nc:       grid number along one axis
# name:     saved figure file name
def plot_image(image, cellsize, nc, name):

    cellsize    /=  60E3

    hcs     =   cellsize * 0.5
    s       =   cellsize * nc * 0.5
    vmin    =   np.min(image)
    vmax    =   np.max(image)

    plt.clf()
    fig =   plt.figure()
    ax  =   fig.add_subplot()
    im  =   ax.imshow(image, vmin = vmin, vmax = vmax, \
                origin = 'lower', cmap = plt.get_cmap('rainbow'), \
                extent = (-s, s, -s, s))
    ax.set_xlabel('X [arcmin]')
    ax.set_ylabel('Y [arcmin]')
#    cb  =   plt.colorbar(im, ax = ax)
#    cb.ax.set_ylabel('Flux [Jy]', rotation=90, va='bottom')
    cb  =   plt.colorbar(im, orientation='vertical')
    cb.ax.set_ylabel('Flux [Jy]')

    plt.savefig(name)

def plot_crs(task):

    fig =   plt.figure()
    ax  =   fig.add_subplot(projection = '3d')
    for stn in task.stns:
        
#        if stn.cb != 'Earth':
#            print('Skip non Earth crs...')
#            continue

# Retrive the CRS coordinates of each station
        x, y, z =   zip(*stn.p_crs.tolist())
        ax.plot(x, y, z)
       
    plt.show()

def plot_lcs(task):

    fig =   plt.figure()
    ax  =   fig.add_subplot(projection = '3d')
    for stn in task.stns:
        if stn.cb != 'Moon':
            print('plot_lcs(): skip none lunar station %s' % (stn.name))
            continue
# The LCS coordinates are only available for Moon related stations 
# (orbit, surface)
        x, y, z =   zip(*stn.p_lcs.tolist())
        ax.plot(x, y, z)

    Rm  =   1737.4E3 # Moon radius, in m
    lim =   (-Rm, Rm)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()

def mas2rad(mas):
    return mas / 1E3 / 3600 / 180. * np.pi

# Input:
# hl, hh:   orbital height in m
# R:        celestial object radius in m
# Output:
# a:        semi major axis in m
# e:        eccentricity
def h2ae(hl, hh, R):
    
    a   =   (hl + hh + 2 * R) * 0.5
    c   =   a - hl - R
    e   =   c / a
    return a, e

# Calculate the beam shape using TPJ's algorithm in DIFMAP
# Input:
# uv:   2-D array, uv in wave number
# Output:
# Major and minor size in rad, angle in rad
def calc_beam_param(uv, ws=None):

    if ws is None:
        ws  =   np.ones(uv.shape[0])

    uu  =   uv[:, 0]
    vv  =   uv[:, 1]

    wmean    =   np.mean(ws)
# u, v should have been used! Otherwise multiply with lam
    muu =   np.average((uu**2) * ws) / wmean
    mvv =   np.average((vv**2) * ws) / wmean
    muv =   np.average((uu*vv) * ws) / wmean
    
    fudge   =   0.7
    ftmp    =   np.sqrt((muu-mvv)**2 + 4.*muv**2)
    e_bpa   =   -0.5 * np.arctan2(2.*muv, muu-mvv)
    e_bmin  =   fudge/np.sqrt(2.*(muu+mvv) + 2.*ftmp)
    e_bmaj  =   fudge/np.sqrt(2.*(muu+mvv) - 2.*ftmp)

    if e_bmin > e_bmaj:
        e_bmaj, e_bmin  =   e_bmin, e_bmaj

    return e_bmaj, e_bmin, e_bpa

def rad2mas(rad):
    return rad / np.pi * 180. * 3600E3

def rad2min(rad):
    return rad / np.pi * 180. * 60.

def rad2deg(rad):
    return rad / np.pi * 180.

def mas2rad(mas):
    return mas / 1E3 / 3600 / 180. * np.pi
