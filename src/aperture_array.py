#!/usr/bin/env python

# Copyright 2022-2024. Lei Liu (liulei@shao.ac.cn) all rights reserved.

import numpy as np
from matplotlib import pyplot as plt, rc, ticker

d2r =   np.pi/180.

def plotxyproj(arr, name):

# Set theta plotting range: 90.-th_max to 90.
    th_max  =   10.
# Pixel number
    ng      =   200

    dth     =   th_max * 2 / ng
    oft     =   (-ng/2+0.5) * dth

    ws  =   arr.ws_table.copy()
    ws  =   20 * np.log10(ws / np.max(ws))

    def xy2pol(iy, ix):
        x   =   ix * dth + oft
        y   =   iy * dth + oft

        th  =   np.pi/2 - np.sqrt(x**2+y**2) * d2r
# CW, zero at North direction
        ph  =   np.arctan2(x, y)

        ith =   int((th-arr.ths[0]) / arr.dth + 0.5)
        iph =   int((ph-arr.phs[0]) / arr.dph + 0.5)

        return ws[ith, iph]
        
    beam    =   np.zeros((ng, ng), dtype=float)
    for iy in range(ng):
        for ix in range(ng):
            beam[iy, ix]    =   xy2pol(iy, ix)

    plt.clf()
    rc('font', size=13)
    fig =   plt.figure()
    fig.set_figwidth(6.3)
    fig.set_figheight(5)
    fig.subplots_adjust(left=0.1, right=0.97, top=0.97, bottom=0.08)
    s   =   th_max
    im  =   plt.imshow(beam, vmin=-50, vmax=0, \
            cmap=plt.get_cmap('rainbow'), \
            extent=(-s, s, -s, s), origin='lower')
    ax  =   plt.gca()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
    cb  =   plt.colorbar(im, orientation='vertical')
    cb.ax.set_ylabel('Power [dB]')
    plt.savefig('%s.png' % (name))
        
class ApertureArray(object):

    def __init__(self, lam, pos_tiles, tiles = []):

        self.type       =   ''
        self.lam        =   lam
        self.pos_tiles  =   pos_tiles
        self.k          =   2. * np.pi / self.lam
        self.tiles      =   tiles

    def calc_ws(self, th0, ph0, ths, phs, ws = None):

        self.ths    =   ths
        self.phs    =   phs
        self.dth    =   ths[1] - ths[0]
        self.dph    =   phs[1] - phs[0]

        ntile       =   self.pos_tiles.shape[0]
        nth         =   len(ths)
        nph         =   len(phs)

        self.ntile  =   ntile
        self.nth    =   nth
        self.nph    =   nph

#        if dtype(ws) == str 
#            if ws == 'uniform':
#                return np.ones(nth, nph), dtype = float)
#            else:
#                print('ApertureArray.calc_ws(): unsupported ws type of %s!' % (ws))
#                sys.exit(0)
#        if dtype(ws) == np.ndarray:
#            if ws.shape != (nth, nph):
#                print('ApertureArray.calc_ws(): ws array size unmached!')
#                print('Expected: ', (nth, nph), 'get: ', ws.shape)
#                sys.exit(0)
#            self.ws_table   =   ws.copy()
#            return self.ws_table
        
        hpi         =   np.pi/2
#        _ths        =   ths + hpi - th0
        cosths      =   np.cos(ths)
        sincosphs   =   np.array(list(zip(np.sin(phs), np.cos(phs))))

# (ntile)
        dpsi    =   self.k * np.cos(th0) * np.einsum('jk,k->j', \
                self.pos_tiles, [np.sin(ph0), np.cos(ph0)])
        dpsi    =   np.repeat(dpsi[:,    np.newaxis], len(ths), axis=1)
        dpsi    =   np.repeat(dpsi[:, :, np.newaxis], len(phs), axis=2)

# k * np.cos(theta) * (x * np.sin(phi) + y * np.cos(phi))
# (ntile,2), (nph, 2) -> (ntile, nph)
        s1  =   np.einsum('jk,lk->jl', self.pos_tiles, sincosphs)
# (nth), (ntile, nph) -> (ntile, nth, nph)
        psi  =   np.einsum('i,jl->jil', cosths, s1) * self.k - dpsi
# No sub tiles, uniform weight
        if self.tiles == []:
            ws_tile =   np.ones((ntile, nth, nph), dtype=float)
# Use weight provided by sub tiles
        else:
# (nth, nph)
            ws_tile =   self.tiles[0].get_ws(ths, phs)
# (ntile, nth, nph)
            ws_tile =   np.repeat(ws_tile[np.newaxis, :, :], \
                        ntile, axis=0)
# (ntile, nth, nph) -> (nth, nph)
        ws_table      =   np.einsum('jil->il', np.exp(1j * psi) * ws_tile)
        self.ws_table   =   np.absolute(ws_table)

        return self.ws_table

# This method could be overloaded by the user to provde weight.
    def get_ws(self, ths, phs):

        iys     =   ((ths - self.ths[0]) / self.dth + 0.5).astype(int)
        ixs     =   ((phs - self.phs[0]) / self.dph + 0.5).astype(int)

        iXs, iYs=   np.meshgrid(ixs, iys)

        ids     =   iYs.flatten() * self.nph + iXs.flatten()

        vals    =   self.ws_table.flatten()[ids].reshape(iXs.shape)

        return vals
 
def create_tile():

    lam     =   0.3
    nx      =   16
    ny      =   nx
    d       =   lam * 0.5
    x       =   (np.arange(nx) - nx/2 + 0.5) * d
    y       =   (np.arange(ny) - ny/2 + 0.5) * d

    X, Y    =   np.meshgrid(x, y)

    pos     =   np.array(list(zip(X.flatten(), Y.flatten())))

# Create tile, provide wavelength, position of each antenna in this
# tile, and tile list as input.
# We assume uniform weight for each ant. in this tile, therefore 
# a default empty list is provided.
    tile    =   ApertureArray(lam, pos, tiles=[])

    th_r    =   90.
    dth     =   0.1
    nth     =   int(th_r/dth+1E-6) + 1
    ths     =   np.linspace(90.-th_r, 90., nth) * d2r

    nph     =   360
    phs     =   np.linspace(0, 360., nph, endpoint=False) * d2r

    th0     =   90. * d2r
    ph0     =    0. * d2r

# Calculate weight at given phase center for theta and phi arrays.
    tile.calc_ws(th0, ph0, ths, phs)

    plotxyproj(tile, 'tile')

    return tile

def test_array():

    lam     =   0.3
    nx      =   16
    ny      =   nx
    d_tile  =   lam * 0.5 * 16
    x       =   (np.arange(nx) - nx/2 + 0.5) * d_tile
    y       =   (np.arange(nx) - nx/2 + 0.5) * d_tile

    X, Y    =   np.meshgrid(x, y)

# Prepare xy coordinates of each tile:
    pos     =   np.array(list(zip(X.flatten(), Y.flatten())))

    tile    =   create_tile()

# Create an aper. array, provide wavelength, position of each 
# tile, and a tile list as input.

# At present we assume same configuration for each tile, therefore
# one tile is enough

# To specify your own tile beam pattern, you may inherit the 
# ApertureArray class and overload the get_ws() method for providing
# the user defined weight.
    aarr    =   ApertureArray(lam, pos, tiles=[tile])

    th_r    =   15.
    dth     =   0.1
    nth     =   int(th_r/dth+1E-6) + 1
    ths     =   np.linspace(90.-th_r, 90., nth) * d2r

    nph     =   360
    phs     =   np.linspace(0, 360., nph, endpoint=False) * d2r

# Speficy phase center:
    th0     =   90. * d2r
    ph0     =   0. * d2r

# Calculate weight for the given theta and phi arrays, at phase 
# centre (th0, ph0)
    aarr.ws =   aarr.calc_ws(th0, ph0, ths, phs)

# Plot beam pattern. the plotting range is set with th_max in this 
# functiion from 90.-th_max to 90.
    plotxyproj(aarr, 'array_%d_%d' % (ph0/d2r, th0/d2r))

if __name__ == '__main__':
    test_array()
    
