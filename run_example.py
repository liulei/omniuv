#!/usr/bin/env python

import sys
import numpy as np
from datetime import datetime
from tool import *

# Specify the path that contains the omniuv package. 
# Necessary when your source code and the pacakge are not in the same
# folder.
sys.path.append('.')

# Import the specific OmniUV package
from omniuv import * 

with_moon   =   True

def gen_ts1():
    ts  =   86400*0.0 + np.arange(0, 24 * 60 * 60, 60, dtype = float)
    return ts

def gen_ts2():
# 2s AP, 1 hour per day, 28 days
    ts  =   []
#    for i in range(28):
    for i in [0, 7, 14, 21]:
        _ts =   (i+0.5) * 86400 + np.arange(0, 3600, 60, dtype = float)
        ts.append(_ts)
    ts  =   np.concatenate(ts)
    return ts
   
def gen_ts3():
# 2s AP, 0.5 hour per day, twice per day, 28 days
    ts  =   []
    for i in range(28):
#    for i in [0, 7, 14, 21]:
        _ts =   (i    ) * 86400 + np.arange(0, 900, 60, dtype = float)
        ts.append(_ts)
        _ts =   (i+0.5) * 86400 + np.arange(0, 900, 60, dtype = float)
        ts.append(_ts)
    ts  =   np.concatenate(ts)
    return ts
 
def main():

# Instance of Task class. Entry of the program.
    task    =   Task()
    
# OmniUV supports uvw calculations for multiple sources.
    srcs    =   []

# Instance of Source class. 
# "src" here refers to phase center
    src     =   Source()
    src.name=   'T1'
# ra and dec are provided in radian
    src.ra   =   180. / 180.  * np.pi
    src.dec  =   30.0 / 180. * np.pi
    srcs.append(src)

#    src     =   Source()
#    src.name=   'M87'
#    src.ra   =   187.705930 / 180.  * np.pi
#    src.dec  =   12.391123 / 180. * np.pi
#    srcs.append(src)

#    src     =   Source()
#    src.name=   'SouthPole'
#    src.ra   =  0.0
#    src.dec  =  -np.pi/2
#    srcs.append(src)
#

# Register srcs in task. 
    task.set_srcs(srcs)

# If you are not familier with EOPs, set them to 0
    tmu     =   35.
    dut1    =   -2.211724
    xp      =   0.027840
    yp      =   0.370850
    task.set_eops(tmu = tmu, dut1 = dut1, xp=xp, yp=yp)

# Starting date and time
    t0  =   datetime(2020, 3, 11, 0, 0, 0)

# Time series to calculate coordinate and uvw, as offset to t0,
# in second. 
# Two functions provides different schedule.
    if with_moon:
        ts  =   gen_ts3()
    else:
        ts  =   gen_ts1()
# Register time series
    task.set_ts(t0, ts)

    stns    =   []

    hl  =   10000E3
    hh  =   100000E3

# Calculate semi-major and eccentricity for Earth orbit.
    a, e    =   util.h2ae(hl, hh, util.Re)
    print('Earth satellite, a: %.2f x 1E4 km, e: %.2f'\
        % (a/1E7, e))

# Earth orbit station
    s =   EarthOrbit('t1')
# Specify the seperation of the target source from the celestial 
# object, in degree.
# If the seperation at one moment is less than this value, 
# uvw calculations for this moment is skipped.
# You may set multiple objects by specifying their name.
# E.g.: set_sep_min_deg(Earth=5.0, Moon=1.0, Sun=10.0)
# At present separation calc. with Earth, Moon and Sun are supported.
    s.set_sep_min_deg(Earth = 5.0)

# Standard six orbital elements, a in meter, angles in radian
# set t_ref to 2020-01-01T00:00:00 UTC 
    s.set_orbit(a=a, e=e, i=np.pi/6, raan=0.0, arg_pe=0.0, M0=0.0, \
            t_ref=datetime(2020, 1, 1, 0, 0, 0))

# 3.6 cm, 30 m diameter
    s.set_SEFD(225.0) # in Jy
    stns.append(s)

    s =   EarthOrbit('t2')
    s.set_sep_min_deg(Earth = 5.0)
# Standard six orbital elements, a in meter, angles in radian
# t_ref defaults to task.t0
    s.set_orbit(a=a, e=e, i=-np.pi/6, raan=0.0, arg_pe=0.0, M0=np.pi, t_ref=t0)
# 3.6 cm, 30 m diameter
    s.set_SEFD(225.0)
    stns.append(s)

# TMRT
    g   =   EarthFixed('TMRT')
# Terrestial reference frame system coordinate, in meter
    g.set_trs(np.array([-2826708.82869, 4679236.99691, 3274667.48709]))
# Minimum elevation, in degree
    g.set_el_min_deg(15.0)
    g.set_SEFD(48.0)
    stns.append(g)

# Ef
    g   =   EarthFixed('EF')
# Terrestial reference frame system coordinate
    g.set_trs(np.array([4033947.23550, 486990.79430, 4900431.00170]))
# Minimum elevation, in degree
    g.set_el_min_deg(15.0)
    g.set_SEFD(20.0)
    stns.append(g)

# Lunar fixed, far side of the moon
# Please contact the author (liulei@shao.ac.cn) for collaboration.
#    lf   =   LunarFixed('M1')
# Set lunar fixed frame coordinate (Principal Axis)
# This one in the farside of the Moon
#    lf.set_lfs(np.array([-util.Rm, 0.0, 0.0])) 
#    lf.set_el_min_deg(15.0) 
# Specify the celestial object that you want to calculate the 
# seperation and availabilty accordingly.
#    lf.set_sep_min_deg(Earth=5.0)
#    lf.set_SEFD(2028.0)
#    if with_moon:
#        stns.append(lf)

# Lunar orbit
    lo =   LunarOrbit('lo')
    lo.set_sep_min_deg(Moon=1.0, Sun=10.0)
# t_ref defaults to task.t0
    lo.set_orbit(a=util.Rm*3, e=0.0, i=0.0, raan=0.0, arg_pe=0.0, M0=0.0, t_ref=t0)
    lo.set_SEFD(507)
    if with_moon:
        stns.append(lo)

    l2_es   =   EarthSunL2('es')
    l2_es.set_sep_min_deg(Earth=5.0)
    l2_es.set_SEFD(507)
#    stns.append(l2_es)

    l2_me   =   MoonEarthL2('me')
    l2_me.set_sep_min_deg(Moon=1.0)
    l2_me.set_SEFD(507)
    if with_moon:
        stns.append(l2_me)

# Gain error, see Explanation of Eq. 4.
    task.gain_error =   0.1
    
# Register all stations
    task.set_stns(stns)

# Calculate baseline uvw, in the shape (bl, target, nt, 3(uvw))
# Assuming there are 3 stations, 2 targets.
# The baseline (bl) indices are: 
# bl0: stn0-stn1, bl1: stn0 - stn2, bl2: stn1 - stn2
#    uvw_bl = uvw_bls[2][1] 
# retrieves a 2-D array in the shape (nt, 3) for baseline stn1-stn2 
# of source 1 (0 indexed).
    uvw_bls =   task.calc_uvw_bl()

# One may also retrieve the station crs coordinate and uvw data 
# explicitely:
#    stn     =   task.stns[0]
#    p_crs   =   stn.calc_crs()
#    uvw     =   stn.calc_uvw()

# Note! uvw is calculated only when both stations are available.
# OmniUV provides a list that contains the indices of the available 
# moments in ts for all baselines and srcs:
#    idt_bls =   task.idt_avail
# idt_bl retrives the indices of avail. moments for bl 2, src 1:
#    idt_bl =   idt_bls[2][1]
# The corresponding time in ts array could be retrived as:
#    ts_avail    =   task.ts[idt_bl]

# Aux function to flat the uvw result
# src, ts, uvw(3)
    uvw_flat    =   task.flat_uvw_bl(uvw_bls)

# Select uv of src 0:
# ts, uvw(3)
    uv  =   uvw_flat[0][:, :2]

# Add conjugate:
    uv  =   np.concatenate([uv, -uv], axis = 0)

# Plot to check CRS and LCS calculation result
# Refer to the corresponding functions in tool.py to find out how 
# these data are retrived.
#    tool.plot_crs(task)
#    tool.plot_lcs(task)

##### Parameters for vis calculatio and imaging ######

# Sky frequency of each IF. The wave length of each IF is calculated with these
# frequenceis and has ** nothing ** to do with the bandwidth.
    task.freqs  =   np.array([8.4E9])     # in Hz
 
# t_ap and bandwidth are ** only ** used for calculating system 
# noise. See Eq. 5 of the paper.

# Accumulation period (integration time)
    task.t_ap =   2. # in second, could be float, e.g. 0.128

# Bandwidth of each IF
    task.bandwidth  =   32E6    # in Hz

# Resolution calculations that helps determine the cellsize. 
# Usually the cell size is a fraction of the estimated resolution.
    uvmax   =   np.max(np.abs(uv), axis = 0) / 1E7 # in 10^4 km
    rmax    =   np.max(uvmax) # Maximum uv along certain axis
    print('uv max: %.1f x 10^4 km, %.1f x 10^4 km, rmax = %.3f x 10^4 km' \
            % (uvmax[0], uvmax[1], rmax))
    lam     =   3E8 / task.freqs[0]
    rmax_lam=   rmax / lam # Maximum uv in wave number
    res     =   1. / rmax_lam / 1E7 / np.pi * 180 * 3600E3 # in mas
    print('res at %.1f GHz: %f mas.' % \
            (task.freqs[0]/1E9, res))

# Gridding cell size, in the current version of OmniUV, always given
# in ** mas **, even for wide field array.
    task.cellsize   =   0.005 # in mas

# Gridding shape, which leads to an image size of nc X nc.
    task.nc         =   128

##### Preparation of sample image for simulation #####

# Select the first src, retrive ra and dec
    src =   task.srcs[0]
    ra  =   src.ra
    dec =   src.dec

#    ds  =   0.5 # in mas
    ds      =   task.cellsize * 30. # in mas
    ddec    =   util.mas2rad(ds)
# For RA, take the coordinate contraction into account.
    dra     =   util.mas2rad(ds) / np.cos(dec)

# Instance of Image class
    img =   Image(src)

# Fluxes of each point, in Jy
    img.fluxes   =   [5.0,   5.0,    5.0,        5.0,    5.0]

# Ra and dec, in radian:
    img.ras     =   [ra,    ra+dra, ra,         ra-dra, ra]
    img.decs    =   [dec,   dec,    dec+ddec,   dec,    dec-ddec]

# Total number of pixels
    img.npixel  =   len(img.fluxes)

# Calculate lmn for each pixel in the image
# This part is exposed to users purposely. See Image section in 
# GitHub repo for a detailed explanation.

# rqu: pixel position in CRS, or s0 in GitHub document.
# lmn: shape (npixel, 3), projection of rqu (s0) in uvw system.
    lmn    =   []
    for i in range(img.npixel):
# util.crs2rqu converts R.A. and decl. to unit vector form (rqu or s0)
        rqu =   util.crs2rqu(img.ras[i], img.decs[i])
        lmn.append(np.dot(img.ruvw0, rqu))
    img.lmn    =   np.array(lmn)
    src.img =   img

# Generate visibility of each baseline using FFT method
# Input: src generated above
# Output: bls, see below.
# bls is a baseline list. The index of each baseline is the same 
# as the output of cacl_uvw_bl.
# bl = bls[bl_id] is a dict, which has the following keys:
# bl['uvw_m']:      2-D float array, shape (nt, 3), uvw in meter
# bl['t']:          1-D array, time of available moments for the 
#                   corresponding  uvw_m, see explanation of 
#                   idt_avail after explanation of calc_uvw_bl()
# bl['uvw_wav']:    3-D float array, shape (nt, nfreq, 3), 
#                   uvw in wavenumber
# bl['vis']:        2-D complex array, shape (nt, nfreq), 
#                   visibility at each available moment and freq
    bls =   task.gen_vis_fft(src)

# Add noise for each baseline. See Sec. 2.3 for an explanation.
# Input: src, bls generated above
# Output: bls orgainzed in the same structure as input
    bls =   task.vis_add_noise(src, bls)

# The visibility simumation result together with the calculated UVW 
# could be exported to FITS-IDI format:
    task.to_fitsidi('EXAMPLE', bls)

# Radial weighting, default to False
    task.do_rad     =   False

# Uniform weighting, default to False
    task.do_unif    =   False

# For diffuse source simulation, image flux must be corrected according
# to the beam size and source image resolution. Default to False. 
    task.do_beam_correction =   False
# If do_beam_correction is on, cellsize of souce image must be set 
# for normalization. Default to the cellsize of the output image, 
# you may set it according 
#    if task.do_beam_correction:
#        task.cs_src_rad =   util.mas2rad(task.cellsize)

# Generate beam pattern.
# Input: bls above
# Output: beam pattern, 2-D float array in the shape (nc, nc)
    beam        =   task.gen_beam(bls)

# Generate image.
# Input: bls above
# Output: 
#   uv: 2-D float array, in the shape (nvis, 2)
#   image: 2-D float array, in the shape (nc, nc)
    uv, image   =   task.gen_image_fft(bls)

# Or you may use the DFT method:
#    uv, image   =   task.gen_image_direct(bls)

    nc  =   task.nc # grid size, same as in image and beam shape
    cs  =   task.cellsize # cellsize, in mas

    plot_beam(beam, cs, nc, 'example_beam.png')
    plot_uv(uv, cs, nc, 'example_uv.png')
    plot_image(image, cs, nc, 'example_image.png')
    
if __name__ == '__main__':
    main()
