#!/usr/bin/env python

import sys, os
import numpy as np
from datetime import datetime
from plot_dros import plot_dros

# Path that contains the uvom folder
sys.path.append('/home/liulei/program/VLBI')
from uvom import *
from matplotlib import pyplot as plt 

def gen_ts1():
#    ts  =   np.arange(0, 14 * 86400, 600, dtype = float)
#    ts  =   np.arange(0, 4.34 * 2.69359 * 86400, 600, dtype = float)
    ts  =   np.arange(0, 3.897 * 2.69359 * 86400, 600, dtype = float)
#    ts  =   np.arange(0, 31. * 86400, 600, dtype = float)
    return ts

def main():

    task    =   Task()

    task.nc =   128
    task.cellsize   =   1 # in mas
   
    srcs    =   []
    src     =   Source()
    src.name=   'T1'
    src.ra   =  np.pi/6
    src.dec  =  0.0
    srcs.append(src)

    task.set_srcs(srcs)

# If you are not familier with EOP, keep dut1, tmu, PN, W unchanged.
    tmu     =   35.
    dut1    =   -2.211724
    xp      =   0.027840
    yp      =   0.370850
    task.set_eops(tmu = tmu, dut1 = dut1, xp=xp, yp=yp)

# Starting date and time
#    t0  =   datetime(2020, 3, 11, 0, 0, 0)
    t0  =   datetime(2020, 3, 25, 0, 0, 0)
#  time series to calculate crs and uvw, in second
    ts  =   gen_ts1()
    task.set_ts(t0, ts)

    stns    =   []

# Calculate orbit in inertial frame. Telescope trajectory is calculated 
# based on equation of motion. 
# do_dro_rot = False
# "p_crs" is first calculated and is then transformed to rotation frame
# for "p_rot". The result can be used for further uv calculation.
# For testing, uncomment the following 4 lines:
#    s   =   DRO('S0')
#    s.set_param(orbit_id=5, method='RK45', max_step=10.0)
#    s.do_dro_rot    =   False
#    stns.append(s)

# Calculate orbit in rotation frame. For demonstration only!!!
# do_dro_rot = True
# In this mode:
# Only "p_rot" is calculated for 20 initial conditions.
# "p_crs" is not calculated, which means the result cannot be used for 
# further uv calculation.

    for id in np.arange(20):
        s   =   DRO('S%d' % (id))
        s.set_param(orbit_id=id, method='RK45', max_step=60.0)
        s.do_dro_rot    =   True
        stns.append(s)

# Lunar fixed, far side of the moon
    lf   =   LunarFixed('M1')
# Set lunar fixed frame coordinate (Principal Axis)
# This one in the farside of the Moon
    lf.set_lfs(np.array([-util.Rm, 0.0, 0.0])) 
    lf.set_el_min_deg(15.0) 
# Specify the celestial object that you want to calculate the 
# seperation and availabilty accordingly.
    lf.set_sep_min_deg(Earth=5.0)
    lf.set_SEFD(2028.0)
    stns.append(lf)

    task.gain_error =   0.1

    task.set_stns(stns)

# Calculate baseline uvw, result in the form: bl, target, ts, uvw(3)
# Note! uvw is output only when both stations are available for obs!
#    uvw_bls =   task.calc_uvw_bl()
    task.calc_crs_stn()
    
# plot orbit in Moon-Earth rotation frame 
    plot_dros(task)
    print('Exit after plot_dros dump')
    sys.exit(0)
   
if __name__ == '__main__':
    main()
