#!/usr/bin/env python

import numpy as np
from jplephem.spk import SPK
from datetime import datetime, timedelta
from astropy import constants as const
import ctypes
import sys, os
from matplotlib import pyplot as plt, rc
from mpl_toolkits.mplot3d import Axes3D

gme     =   3.986004418e14 # [m^3/s^2] (REF: 1)
c       =   299792458.0
Re      =   const.R_earth.value
Rm      =   1737.4E3 # Moon mean radius, in m
Rs      =   6.957E8 # Sun radius, in m

r_ES_L1 =   0.00997040269846281 # L1 ES
r_ES_L2 =   0.0100371199532807 # L2 ES
r_ME_L1 =   0.150909863379802 # L1 ME
r_ME_L2 =   0.167802559989682 # L2 ME

# central celestial body
cbs =   { \
    'Earth' :    {'R': Re}, \
    'Moon'  :    {'R': Rm}, \
}

# 0: earth, 1: sun, 2: moon, 3: mercury, 4: venus, 5: mars, 6: jup, 
# 7: saturn, 8: uranus, 9: neptune

M_in_E   =   [  1.0, \
                const.GM_sun.value / const.GM_earth.value, \
                0.0123, \
                0.0553, \
                0.815, \
                0.107, \
                const.GM_jup.value / const.GM_earth.value, \
                95.159, \
                14.536, \
                17.147]

DT_JD   =   2400000.5
T0_MJD  =   datetime(1858, 11, 17, 0, 0, 0)

path_home   =   '.'

def load_spk():

    if 'spk' in globals():
        return

    global spk
    from jplephem.spk import SPK
    spk  =   SPK.open('%s/data/de421.bsp' % (path_home))

def load_pck():
    if 'pck' in globals():
        return

    global pck
    from jplephem.pck import PCK
    pck =   PCK.open('%s/data/moon_pa_de421_1900-2050.bpc' % (path_home))

# input: list of ids
def intersect1d_many(idss):

    if len(idss) == 1:
        return idss[0]
    _ids    =   np.intersect1d(idss[0], idss[1])
    for ids in idss[2:]:
        _ids    =   np.intersect1d(_ids, ids)
    return _ids

# t in utc in datetime format
def utc2mjd(t):
    return (t - T0_MJD).total_seconds() / 86400.
        
def crs2rqu(ra, dec):

    rq      =   np.array([  np.cos(dec) * np.cos(ra), \
                            np.cos(dec) * np.sin(ra), \
                            np.sin(dec)])
    return  rq / norm(rq)

def rqu2ruvw(rqu):

    rw  =   rqu 
    
# north pole
    rnp =   np.array([0., 0., 1.])

# u direction: rnp x rw
    ru  =   np.cross(rnp, rw)
    ru  =   ru / norm(ru)

# v direction: rw x ru
    rv  =   np.cross(rw, ru)
    rv  =   rv / norm(rv)

# w direction: rqu
    rw  =   rqu 
    
# north pole
    rnp =   np.array([0., 0., 1.])

# u direction: rnp x rw
    ru  =   np.cross(rnp, rw)
    ru  =   ru / norm(ru)

# v direction: rw x ru
    rv  =   np.cross(rw, ru)
    rv  =   rv / norm(rv)
    
    return (ru, rv, rw)

def calc_uvw(rqu, b):

    ru, rv, rw  =   rqu2ruvw(rqu)

    return np.array([np.dot(b, ru), np.dot(b, rv), np.dot(b, rw)])

libsofa =   None
path_libsofa    =   '%s/data/sofa/libsofa_c.so' % (path_home)
if os.path.exists(path_libsofa):
#    libsofa =   ctypes.CDLL('./libsofa_c.so')
    libsofa =   ctypes.CDLL(path_libsofa)
def xys2006a(tt):

#    date1   =   np.floor(tt)
#    date2   =   tt - date1
    date1   =   DT_JD
    date2   =   tt

    x   =   np.zeros(1, dtype=np.float64)
    y   =   np.zeros(1, dtype=np.float64)
    s   =   np.zeros(1, dtype=np.float64)

#    if libsofa is None:
#        return 0, 0, 0
    libsofa.iauXys06a(  ctypes.c_double(date1), \
                        ctypes.c_double(date2), \
                        ctypes.c_void_p(x.ctypes.data), \
                        ctypes.c_void_p(y.ctypes.data), \
                        ctypes.c_void_p(s.ctypes.data))

    return x[0], y[0], s[0]

def rotm(angle, tp):

    ca = np.cos (angle)
    sa = np.sin (angle)

    R = np.zeros((3, 3), dtype = np.float64)

    if tp == 1:
        R[0,0] = 1
        R[1,1] = ca
        R[1,2] = sa
        R[2,1] = -sa
        R[2,2] = ca

    if tp == 2:
        R[0,0] = ca
        R[0,2] = -sa
        R[1,1] = 1
        R[2,0] = sa
        R[2,2] = ca

    if tp == 3:
        R[0,0] = ca
        R[0,1] = sa
        R[1,0] = -sa
        R[1,1] = ca
        R[2,2] = 1

    return R

def as2rad(v):
    return v * np.pi / 180. / 3600.

def interp(n, y, x0):
    x   =   np.arange(n, dtype=float)
    p   =   np.polyfit(x, y, 2)
    y0  =   np.polyval(p, x0)
    return y0

def get_eop(eop, mjd):

    neop    =   eop.neop
    mjd0    =   eop.EOP_time[0]
    dmjd    =   mjd - mjd0
    assert dmjd >= 0 and dmjd < neop
    idx     =   int(mjd + 0.5 - mjd0)
    tmu     =   eop.tai_utc[idx]
    dut1    =   interp(neop, eop.ut1_utc, dmjd) 
    xp      =   interp(neop, eop.xpole, dmjd) 
    yp      =   interp(neop, eop.ypole, dmjd) 
    return tmu, dut1, as2rad(xp), as2rad(yp)

def calc_t2c_R(dut1, mjd):

# earth rotation angle
    ut   = mjd + dut1/86400.
    tu   = ut - 51544.5             # days since fundamental epoch
    frac = ut - np.floor(ut) + 0.5  # UT1 Julian day fraction
    fac  = 0.00273781191135448
    era = 2. * np.pi * (frac + 0.7790572732640 + fac * tu )
    era = era % (2 * np.pi)              # [rad]
# rotation around pole axis
    R     = rotm(-era,3)
    return R

def calc_t2c_PN_W(mjd, tmu, dut1, xp, yp):

    tt  =   mjd + (32.184 + tmu)/86400.

    tjc =(tt-51544.5)/36525.  # time since J2000 in jul .centuries

    ss = as2rad(-47e-6*tjc)

    _xp =   as2rad(xp)
    _yp =   as2rad(yp)

#    W     = np.matmul(np.matmul(rotm(-ss,3), rotm(xp,2)), rotm(yp,1))
    W     = np.einsum('ij,jk,kl->il', rotm(-ss,3), rotm(_xp,2), rotm(_yp,1))

    if libsofa is None:
        PN  =   np.eye(3, dtype=float)
        return PN, W

    X, Y, S =   xys2006a(tt)

# precession/nutation matrix:
    v     = -np.sqrt(X**2+Y**2)
    E     = np.arctan2((Y/v),(X/v))
    z     = np.sqrt(1.0-(X**2+Y**2))
    d     = np.arctan2(v,z)
#    PN    = np.matmul(np.matmul(np.matmul(rotm(-E,3),rotm(-d,2)),rotm(E,3)),rotm(S,3))
    PN    = np.einsum('ij,jk,kl,lm->im', rotm(-E,3),rotm(-d,2),rotm(E,3),rotm(S,3))

    return PN, W

def t2c(din):

#    return np.eye(3, dtype = np.float)
    mjd =   din.date + din.time
    tmu, dut1, xp, yp   =   get_eop(din, mjd)

#    tmu =   35.         # TAI-UTC  
#    dut1=   -0.219083   # ut1-utc
#    xp  =   as2rad(0.054190)
#    yp  =   as2rad(0.430600)

    tt  =   mjd + (32.184 + tmu)/86400.

    tjc =(tt-51544.5)/36525.  # time since J2000 in jul .centuries

    ss = as2rad(-47e-6*tjc)

# earth rotation angle

    ut   = mjd + dut1/86400.
    tu   = ut - 51544.5             # days since fundamental epoch
    frac = ut - np.floor(ut) + 0.5  # UT1 Julian day fraction
    fac  = 0.00273781191135448
            
    era = 2. * np.pi * (frac + 0.7790572732640 + fac * tu )
    era = era % (2 * np.pi)              # [rad]

    X, Y, S =   xys2006a(tt)

    W     = np.matmul(np.matmul(rotm(-ss,3), rotm(as2rad(xp),2)), rotm(as2rad(yp),1))
# rotation around pole axis
    R     = rotm(-era,3)
# precession/nutation matrix:
    v     = -np.sqrt(X**2+Y**2)
    E     = np.arctan2((Y/v),(X/v))
    z     = np.sqrt(1.0-(X**2+Y**2))
    d     = np.arctan2(v,z)
    PN    = np.matmul(np.matmul(np.matmul(rotm(-E,3),rotm(-d,2)),rotm(E,3)),rotm(S,3))
 
    return np.matmul(np.matmul(PN, R), W)

def norm(v):
    return np.sqrt(np.dot(v, v))

def valid_jd(jd):
    if jd < 2414864.50:
        return False
    if jd > 2471184.50:
        return False
    return True

   
# jd in tdb!
def get_crs_moon(jd):

    global spk

    assert valid_jd(jd)
    p   =   spk[3, 301].compute(jd) - spk[3, 399].compute(jd)
    return p*1E3

def get_crs_sun(jd):

    global spk

    assert valid_jd(jd)
# earth to earth bary 
# earth bary to solar sys bary 
# solar sys bary to sun
    p   =   - spk[3, 399].compute(jd) \
            - spk[0, 3].compute(jd) \
            + spk[0, 10].compute(jd)
    return p*1E3

def plot_crs(task):

    fig =   plt.figure()
    ax  =   fig.add_subplot(projection = '3d')
    for stn in task.stns:
        
#        if stn.cb != 'Earth':
#            print('Skip non Earth crs...')
#            continue
        x, y, z =   zip(*stn.p_crs.tolist())
        ax.plot(x, y, z)

# check radius
#        if stn.type == 'orbit':
#            p   =   stn.p_crs
#            r   =   np.sqrt(p[:, 0]**2+p[:, 1]**2+p[:, 2]**2)     
#            print(r)
        
    plt.show()

def plot_lcs(task):

    fig =   plt.figure()
    ax  =   fig.add_subplot(projection = '3d')
    for stn in task.stns:
        if stn.cb != 'Moon':
            print('util.plot_lcs(): skip none lunar station %s' % (stn.name))
            continue
        x, y, z =   zip(*stn.p_lcs.tolist())
        ax.plot(x, y, z)

    lim =   (-Rm, Rm)
#    ax.set_xlim(lim)
#    ax.set_ylim(lim)
#    ax.set_zlim(lim)
#
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()

    print('Exit after util.plot_lcs()')
    sys.exit(0)

def mas2rad(mas):
    return mas / 1E3 / 3600 / 180. * np.pi

def deg2rad(deg):
    return deg / 180. * np.pi

def h2ae(hl, hh, R):
    
    a   =   (hl + hh + 2 * R) * 0.5
    c   =   a - hl - R
    e   =   c / a
    return a, e

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

