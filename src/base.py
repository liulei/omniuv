#!/usr/bin/env python

import sys, os
import numpy as np
from datetime import datetime, timedelta
from astropy import units as un, constants as const, time as atime

from . import util

class Image(object):

    def __init__(self, src):
       
        self.ra0    =   src.ra
        self.dec0   =   src.dec
        self.ruvw0  =   src.ruvw
        self.lmn    =   []
 
class Source(object):

    def __init__(self):
        self.ra     =   0.0
        self.dec    =   0.0
        self.name   =   ''

class Task(object):
    
    def __init__(self):
        self.stns   =   {}
        self.srcs   =   []
        self.uvws    =   {}

        self.nsrc   =   0
        self.nstn   =   0

# two bits quantization:
        self.eta    =   0.88

        self.do_unif =   False
        self.do_rad  =   False
        self.do_beam_correction =   False

    def set_eops(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)

# Input:    UTC
# Output:   ET, TT, TDB
    def calc_jd(self, t):
        if not hasattr(self, 'tmu'):
            print('Task.calc_jd(): please call set_eops() first!')
            sys.exit(0)
#        assert t >= self.t0 # actual UTC time, not t in ts array!
        return util.DT_JD + (t - util.T0_MJD).total_seconds() / 86400. + (32.184 + self.tmu)/86400.

    def set_ts(self, t0, ts):
        self.t0     =   t0
        self.ts     =   ts
        self.jds    =   np.array([self.calc_jd(t0 + timedelta(seconds = t)) \
                                    for t in ts])

    def calc_gain(self):
        
        nt          =   len(self.ts)
        gain_amp    =   np.random.normal(loc = 1, \
                                        scale = self.gain_error, \
                                        size = nt)
        ids     =   np.where(gain_amp < 0.0)[0]
        gain_amp[ids]   =   0.0
#        gain_phase     =   np.random.uniform(0, 2.*np.pi, nt)
        gain_phase      =   np.zeros(nt)
        return gain_amp, gain_phase


    def set_stns(self, stns):

        self.nstn   =   len(stns)
        self.nbl    =   (self.nstn * (self.nstn-1)) // 2
        print('Stations: %d, baselines: %d' % \
                (self.nstn, self.nbl))
        self.stn_name2id =   {}
        for id in range(self.nstn):
            stn     =   stns[id]
            name    =   stn.name
            if name in self.stn_name2id.keys():
                print('stn name %s is already used!' % (name))
                sys.exit(0)
            self.stn_name2id[name]  =   id
            stn.set_task(self) 

            if  'Lunar' in type(stn).__name__ or \
                'Moon' in type(stn).__name__ or \
                'Moon' in stn.sep_min.keys():
                util.load_spk() # in util.py
                self.crs_moon   =   self.calc_crs_moon()

            if  'Sun' in type(stn).__name__ or \
                'Sun' in stn.sep_min.keys():
                self.crs_sun    =   self.calc_crs_sun()

            if 'LunarFixed' in type(stn).__name__:
                util.load_pck()

        self.bl2stn    =   []
        for i in range(self.nstn):
            for j in range(i+1, self.nstn):
                self.bl2stn.append((i, j))

        self.stns   =   stns

        if hasattr(self, 'gain_error'):
            for stn in self.stns:
                stn.gain_amp, stn.gain_phase = self.calc_gain()

    def set_srcs(self, srcs):

        self.nsrc   =   len(srcs)
        self.srcs   =   srcs

        id  =   0
        for src in self.srcs:
            src.ruvw    =   util.rqu2ruvw(util.crs2rqu(src.ra, src.dec))
            src.id  =   id
            id  +=  1
        
    def calc_uvw_per_bl(self, s1, s2):

        uvw =   []
        idt =   []

        for i in range(self.nsrc):

            idt_src =   np.intersect1d(s1.idt_avail[i], s2.idt_avail[i])
            uvw_src =   s1.uvw[i][idt_src] - s2.uvw[i][idt_src]
        
            uvw.append(uvw_src)
            idt.append(idt_src)
            
        return uvw, idt

# sun position 
    def calc_crs_sun(self):
        crs_sun    =   np.array([util.get_crs_sun(jd) for jd in self.jds])
        return crs_sun

# moon position 
    def calc_crs_moon(self):
        crs_moon    =   np.array([util.get_crs_moon(jd) for jd in self.jds])
        return crs_moon

    def calc_crs_stn(self):
        for stn in self.stns:
            stn.p_crs   =   stn.calc_crs()

# Task()
    def calc_uvw_stn(self):

        for stn in self.stns:
            stn.calc_uvw()

    def calc_uvw_bl(self):

        self.calc_uvw_stn()
          
        uvw =   []
        idt =   []
        for i in range(self.nstn):
            for j in range(i+1, self.nstn):
                uvw_bl, idt_bl  =   self.calc_uvw_per_bl(self.stns[i], self.stns[j])
                uvw.append(uvw_bl)
                idt.append(idt_bl)

# nbl, nsrc, idt
        self.idt_avail  =   idt
# bl, src, ts, 3(uvw)
        self.uvw        =   uvw
        return uvw

# uvw: bl, src, ts, 3(uvw)
    def flat_uvw_bl(self, uvw):
        
        uvw_flat =   []
        for i in range(self.nsrc):
            uvw_src =   []
            for j in range(self.nbl):
                uvw_src.append(uvw[j][i])
            uvw_flat.append(np.concatenate(uvw_src, axis = 0).copy())

# src, ts, 3
        return uvw_flat

    from ._sim import   gen_vis_direct, gen_vis_fft, \
                        gen_image_fft, gen_image_direct, gen_beam, \
                        plot_uv, plot_image, plot_beam, plot_src, \
                        vis_add_noise, gen_image_fft, gen_weight, \
                        uv2id, set_uv_image_param

    from ._fits import  to_fitsidi, gen_Primary, gen_UV, gen_SU, \
                        gen_FR, gen_AN, gen_AG, add_fits_keys
                        

class Station(object):

    def __init__(self, name):
        self.name   =   name

#        self.dut1   =   -2.21   # UT1-UTC
#        self.tmu    =   35.0    # TAI-UTC 

        self.uvw_updated    =   False

        self.sep_min    =   {}

    def set_task(self, task):
        self.task   =   task

    def set_orbit(self, a = 0.0, e = 0.0, i = 0.0, raan = 0.0, \
                        arg_pe = 0.0, M0 = 0.0, t_ref=None):

        if self.type != 'orbit':
            print('Error: station %s, set_orbit() can only be called in orbit type (this type: %s)' % (self.name, self.type))
            sys.exit(0)

        if (1.0 - e) * a < util.cbs[self.cb]['R']:
            print('Error: sation %s, (1-e)*a is smaller than the radius of %s!' % (self.name))
            sys.exit(0)

        self.orbit.a    =   a
        self.orbit.e    =   e
        self.orbit.i    =   i
        self.orbit.raan     =   raan
        self.orbit.arg_pe   =   arg_pe
        self.orbit.M0       =   M0

        if t_ref != None:
            self.orbit.ref_epoch    =   atime.Time(t_ref, scale='utc')

        print('Station %s:' % (self.name))
        print(self.orbit)
        print('')

        self.t_ref  =   t_ref

        self.uvw_updated    =   False

    def set_SEFD(self, SEFD):
        self.SEFD   =   SEFD
        
    def calc_crs(self):
        print('Error: station %s, class %s, calc_crs() method has not been implemented!' % (self.name, type(self).__name__))
        sys.exit(0)
        pass

    def calc_uvw(self):
        
        if self.uvw_updated:
            return self.uvw

# nt, 3 (xyz)
        self.p_crs  =   self.calc_crs()
        uvw =   []
# ruvw: 3, 3 (uvw unit vec)
        for src in self.task.srcs:
            _uvw    =   np.einsum('ik,jk->ij', self.p_crs, src.ruvw)
            uvw.append(_uvw)

        self.idt_avail  =   self.calc_idt_avail()
        self.uvw        =   np.array(uvw)

        self.uvw_updated    =   True

# nsrc, nt, 3(uvw)
        return self.uvw

    from ._el_sep import set_sep_min_deg, \
                            _calc_sep, \
                            calc_sep_moon, \
                            calc_sep_earth, \
                            calc_idt_sep_src, \
                            set_el_min_deg, \
                            _calc_el, \
                            calc_idt_avail
