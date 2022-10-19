#!/usr/bin/env python

import numpy as np
from scipy.integrate import solve_ivp
from datetime import datetime, timedelta
from astropy import units as un, constants as const
from .orbital import earth, KeplerianElements as KE, earth_mu, moon_mu
from . import util
from .base import Station

class DRO(Station):

    def __init__(self, name):
        super().__init__(name)
#        self.type   =   type(self).__name__
        self.type   =   'dro'
        self.cb     =   'Moon'

    def set_param(self, **kw):

        self.ob_id  =   kw['orbit_id']
        del kw['orbit_id']

        if not hasattr(self, 'ic_dro'):
            self.ic_dro =   np.loadtxt('%s/data/ic_dro.txt' % (util.path_home))
        self.ob  =   self.ic_dro[self.ob_id]
        self.kw   =   kw

    def gen_init(self):
        
        tt      =   self.utc2tt(self.task.t0)

        spk     =   self.task.spk
        p1, v1  =   spk[3, 399].compute_and_differentiate(tt)
        p1  *=  1E3
        v1  *=  (1E3/86400.)

        p2, v2  =   spk[3, 301].compute_and_differentiate(tt)
        p2  *=  1E3
        v2  *=  (1E3/86400.)
 
        r1      =   util.norm(p1)
        r2      =   util.norm(p2)
        
        r       =   r1 + r2
        f_mu    =   r1 / r

        v_kep   =   np.sqrt((earth_mu+moon_mu) / r)
        
        v2_norm =   util.norm(v2)

        t_unit  =   r / v_kep
        print('mu: %.6f, r: %.3fx10^4 km, v_kep: %.3f km/s, v2_norm: %.3f km/s, t_unit: %.3f days' % \
                (f_mu, r/1E7, v_kep/1E3, v2_norm/1E3, t_unit/86400.))

# Moon move direction
        e_v2    =   v2 / v2_norm
        e_p2    =   p2 / util.norm(p2)

#        p_m, v_m    =   util.get_crs_pv_moon(tt) 
#        return np.concatenate((p_m, v_m))

        x_nodim   =   self.ob[3]
        v_nodim   =   self.ob[4]

        x_init  =   e_p2 * r * x_nodim
        v_init  =   e_v2 * v_kep * v_nodim
        v_init  +=  v2 / (1-f_mu) * x_nodim

        return np.concatenate((x_init, v_init))
   
    def calc_p_rot(self, p_crs):
        
        ts  =   self.task.ts
        
        p_rot   =   []
        for i in range(len(ts)):

            t   =   ts[i]
            p   =   p_crs[i, :]
            
            t_utc       =   self.task.t0 + timedelta(seconds=t)
            tt          =   self.utc2tt(t_utc)
            p_m, v_m    =   util.get_crs_pv_moon(tt) 
            
            p_norm  =   util.norm(p_m)
            v_norm  =   util.norm(v_m)
        
            ex      =   p_m / p_norm
            _ey     =   v_m / v_norm
            _ez     =   np.cross(ex, _ey)
            ez      =   _ez / util.norm(_ez)
            ey      =   np.cross(ez, ex)

            e_rot   =   np.array([ex, ey, ez])
            
            p_rot.append(np.einsum('ij,j->i', e_rot, p-p_m))
#            p_rot.append(p-p_m)

        return np.array(p_rot)
            
    def utc2tt(self, t):
        return util.DT_JD + util.utc2mjd(t) + (32.184+self.task.tmu)/86400.

    def gen_init_rot(self):
        
        self.v_n =   0.73823E3 / 0.720544
        self.r_n =   9000E3 / 0.023413
        self.t_n =   10.639 * 60. / 0.102081
        self.a_n =   self.v_n / self.t_n
        self.m2     =   1.215059E-2
        self.m1     =   1 - self.m2
        self.p_e    =   np.array([-self.m2, 0., 0.])
        self.p_m    =   np.array([1.-self.m2, 0., 0.])
        
        print('gen_init_rot(): v0: %.3f km/s, r0: %.3fX10^4 km, t0: %.3f h, a0: %.3f m/s^2' % (self.v_n/1E3, self.r_n/1E7, self.t_n/3600., self.a_n))
        x_nodim   =   self.ob[3]
        v_nodim   =   self.ob[4]

#        x_nodim =   v_nodim =   1.0

        x_init  =   [x_nodim, 0., 0.]
        v_init  =   [0.0, v_nodim, 0.0]

        return np.concatenate((x_init, v_init)), self.ob[2]

    def calc_rot(self):

        print('##### calc_rot() is called. #####')
        
        def calc_grav(m, p0, p1):
            
            r   =   p1 - p0
            rabs    =   util.norm(r)
            rqu     =   r / rabs

            return -m / (rabs**2) * rqu

        def calc_centrifugal(pos, vel):

            omega   =   np.array([0.0, 0.0, 1.0])
            
            acc     =   - 2. * np.cross(omega, vel) \
                        - np.cross(omega, np.cross(omega, pos))
            
            return acc

        def f(t, y):
            
            pos =   y[0:3]
            vel =   y[3:6]

#            acc =   calc_grav(1.0, np.zeros(3), pos)
            acc_e   =   calc_grav(self.m1, self.p_e, pos)
            acc_m   =   calc_grav(self.m2, self.p_m, pos)
            acc_c   =   calc_centrifugal(pos, vel)

            acc     =   acc_e + acc_m + acc_c
#            print(acc)

            dydt    =   np.concatenate((vel, acc))
#            print(dydt)

            return dydt

        y0, P   =   self.gen_init_rot()
        
        dt_max  =   0.01    

        n   =   int(P / dt_max) + 1
        ts  =   np.arange(n) * dt_max

#        self.kw['t_eval']       =   ts
        self.kw['max_step'] =   dt_max
        t_span  =   [ts[0], ts[-1]]
#        self.kw['t_eval']    =   ts
#        sol     =   solve_ivp(f, t_span, y0, **self.kw)
        sol     =   solve_ivp(f, t_span, y0, t_eval=ts, max_step=dt_max)

        p_crs   =   sol.y[0:3, :].T
        self.p_rot  =   p_crs
        print(p_crs)
        return p_crs

# nt, 3
    def calc_crs(self):

        if hasattr(self, 'do_dro_rot'):
            return self.calc_rot()

# gravity on object 1:
        def calc_grav(mu0, p0, p1):
            
            r   =   p1 - p0
            rabs    =   util.norm(r)
            rqu     =   r / rabs

            return -mu0 / (rabs**2) * rqu

        spk =   self.task.spk 
        def f(t, y):
            
            pos =   y[0:3]
            vel =   y[3:6]

            t_utc   =   self.task.t0 + timedelta(seconds=t)
            tt      =   self.utc2tt(t_utc)

            p_m  =   spk[3, 301].compute(tt) * 1E3
            p_e  =   spk[3, 399].compute(tt) * 1E3

#            p_m_crs   =   p_m - p_e
#            r   =   util.norm(p_m_crs)
#            print('Earth-Moon dist: %.3fX10^4 km' % (r/1E7))
                
#            p_m     =   util.get_crs_moon(tt)
#            p_e     =   np.zeros(3, dtype=np.float)
            
            acc     =   calc_grav(earth_mu, p_e, pos) + \
                        calc_grav(moon_mu, p_m, pos)

#            acc     =   calc_grav(earth_mu, p_e, pos)

            dydt    =   np.concatenate((vel, acc))
#            print(t, dydt)

            return dydt

#        first_step  =   10.0 # in seconds
        y0  =   self.gen_init()
        
        ts  =   self.task.ts
        t_span  =   [ts[0], ts[-1]]
#        sol     =   solve_ivp(f, t_span, y0, method='RK45', t_eval=ts, first_step=dt)
        self.kw['t_eval']    =   ts
        sol     =   solve_ivp(f, t_span, y0, **self.kw)

        p_crs   =   []
        for i in range(len(self.task.ts)):
            t   =   self.task.ts[i]
            t_utc   =   self.task.t0 + timedelta(seconds=t)
            tt      =   self.utc2tt(t_utc)
            p_e  =   spk[3, 399].compute(tt) * 1E3
            p_crs.append(sol.y[0:3, i]-p_e)
 
        p_crs   =   np.array(p_crs)
#        p_crs   =   np.transpose(sol.y[0:3, :])         
        
        self.p_rot  =   self.calc_p_rot(p_crs)
#        self.p_rot  =   p_crs

        return p_crs
        
