#!/usr/bin/env python

import numpy as np
from datetime import datetime, timedelta
from astropy import units as un, constants as const
from . import util
from .base import Station

# ground
class EarthFixed(Station):

    def __init__(self, name):
        super().__init__(name)
#        self.type   =   type(self).__name__
        self.type   =   'fixed'
        self.cb     =   'Earth'

    def set_trs(self, p_trs):
        self.p_trs  =   p_trs
        self.uvw_updated    =   False
        
    def _trs2crs(self, t):
        utc     =   self.task.t0 + timedelta(seconds = t)
        mjd     =   util.utc2mjd(utc)
        R       =   util.calc_t2c_R(self.task.dut1, mjd)
        task    =   self.task
#        dut1, PN, W =   util.calc_t2c_dut1_PN_W(mjd, task.tmu, task.dut1, task.xp, task.yp)
        PN, W =   util.calc_t2c_PN_W(mjd, task.tmu, task.dut1, task.xp, task.yp)
#        m   =   np.matmul(np.matmul(PN, R), W)
        m   =   np.einsum('ij,jk,kl->il', PN, R, W)
        return m

# Earth Fixed
    def calc_crs(self):
# nt, 3, 3
        m_R     =   np.array([self._trs2crs(t) for t in self.task.ts])
# nt, 3
        p_crs   =   np.einsum('ijk,k->ij', m_R, self.p_trs)
        return p_crs    

