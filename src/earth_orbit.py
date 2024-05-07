#!/usr/bin/env python

import numpy as np
from datetime import datetime, timedelta
from astropy import units as un, constants as const
from .orbital import earth, KeplerianElements as KE
from . import util
from .base import Station

class EarthOrbit(Station):

    def __init__(self, name):
        super().__init__(name)
#        self.type   =   type(self).__name__
        self.type   =   'orbit'
        self.cb     =   'Earth'
        self.orbit  =   KE(body = earth)

    def _ob2crs(self, t):
        self.orbit.t    =   t
        return self.orbit.r
        
# nt, 3
    def calc_crs(self):

        if self.t_ref == None:
            print('EarthOrbit.calc_crs(%s): t_ref is not set, use task.t0' % (self.name))
            t_ref   =   self.task.t0
        else:
            t_ref   =   self.t_ref

        t_offset    =   (self.task.t0 - t_ref).total_seconds()

        return np.array([self._ob2crs(t+t_offset) for t in self.task.ts])

