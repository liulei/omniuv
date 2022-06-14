#!/usr/bin/env python

import numpy as np
from datetime import datetime, timedelta
from astropy import units as un, constants as const
from .orbital import moon, KeplerianElements as KE
from . import util
from .base import Station

class LunarOrbit(Station):

    def __init__(self, name):
        super().__init__(name)
#        self.type   =   type(self).__name__
        self.type   =   'orbit'
        self.cb     =   'Moon'
        self.orbit  =   KE(body = moon)

    def _ob2lcs(self, t):
        self.orbit.t    =   t
        return self.orbit.r
        
# nt, 3
    def calc_crs(self):

        self.p_lcs  =   np.array([self._ob2lcs(t) for t in self.task.ts])
        return self.p_lcs + self.task.crs_moon

