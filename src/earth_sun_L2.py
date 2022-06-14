#!/usr/bin/env python

import numpy as np
from datetime import datetime, timedelta
from astropy import units as un, constants as const
from .orbital import moon, KeplerianElements as KE
from . import util
from .base import Station

class EarthSunL2(Station):

    def __init__(self, name):
        super().__init__(name)
#        self.type   =   type(self).__name__
        self.type   =   'orbit'
        self.cb     =   ''

# nt, 3
    def calc_crs(self):

        p_crs   =   -self.task.crs_sun * util.r_ES_L2
        return p_crs

