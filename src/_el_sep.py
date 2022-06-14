import numpy as np
from . import util

def set_sep_min_deg(self, **kw):
    
    for cb in kw.keys():
        if cb not in util.cbs:
            print('Warning: station %s, %s is not in central celestial body list, skip sep calculation!' % (self.name, cb))
            continue
        
        self.sep_min[cb]    =   kw[cb] / 180. * np.pi
        
def _calc_sep(self, dp, ruvw, R0):
    r       =   util.norm(dp)
    angle   =   np.arcsin(R0 / r)
    zenith  =   np.arccos(np.dot(dp / r, ruvw[2])) # zenith of the source
    return np.pi - angle - zenith

def calc_sep_moon(self, ruvw):
    dps =   self.p_crs - self.task.crs_moon
    R   =   util.cbs['Moon']['R']
    sep =   np.array([self._calc_sep(dp, ruvw, R) for dp in dps])
    return sep

def calc_sep_earth(self, ruvw):
    dps =   self.p_crs
    R   =   util.cbs['Earth']['R']
    sep =   np.array([self._calc_sep(dp, ruvw, R) for dp in dps])
    return sep

def calc_sep_sun(self, ruvw):
    dps =   self.p_crs - self.task.crs_sun
    R   =   util.cbs['Sun']['R']
    sep =   np.array([self._calc_sep(dp, ruvw, R) for dp in dps])
    return sep

def calc_idt_sep_src(self, src):

# if no cb is set for sep calc, all times are available
    if len(self.sep_min.keys()) == 0:
        nts =   len(self.task.ts)
        return np.arange(nts), np.zeros(nts) + np.pi/2

    sep_src =   {}

    idts    =   []
    for cb in self.sep_min.keys():

        if self.type == 'fixed' and self.cb == cb:
            print('Warning: station %s is fixed with %s, skip sep calculation for src %s!' % \
                    (self.name, self.cb, src.name))
            continue

        if cb == 'Moon':
            sep =   self.calc_sep_moon(src.ruvw)
            idt =   np.where(sep > self.sep_min[cb])[0]
            idts.append(idt)

        if cb == 'Earth':
            sep =   self.calc_sep_earth(src.ruvw)
            idt =   np.where(sep > self.sep_min[cb])[0]
            idts.append(idt)

        if cb == 'Sun':
            sep =   self.calc_sep_sun(src.ruvw)
            idt =   np.where(sep > self.sep_min[cb])[0]
            idts.append(idt)

        sep_src[cb]   =   sep

    return util.intersect1d_many(idts), sep_src

def set_el_min_deg(self, deg):
    self.el_min =   deg / 180. * np.pi

# celestial system
def _calc_el(self, dp, ruvw):
    dp_norm =   dp / util.norm(dp)
    return np.pi / 2 - np.arccos(np.dot(dp_norm, ruvw[2]))

def calc_idt_avail(self):
    
    el  =   []
    sep =   []
    idt =   []
    for src in self.task.srcs:
# el part
        idt_el_src  =   np.arange(len(self.task.ts))
        if self.type == 'fixed':

            if self.cb  ==  'Earth':
                dps =   self.p_crs

            if self.cb  ==  'Moon':
                dps     =   self.p_crs - self.task.crs_moon

            el_src  =   np.array([self._calc_el(dp, src.ruvw) \
                            for dp in dps])
            idt_el_src  =   np.where(el_src > self.el_min)[0]
            el.append(el_src)

# sep part 
        idt_sep_src, sep_src    =   self.calc_idt_sep_src(src)
        sep.append(sep_src)

# combine el and sep result
        idt_src =   np.intersect1d(idt_el_src, idt_sep_src)

# append this src
        idt.append(idt_src)

    self.el     =   el
    self.sep    =   sep
    return idt
 
