
import sys, os
import numpy as np
from astropy.constants import c as c_light
from astropy.io import fits
from datetime import datetime, timedelta
from . import util

NCHAN   =   1

def arrayGMST(mjd):

    M_PI    =   np.pi
    mjd2000 = 51545.0
    convhs  = 7.2722052166430399e-5
    gmstc   = np.zeros(4, dtype=float)
    gmstc[0] = 24110.548410;
    gmstc[1] = 8640184.8128660;
    gmstc[2] = 0.0931040;
    gmstc[3] = -6.2e-6;

    daysj = mjd - mjd2000 + 0.5;

    cent = daysj / 36525;

    gstmn = (gmstc[0] + gmstc[1]*cent + gmstc[2]*cent*cent + gmstc[3]*cent*cent*cent)*convhs;

    igstmn = gstmn // (2.0*M_PI);
    gstmn = gstmn - igstmn * (2.0*M_PI);
    if gstmn < 0.0:
        gstmn += (2.0*M_PI);
  
    return gstmn / (2.0*M_PI)

def add_fits_keys(self, h):
    
    if not hasattr(self, 'obscode'):
        obscode =   'RL007'
    else:
        obscode =   self.obscode

    h['OBSCODE']    =   obscode
    h['RDATE']      =   self.t0.strftime('%Y-%m-%d')
    h['NO_STKD']    =   1
    h['STK_1']      =   -1
    h['NO_BAND']    =   len(self.freqs)
    h['NO_CHAN']    =   NCHAN
    h['REF_FREQ']   =   self.freqs[0]
    h['CHAN_BW']    =   self.bandwidth / NCHAN
    h['REF_PIXL']   =   1.0

def gen_Primary(self):

    hdu =   fits.PrimaryHDU()

    h   =   hdu.header
    
    h['SIMPLE']     =   True
    h['BITPIX']     =   8
    h['NAXIS']      =   0
    h['EXTEND']     =   True
    h['BLOCKED']    =   True
    h['OBJECT']     =   'BINARYTB'
    h['TELESCOP']  =   'VLBA'
    h['OBSERVER']   =   'RL007'
    h['ORIGIN']     =   'VLBA Correlator'
    h['CORRELAT']   =   'DIFX'
    h['DATE-OBS']   =   self.t0.strftime('%Y-%m-%d')
    h['DATE-MAP']   =   datetime.now().strftime('%Y-%m-%d')
    h['GROUPS']     =   True
    h['GCOUNT']     =   0
    h['PCOUNT']     =   0

    return hdu

def gen_AG(self):

    nfreq   =   self.freqs.shape[0]
    nstn    =   self.nstn
    cols    =   []

    cols    =   []
    ANNAME  =   []
    STABXYZ =   []
    DERXYZ  =   []
    ORBPARM =   []
    NOSTA   =   []
    MNTSTA  =   []
    STAXOF  =   []
    DIAMETER    =   []
    for i in range(nstn):
        stn =   self.stns[i]
        ANNAME.append(stn.name)
        if stn.type == 'fixed' and stn.cb == 'Earth':
            xyz =   stn.p_trs.astype(np.float64)
        else:
            xyz =   np.ones(3, dtype=np.float64)
        STABXYZ.append(xyz)

        DERXYZ.append(np.zeros(3, dtype=np.float32))

        NOSTA.append(i+1)

        MNTSTA.append(0)

        STAXOF.append(np.zeros(3, dtype=np.float32))

        DIAMETER.append(0.0)

    cols.append(fits.Column(name='ANNAME', format='8A', unit=0, array=ANNAME))
    cols.append(fits.Column(name='STABXYZ', format='3D', unit='METERS', array=STABXYZ))
    cols.append(fits.Column(name='DERXYZ', format='3E', unit='M/SEC', array=DERXYZ))
    cols.append(fits.Column(name='ORBPARM', format='0D', unit=0, array=ORBPARM))
    cols.append(fits.Column(name='NOSTA', format='1J', unit=0, array=NOSTA))
 
    cols.append(fits.Column(name='MNTSTA', format='1J', unit=0, array=MNTSTA))

    cols.append(fits.Column(name='STAXOF', format='3E', unit='METERS', array=STAXOF))

    cols.append(fits.Column(name='DIAMETER', format='1E', unit='METERS', array=DIAMETER))

    h   =   fits.Header()
    h['EXTNAME']    =   'ARRAY_GEOMETRY'
    h['EXTVER']     =   1
    hdu =   fits.BinTableHDU.from_columns(cols, header=h)
    h   =   hdu.header

    h['ARRAYX']     =   0.0
    h['ARRAYY']     =   0.0
    h['ARRAYZ']     =   0.0
    h['ARRNAM']     =   'VLBA'
    h['NUMORB']     =   0
    h['FREQ']       =   self.freqs[0]
    h['FRAME']      =   'GEOCENTRIC'
    h['TIMSYS']     =   'UTC'
    h['TIMESYS']    =   'UTC'
    h['GSTIA0']     =   arrayGMST(util.utc2mjd(self.t0)) * 360.
    h['DEGPDY']     =   360.9856449733

    h['POLARX']     =   self.xp
    h['POLARY']     =   self.yp
    h['UT1UTC']     =   self.dut1
    h['IATUTC']     =   self.tmu

    self.add_fits_keys(h)

    h['TABREV']     =   1 
    
    return hdu


def gen_SU(self):

    nfreq   =   self.freqs.shape[0]
    src     =   self.srcs[0]

    ra      =   util.rad2deg(src.ra)
    dec     =   util.rad2deg(src.dec)

    FLUX    =   [np.array(np.zeros(nfreq, dtype='>f4'))]

    src.name    =   '3C288'
    cols    =   []
    cols.append(fits.Column(name='SOURCE_ID', format='1J', array=[1]))
    cols.append(fits.Column(name='SOURCE', format='16A', array=[self.srcs[0].name]))
    cols.append(fits.Column(name='QUAL', format='1J', array=[0]))
    cols.append(fits.Column(name='CALCODE', format='4A', array=['']))
    cols.append(fits.Column(name='FREQID', format='1J', array=[1]))
    cols.append(fits.Column(name='IFLUX', format='%dE'%(nfreq), unit='JY', array=FLUX))
    cols.append(fits.Column(name='QFLUX', format='%dE'%(nfreq), unit='JY', array=FLUX))
    cols.append(fits.Column(name='UFLUX', format='%dE'%(nfreq), unit='JY', array=FLUX))
    cols.append(fits.Column(name='VFLUX', format='%dE'%(nfreq), unit='JY', array=FLUX))
    cols.append(fits.Column(name='ALPHA', format='%dE'%(nfreq), array=FLUX))
    cols.append(fits.Column(name='FREQOFF', format='%dE'%(nfreq), unit='HZ', array=FLUX))
    cols.append(fits.Column(name='RAEPO', format='1D', unit='DEGREES', array=[ra]))
    cols.append(fits.Column(name='DECEPO', format='1D', unit='DEGREES', array=[dec]))
    cols.append(fits.Column(name='EQUINOX', format='8A', array=['J2000']))
    cols.append(fits.Column(name='RAAPP', format='1D', unit='DEGREES', array=[ra]))
    cols.append(fits.Column(name='DECAPP', format='1D', unit='DEGREES', array=[dec]))
    cols.append(fits.Column(name='SYSVEL', format='%dD'%(nfreq), unit='M/SEC', array=[[0.0]*nfreq]))
    cols.append(fits.Column(name='VELTYP', format='8A', array=['GEOCENTR']))
    cols.append(fits.Column(name='VELDEF', format='8A', array=['OPTICAL']))
    cols.append(fits.Column(name='RESTFREQ', format='%dD'%(nfreq), unit='HZ', array=FLUX))
    cols.append(fits.Column(name='PMRA', format='1D', unit='DEG/DAY', array=[0.0]))
    cols.append(fits.Column(name='PMDEC', format='1D', unit='DEG/DAY', array=[0.0]))
    cols.append(fits.Column(name='PARALLAX', format='1E', unit='DEG/DAY', array=[0.0]))
    cols.append(fits.Column(name='EPOCH', format='1D', unit='YEARS', array=[2000.0]))
    cols.append(fits.Column(name='RAOBS', format='1D', unit='DEGREES', array=[ra]))
    cols.append(fits.Column(name='DECOBS', format='1D', unit='DEGREES', array=[dec]))
   
    h   =   fits.Header()
    h['EXTNAME']    =   'SOURCE'
    h['EXTVER']     =   1
    hdu =   fits.BinTableHDU.from_columns(cols, header=h)
    h   =   hdu.header
    self.add_fits_keys(h)
    h['TABREV']     =   1 

    assert h['NAXIS2'] == 1
    return hdu
 
def gen_FR(self):

    nfreq   =   len(self.freqs)
    ref_freq    =   self.freqs[0]

    cols    =   []
    cols.append(fits.Column(name='FREQID', format='1J', unit=0, array=[1]))

    BANDFREQ   =   []
    for i in range(nfreq):
        BANDFREQ.append(self.freqs[i] - ref_freq)
    cols.append(fits.Column(name='BANDFREQ', format='%dD'%(nfreq), unit='HZ', array=[BANDFREQ]))

    cols.append(fits.Column(name='CH_WIDTH', format='%dE'%(nfreq), unit='HZ', array=[[self.bandwidth/NCHAN]*nfreq]))

    cols.append(fits.Column(name='TOTAL_BANDWIDTH', format='%dE'%(nfreq), unit='HZ', array=[[self.bandwidth]*nfreq]))

    cols.append(fits.Column(name='SIDEBAND', format='%dJ'%(nfreq), unit=0, array=[[1]*nfreq]))

    cols.append(fits.Column(name='BB_CHAN', format='%dJ'%(nfreq), unit=0, array=[[0]*nfreq]))

    h   =   fits.Header()
    h['EXTNAME']    =   'FREQUENCY'
    h['EXTVER']     =   1
    hdu =   fits.BinTableHDU.from_columns(cols, header=h)
    h   =   hdu.header
    self.add_fits_keys(h)
    h['TABREV']     =   2

    assert h['NAXIS2'] == 1
    return hdu

def gen_AN(self):

    cols        =   []

    TIME        =   []
    TIMEINT     =   []
    ANNAME      =   []
    ANTENNA_NO  =   []

    nstn    =   self.nstn
    nfreq   =   self.freqs.shape[0]

    mjd     =   util.utc2mjd(self.t0)
    start   =   mjd - int(mjd)
    stop    =   start + (self.ts[-1]-self.ts[0])/86400.
    time    =   (start + stop) * 0.5
    timeInt =   stop - start

    cols.append(fits.Column(name='TIME', format='1D', unit='DAYS', array=[time]*nstn))
    cols.append(fits.Column(name='TIME_INTERVAL', format='1E', unit='DAYS', array=[timeInt]*nstn))

    names   =   []
    for i in range(nstn):
        names.append(self.stns[i].name)
    cols.append(fits.Column(name='ANNAME', format='8A', array=names))

    NO  =   np.arange(nstn, dtype=int) + 1
    cols.append(fits.Column(name='ANTENNA_NO', format='1J', array=NO))

    cols.append(fits.Column(name='ARRAY', format='1J', array=[1]*nstn))

    cols.append(fits.Column(name='FREQID', format='1J', array=[1]*nstn))

    cols.append(fits.Column(name='NO_LEVELS', format='1J', array=[4]*nstn))

    cols.append(fits.Column(name='POLTYA', format='1A', array=['R']*nstn))
    cols.append(fits.Column(name='POLAA', format='%dE'%(nfreq), unit='DEGREES', array=[[0.0]*nfreq]*nstn))
    cols.append(fits.Column(name='POLCALA', format='0E', array=[]))

    cols.append(fits.Column(name='POLTYB', format='1A', array=['L']*nstn))
    cols.append(fits.Column(name='POLAB', format='%dE'%(nfreq), unit='DEGREES', array=[[0.0]*nfreq]*nstn))
    cols.append(fits.Column(name='POLCALB', format='0E', array=[]))

    h   =   fits.Header()
    h['EXTNAME']    =   'ANTENNA'
    h['EXTVER']     =   1 
    hdu =   fits.BinTableHDU.from_columns(cols, header=h)
    h   =   hdu.header
    self.add_fits_keys(h)
    h['TABREV']     =   1
    h['NOPCAL']     =   0
    h['POLTYPE']    =   'APPROX'
 
    assert h['NAXIS2'] == nstn
    return hdu

def gen_UV(self, bls):

    cols    =   []

    UU      =   []
    VV      =   []
    WW      =   []
    DATE    =   []
    TIME    =   []
    BASELINE=   []
    FILTER  =   []
    SOURCE  =   []
    FREQID  =   []
    INTTIM  =   []
    WEIGHT  =   []
    GATEID  =   []
    FLUX    =   []

    ref_freq    =   self.freqs[0]
    nfreq   =   len(self.freqs)
    npolar  =   1
    nchan   =   NCHAN
    ndata   =   2 * nfreq * npolar * nchan
    nvis    =   0
    for bl_id in range(self.nbl):
            
        bl  =   bls[bl_id]
        if bl == {}:
            continue
        vis     =   bl['vis']
        nvis_bl =   vis.shape[0]
        nvis    +=  nvis_bl

        s0, s1  =   self.bl2stn[bl_id]
        bl_no   =   (s0+1) * 256 + s1+1
        
        for i in range(nvis_bl):
           
            uvw =   bl['uvw_m'][i].copy() / c_light.value
            UU.append(uvw[0])
            VV.append(uvw[1])
            WW.append(uvw[2])

            mjd =   util.utc2mjd(self.t0+timedelta(seconds=bl['t'][i]))
            imjd    =   int(mjd)
            DATE.append(util.DT_JD+imjd)
            TIME.append(mjd - imjd)
            BASELINE.append(bl_no)
            FILTER.append(0)
            SOURCE.append(1)
            FREQID.append(1)
            INTTIM.append(self.t_ap)
            WEIGHT.append([1.0] * (nfreq*npolar))

            vis =   bl['vis'][i].copy()
# nfreq, 2 (re, im), f32(E)
            vis =   vis.astype(np.complex64).view(np.float32).flatten()
            
            nv  =   len(vis) // 2
            if nv < nfreq * nchan:
                vis =   vis.tolist()
                vis =   np.array([vis]*(nfreq*nchan//nv))

            FLUX.append(vis)

    cols    =   []
    cols.append(fits.Column(name='UU---SIN', format='1E', unit='SECONDS', array=UU))
    cols.append(fits.Column(name='VV---SIN', format='1E', array=VV, unit='SECONDS'))
    cols.append(fits.Column(name='WW---SIN', format='1E', array=WW, unit='SECONDS'))
    cols.append(fits.Column(name='DATE', format='1D', array=DATE, unit='DAYS'))
    cols.append(fits.Column(name='TIME', format='1D', array=TIME, unit='DAYS'))
    cols.append(fits.Column(name='BASELINE', format='1J', array=BASELINE))
    cols.append(fits.Column(name='FILTER', format='1J', array=FILTER))
    cols.append(fits.Column(name='SOURCE', format='1J', array=SOURCE))
    cols.append(fits.Column(name='FREQID', format='1J', array=FREQID))
    cols.append(fits.Column(name='INTTIM', format='1E', array=INTTIM, unit='SECONDS'))
    cols.append(fits.Column(name='WEIGHT', format='%dE'%(nfreq), array=WEIGHT))
    cols.append(fits.Column(name='GATEID', format='0J', array=GATEID))
    cols.append(fits.Column(name='FLUX', format='%dE'%(ndata), array=FLUX, unit='UNCALIB'))
           
    h   =   fits.Header()
    h['EXTNAME']    =   'UV_DATA'
    h['EXTVER']     =   1
    hdu =   fits.BinTableHDU.from_columns(cols, header=h)
    h   =   hdu.header
    h['NMATRIX']    =   1

    h['DATE-OBS']   =   self.t0.strftime('%Y-%m-%d')
    h['EQUINOX']    =   'J2000'
    h['WEIGHTYP']   =   'CORRELAT'
    h['TELESCOP']   =   'VLBA'
    h['OBSERVER']   =   'LL'
    
# NMATRIX to TABREV are duplicated
    self.add_fits_keys(h)

    h['TABREV']     =   2

    h['VIS_SCAL']   =   1.0
    h['SORT']       =   'T*'
    h['MAXIS']      =   6

    h['MAXIS1']     =   2 # ncomplex
    h['CTYPE1']     =   'COMPLEX'
    h['CDELT1']     =   1.0
    h['CRPIX1']     =   1.0
    h['CRVAL1']     =   1.0
            
    h['MAXIS2']     =   npolar
    h['CTYPE2']     =   'STOKES'
    h['CDELT2']     =   -1.0
    h['CRPIX2']     =   1.0
    h['CRVAL2']     =   -1.0
            
    h['MAXIS3']     =   nchan
    h['CTYPE3']     =   'FREQ'
    h['CDELT3']     =   self.bandwidth
    h['CRPIX3']     =   1.0
    h['CRVAL3']     =   ref_freq
             
    h['MAXIS4']     =   nfreq
    h['CTYPE4']     =   'BAND'
    h['CDELT4']     =   1.0
    h['CRPIX4']     =   1.0
    h['CRVAL4']     =   1.0
 
    h['MAXIS5']     =   1
    h['CTYPE5']     =   'RA'
    h['CDELT5']     =   0.0
    h['CRPIX5']     =   1.0
    h['CRVAL5']     =   0.0

    h['MAXIS6']     =   1
    h['CTYPE6']     =   'DEC'
    h['CDELT6']     =   0.0
    h['CRPIX6']     =   1.0
    h['CRVAL6']     =   0.0
 
    h['TMATX11']    =   True

    return hdu

def to_fitsidi(self, name, bls):

    if name.split('.')[-1] != 'FITS':
        name += '.FITS'

    hdu_Primary =   self.gen_Primary()
    hdu_AG      =   self.gen_AG()
    hdu_SU      =   self.gen_SU()
    hdu_AN      =   self.gen_AN()
    hdu_FR      =   self.gen_FR()
    hdu_UV      =   self.gen_UV(bls)
    hdus        =   fits.HDUList([hdu_Primary, hdu_AG, hdu_SU, \
                    hdu_AN, hdu_FR, hdu_UV])
    hdus.writeto(name, overwrite=True)
    
