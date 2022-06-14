#!/usr/bin/env python

import sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import rc, pyplot as plt
from matplotlib.patches import Ellipse
from astropy.constants import c as c_light
from numpy.fft import fft2, ifft2, fftshift
from . import util
from .base import Image
from .backend import direct_image_np, direct_image_cp


#cellsize    =   2.5 # mas
#cellsize    =   0.025 # mas
#cellsize    =   0.005 # mas, 8.4 GHz, Moon-Earth L2, 59.6E4 km
#nc      =   256

NA  =   0.0

do_unif =   True
do_rad  =   False

def set_uv_image_param(self):

    cs_mas  =   self.cellsize
    nc      =   self.nc

    self.urange  =   1. / util.mas2rad(cs_mas)
    self.umax    =   self.urange * 0.5  
    self.vmax    =   self.umax
    self.umin    =   -self.umax
    self.vmin    =   -self.vmax
    self.du      =   self.urange / nc
    self.dv      =   self.du

def uv2id(self, uv):

    if not hasattr(self, 'urange'):
        self.set_uv_image_param()

    uf  =   (uv[0] - self.umin) / self.du
    vf  =   (uv[1] - self.vmin) / self.dv
    iu  =   int(uf + 0.5)
    iv  =   int(vf + 0.5)
    if iu < 0 or iv < 0:
        return -1, -1
    if iu >= self.nc or iv >= self.nc:
        return -1, -1
    return iv, iu

def gen_vis_fft(self, src):

    nc          =   self.nc
    cellsize    =   self.cellsize

    src_id  =   src.id

    arr     =   np.zeros((nc, nc), dtype=float)
    ds      =   util.mas2rad(cellsize)
    hs      =   ds * nc * 0.5

    ra      =   src.ra
    dec     =   src.dec
    dfac    =   np.cos(dec)

    img =   src.img
    for i in range(img.npixel):

        dra =   (img.ras[i] - ra) * dfac
        ddec=   (img.decs[i] - dec)
         
        ix  =   int((dra + hs) / ds + 0.5)
        iy  =   int((ddec+ hs) / ds + 0.5)

        arr[iy, ix] =   img.fluxes[i]

    plt.clf()
    plt.imshow(arr, origin='lower')
    plt.savefig('dump_gen_vis_fft.png')

#    arr =   arr[:, ::-1]
    vis_uv  =   fftshift(fft2(fftshift(arr)))

    urange  =   1. / ds
    umin    =   -urange * 0.5  
    vmin    =   umin
    du      =   urange / nc
    dv      =   du

# nearest neighbor
    def get_vis_nnb(uv):
        
        iv, iu  =   self.uv2id(uv)
        if iv < 0:
            return NA

        return vis_uv[iv, iu]

    bls =   []
    for bl_id in range(self.nbl):

        bl  =   {}

        uvw_bl  =   self.uvw[bl_id][src_id]
        idt_bl  =   self.idt_avail[bl_id][src_id]
        if len(uvw_bl) == 0:
            print('No vis in bl %d' % (bl_id))
            bls.append(bl) 
            continue
#        else:
#            print('Num of vis in bl %d: %d' % \
#                (bl_id, len(uvw_bl)))

# t, uvw
        bl['uvw_m']     =   uvw_bl
# t
        bl['t']         =   self.ts[idt_bl]
# t, freq, uvw
        bl['uvw_wav']   =   bl['uvw_m'][:, np.newaxis, :] / \
                c_light.value * self.freqs[np.newaxis, :, np.newaxis]

        uvs     =   bl['uvw_wav'][:, :, :2]
        
        vis_t   =   []
        for idt in range(uvs.shape[0]):

            vis_f   =   []
            for idf in range(uvs.shape[1]):
                vis =   get_vis_nnb(uvs[idt, idf])
                vis_f.append(vis)

            vis_t.append(vis_f)

# vis: t, freq
#        bl['vis']   =   np.array(vis_t) * nc * nc
        bl['vis']   =   np.array(vis_t)
#        bl['vis']       =   np.einsum('h,hij -> ij', \
#                            src.img.fluxes, fringe)
        bls.append(bl)
        
# bl, t, freq
    return bls

def gen_vis_direct(self, src):

    src_id  =   src.id

    bls =   []
    for bl_id in range(self.nbl):

        bl  =   {}

        uvw_bl  =   self.uvw[bl_id][src_id]
        idt_bl  =   self.idt_avail[bl_id][src_id]
        if len(uvw_bl) == 0:
            print('No vis in bl %d' % (bl_id))
            bls.append(bl) 
            continue
        else:
            print('Num of vis in bl %d: %d' % \
                (bl_id, len(uvw_bl)))
# t, uvw
        bl['uvw_m']     =   uvw_bl
# t
        bl['t']         =   self.ts[idt_bl]
# t, freq, uvw
        bl['uvw_wav']   =   bl['uvw_m'][:, np.newaxis, :] / \
                c_light.value * self.freqs[np.newaxis, :, np.newaxis]

        _lmn001         =   np.array([0, 0, 1])[np.newaxis, :]
# pixel, lmn
        lmn1            =   src.img.lmn - _lmn001
# pixel, t, freq
        lmn_uvw         =   np.einsum('hk,ijk -> hij', \
                            lmn1, bl['uvw_wav'])
        fringe          =   np.exp(-2j * np.pi * lmn_uvw)
# fluxes: pixel
# vis: t, freq
        bl['vis']       =   np.einsum('h,hij -> ij', \
                            src.img.fluxes, fringe)
        ids =   np.where(bl['vis'] == NA)[0]
        bls.append(bl)
        
# bl, t, freq
    return bls

def vis_add_noise(self, src, bls):

    src_id  =   src.id

    for bl_id in range(self.nbl):

        idt_bl  =   self.idt_avail[bl_id][src_id]
        if len(idt_bl) == 0:
            continue

        bl      =   bls[bl_id] 
        i0, i1  =   self.bl2stn[bl_id]
        s0      =   self.stns[i0]
        s1      =   self.stns[i1]
        
        n_std   =   np.sqrt(s0.SEFD * s1.SEFD / \
                    (2. * self.bandwidth * self.t_ap)) / self.eta

        size    =   bl['vis'].shape
        eps_amp =   np.abs(np.random.normal(loc=0.0, scale=n_std, \
                    size = size))
        eps_ph  =   np.random.uniform(0, 2.*np.pi, size)
        
        gain    =   (np.sqrt(s0.gain_amp * s1.gain_amp) * np.exp(1j * \
                    (s0.gain_phase - s1.gain_phase)))
        gain    =   gain[idt_bl][:, np.newaxis]
        gain    =   np.repeat(gain, len(self.freqs), axis = 1)
        
#        print('gain shape: ', gain.shape)
#        print('vis shape0: ', bl['vis'].shape)
        bl['vis']   =   gain * (bl['vis'] + eps_amp * np.exp(1j * eps_ph))
#        print('vis shape1: ', bl['vis'].shape)
 
    return bls

def gen_image_direct(self, bls):

    if not hasattr(self, 'umax'):
        self.set_uv_image_param()

    nc          =   self.nc
    cellsize    =   self.cellsize
    
#    bls =   np.load('vis.npy', allow_pickle=True)
    uvw  =   []
    vis =   []
    for bl in bls:
        if bl == {}:
            continue
        vis_bl  =   bl['vis'].flatten()
        uvw_bl  =   bl['uvw_wav'].reshape((-1, 3))
        uvw.append( uvw_bl)
#        uvw.append(-uvw_bl)
        vis.append(vis_bl)
#        vis.append(np.conj(vis_bl))

    urange  =   1. / util.mas2rad(cellsize)
    umax    =   urange * 0.5  
    vmax    =   umax

    uvw     =   np.concatenate(uvw, axis = 0)
    vis     =   np.concatenate(vis, axis = 0)

    uabs    =   np.abs(uvw[:, 0])
    vabs    =   np.abs(uvw[:, 1])
    b_uv    =   np.logical_and(uabs < umax, vabs < vmax)
    b_vis   =   (vis != NA)
    ids     =   np.where(np.logical_and(b_uv, b_vis))[0]

#    print(np.where(np.logical_not(b_vis))[0])
#    self.plot_uv(uvw, cellsize, nc, 'uv_direct_noclip.png')

    nvis0   =   len(vis)
    nvis    =   len(ids)
    if nvis0 != nvis:
        print('gen_image_direct(): select %d out of %d vis' % \
                (nvis, nvis0))
        uvw =   uvw[ids]
        vis =   vis[ids]
#        self.plot_uv(uvw, cellsize, nc, 'uv_direct_clip.png')

#    print(ids)
#    print('gen_image_direct(): clip ids dump...')
#    sys.exit(0)

    uv  =   uvw[:, :2]
    ws  =   self.gen_weight(uv)

    print(uv)

    vis =   vis * ws / np.mean(ws)

    arr =   (np.arange(nc) - nc/2) * util.mas2rad(cellsize) 

    if True:
        src =   self.srcs[0]
        arr_dec =   arr + src.dec
        arr_ra  =   arr / np.cos(src.dec) + src.ra
        ras, decs  =   np.meshgrid(arr_ra, arr_dec)
        ras     =   ras.flatten()[:, np.newaxis]
        decs    =   decs.flatten()[:, np.newaxis]
        rqu =   np.concatenate([np.cos(decs)*np.cos(ras), \
                                np.cos(decs)*np.sin(ras), \
                                np.sin(decs)], axis = 1)
        lmn =   np.einsum('ij,hj->hi', src.ruvw, rqu)

    else:
        ras, decs  =   np.meshgrid(arr, arr)
        ras     =   ras.flatten()[:, np.newaxis]
        decs    =   decs.flatten()[:, np.newaxis]
        rqu =   np.concatenate([np.cos(decs)*np.cos(ras), \
                                np.cos(decs)*np.sin(ras), \
                                np.sin(decs)], axis = 1)
        lmn =   np.roll(rqu, -1, axis=1)

    lmn001  =   np.array([0.0, 0.0, 1.0])[np.newaxis, :]

# npixel, 3
    lmn1    =   lmn - lmn001

##### Splitted: 
#    print('lmn1_uvw ...')
#    print('lmn1: ', lmn1.shape)
#    print('uvw: ', uvw.shape)
#    lmn1_uvw =   np.einsum('hk,jk->hj', lmn1, uvw)
#    print('exp(-2 pi lmn1_uvw) ...')
#    phase       =   2. * np.pi * (lmn1_uvw - np.floor(lmn1_uvw))
#    print('fringe X vis ...')
#    image1  =   np.einsum('ij,j->i', np.cos(phase), np.real(vis))
#    image2  =   np.einsum('ij,j->i', np.sin(phase), np.imag(vis))
#    image   =   (image1 + image2).reshape((nc, nc)) / nvis

##### Parallel:
    image   =   direct_image_np(lmn1, uvw, vis, nc, 12E9)
#    image   =   direct_image_cp(lmn1, uvw, vis, nc, 1E9)

##### Prototype:
#    fringe  =   np.exp(-2j * np.pi * lmn_uvw)
#    _img    =   np.einsum('ij,j->i', fringe, vis)
#    image   =   np.real(_img.reshape((nc, nc))) / len(vis)

#    self.plot_uv(uvw, cellsize, nc, 'uv_direct.png')
#    self.plot_image(image, cellsize, nc, 'image_direct.png')

#    print('Exit after uv_direct()')
#    sys.exit(0)

    if self.do_beam_correction:
        if not hasattr(self, 'res_src_rad'):
            print('Resolution (cs_src_rad) of source image must be set for beam correction!')
            print('Set source image resolution (cs_src_rad) to cellsize')
            self.cs_src_rad    =   util.mas2rad(self.cellsize)
#            sys.exit(0)
        bmaj, bmin, bps    =   util.calc_beam_param(uv, ws=ws)
#        bmaj, bmin, bps    =   util.calc_beam_param(uvw)
        pixel_per_beam      =   np.pi * bmaj * bmin / (4.*np.log(2)) \
                                / (self.cs_src_rad**2)
        print('bmaj: %f mas, bmin: %f mas, cs_src: %f mas, pixel_per_beam: %.2f'% \
                (util.rad2mas(bmaj), util.rad2mas(bmin), \
                 util.rad2mas(self.cs_src_rad), pixel_per_beam))

        image   /=  pixel_per_beam

    image   =   image[:, ::-1]
    return uvw, image

def gen_beam(self, bls, do_unif=False, do_rad=False):

    nc          =   self.nc
    cellsize    =   self.cellsize
    
#    bls =   np.load('vis.npy', allow_pickle=True)
    uv  =   []
    vis =   []
    for bl in bls:
        if bl == {}:
            continue
        vis_bl  =   bl['vis'].flatten()
        uvw_bl  =   bl['uvw_wav'].reshape((-1, 3))
        uv.append( uvw_bl[:, :2])
        uv.append(-uvw_bl[:, :2])
        vis.append(vis_bl)
        vis.append(np.conj(vis_bl))

    uv  =   np.concatenate(uv, axis = 0)
    vis =   np.concatenate(vis, axis = 0) 

    nvis0   =   uv.shape[0]

#    urange  =   1. / util.mas2rad(cellsize)
#    umin    =   -urange * 0.5  
#    vmin    =   umin
#    umax    =   urange * 0.5
#    vmax    =   umax
#    du      =   urange / nc
#    dv      =   du

    if not hasattr(self, 'umax'):
        self.set_uv_image_param()

    uabs    =   np.abs(uv[:, 0])
    vabs    =   np.abs(uv[:, 1])
    b_uv    =   np.logical_and(uabs < self.umax, vabs < self.vmax)
    b_vis   =   (vis != NA)
    ids     =   np.where(np.logical_and(b_uv, b_vis))[0]
    nvis    =   len(ids)
    if nvis0 != nvis:
        print('gen_beam_fft(): select %d out of %d vis' % \
                (nvis, nvis0))
        uv  =   uv[ids]

    ws      =   self.gen_weight(uv)
    beam_uv =   np.zeros((nc, nc), dtype = np.complex64)
    for i in range(nvis):
#        uf  =   (uv[i, 0] - umin) / du
#        vf  =   (uv[i, 1] - vmin) / dv
#        iu  =   int(uf + 0.5)
#        iv  =   int(vf + 0.5)
#        if iu >= nc or iv >= nc:
#            continue
        iv, iu  =   self.uv2id(uv[i, :])
        if iv < 0:
            continue
        beam_uv[iv, iu]    +=  ws[i]

    beam    =   np.real(fftshift(ifft2(fftshift(beam_uv))))
    beam    /=  np.max(beam)

    beam    =   beam[:, ::-1]

#    self.plot_beam(beam, cellsize, nc, 'beam.png')
    return beam

def gen_weight(self, uv):
    
    nc          =   self.nc
    cellsize    =   self.cellsize

    if not hasattr(self, 'urange'):
        self.set_uv_image_param()

    image_uv =   np.zeros((nc, nc), dtype = np.complex64)
#    print('vis shape: ', vis.shape)

    nvis    =   uv.shape[0]
    bc  =   np.zeros((nc, nc), dtype=int)

    for i in range(nvis):
        iv, iu  =   self.uv2id(uv[i, :])
        if iv < 0:
            continue
        bc[iv, iu]    +=  1

    ws  =   []
    for i in range(nvis):

        w   =   1.0
        iv, iu  =   self.uv2id(uv[i, :])

        if iv < 0:
            w   =   0.0

        if self.do_unif:
            w   /=  bc[iv, iu]
        
        if self.do_rad:
            uvrad   =   np.sqrt(uv[i,0]**2 + uv[i,1]**2)
#        w   *=  uvrad/1E6
            w   *=  uvrad/1E6

        ws.append(w)

    if self.do_rad:
        ws  /=  np.mean(ws)

    return np.array(ws)

def gen_image_fft(self, bls, do_unif=False, do_rad=False):

    nc          =   self.nc
    cellsize    =   self.cellsize

#    bls =   np.load('vis.npy', allow_pickle=True)
    uv  =   []
    vis =   []
    for bl in bls:
        if bl == {}:
            continue
        vis_bl  =   bl['vis'].flatten()
        uvw_bl  =   bl['uvw_wav'].reshape((-1, 3))
        uv.append( uvw_bl[:, :2])
        uv.append(-uvw_bl[:, :2])
        vis.append(vis_bl)
        vis.append(np.conj(vis_bl))

    uv  =   np.concatenate(uv, axis = 0)
    vis =   np.concatenate(vis, axis = 0) 

#    urange  =   1. / util.mas2rad(cellsize)
#    umin    =   -urange * 0.5  
#    vmin    =   umin
#    umax    =   urange * 0.5
#    vmax    =   umax
#    du      =   urange / nc
#    dv      =   du

    image_uv =   np.zeros((nc, nc), dtype = np.complex64)
    nvis0    =   vis.shape[0]
    print('vis shape: ', vis.shape)

    if not hasattr(self, 'umax'):
        self.set_uv_image_param()

    uabs    =   np.abs(uv[:, 0])
    vabs    =   np.abs(uv[:, 1])
    b_uv    =   np.logical_and(uabs < self.umax, vabs < self.vmax)
    b_vis   =   (vis != NA)
    ids     =   np.where(np.logical_and(b_uv, b_vis))[0]
    nvis    =   len(ids)
    if nvis0 != nvis:
        print('gen_image_fft(): select %d out of %d vis' % \
                (nvis, nvis0))
        uv  =   uv[ids]
        vis =   vis[ids]
#        self.plot_uv(uvw, cellsize, nc, 'uv_direct_clip.png')

    ws  =   self.gen_weight(uv)

#    bc  =   np.zeros((nc, nc), dtype=int)
#    for i in range(nvis):
#        uf  =   (uv[i, 0] - umin) / du
#        vf  =   (uv[i, 1] - vmin) / dv
#        iu  =   int(uf + 0.5)
#        iv  =   int(vf + 0.5)
#
#        if vis[i] == NA:
#            continue
#        if iu < 0 or iv < 0:
#            continue
#        if iu >= nc or iv >= nc:
#            continue
#        bc[iv, iu]    +=  1

    wsum    =   0.0
    for i in range(nvis):
#        uf  =   (uv[i, 0] - umin) / du
#        vf  =   (uv[i, 1] - vmin) / dv
#        iu  =   int(uf + 0.5)
#        iv  =   int(vf + 0.5)
#
        iv, iu  =   self.uv2id(uv[i, :])

        if iv < 0:
            continue

        if vis[i] == NA:
            continue

#        if do_unif:
#            w   /=  bc[iv, iu]
#        
#        if do_rad:
#            uvrad   =   np.sqrt(uv[i,0]**2 + uv[i,1]**2)
#            w   *=  uvrad/1E6

        wsum    +=  ws[i]
        image_uv[iv, iu]    +=  vis[i] * ws[i]

    image_uv    /=  wsum
    image   =   np.real(fftshift(ifft2(fftshift(image_uv))))
    image   *=  (nc*nc)

    if self.do_beam_correction:
        if not hasattr(self, 'res_src_rad'):
            print('Resolution (cs_src_rad) of source image must be set for beam correction!')
            print('Set source image resolution (cs_src_rad) to cellsize')
            self.cs_src_rad    =   util.mas2rad(self.cellsize)
#            sys.exit(0)
        bmaj, bmin, bps    =   util.calc_beam_param(uv, ws=ws)
        pixel_per_beam      =   np.pi * bmaj * bmin / (4.*np.log(2)) \
                                / (self.cs_src_rad**2)
        print('bmaj: %f mas, bmin: %f mas, cs_src: %f mas, pixel_per_beam: %.2f'% \
                (util.rad2mas(bmaj), util.rad2mas(bmin), \
                 util.rad2mas(self.cs_src_rad), pixel_per_beam))

        image   /=  pixel_per_beam

#  make it consistent with DFT 
    image   =   image[:, ::-1]
#    self.plot_uv(uv, cellsize, nc, 'uv_FFT.png')
#    self.plot_image(image, cellsize, nc, 'image_FFT.png')

    return uv, image

def gen_image_fft_nowt(self, bls):

    nc          =   self.nc
    cellsize    =   self.cellsize
    
#    bls =   np.load('vis.npy', allow_pickle=True)
    uv  =   []
    vis =   []
    for bl in bls:
        if bl == {}:
            continue
        vis_bl  =   bl['vis'].flatten()
        uvw_bl  =   bl['uvw_wav'].reshape((-1, 3))
        uv.append( uvw_bl[:, :2])
        uv.append(-uvw_bl[:, :2])
        vis.append(vis_bl)
        vis.append(np.conj(vis_bl))

    uv  =   np.concatenate(uv, axis = 0)
    vis =   np.concatenate(vis, axis = 0) 

    urange  =   1. / util.mas2rad(cellsize)
    umin    =   -urange * 0.5  
    vmin    =   umin
    umax    =   urange * 0.5
    vmax    =   umax
    du      =   urange / nc
    dv      =   du

    image_uv =   np.zeros((nc, nc), dtype = np.complex64)
    nvis0    =   vis.shape[0]
    print('vis shape: ', vis.shape)

    uabs    =   np.abs(uv[:, 0])
    vabs    =   np.abs(uv[:, 1])
    b_uv    =   np.logical_and(uabs < umax, vabs < vmax)
    b_vis   =   (vis != NA)
    ids     =   np.where(np.logical_and(b_uv, b_vis))[0]
    nvis    =   len(ids)
    if nvis0 != nvis:
        print('gen_image_fft(): select %d out of %d vis' % \
                (nvis, nvis0))
        uv  =   uv[ids]
        vis =   vis[ids]
#        self.plot_uv(uvw, cellsize, nc, 'uv_direct_clip.png')

    for i in range(nvis):
        uf  =   (uv[i, 0] - umin) / du
        vf  =   (uv[i, 1] - vmin) / dv
        iu  =   int(uf + 0.5)
        iv  =   int(vf + 0.5)
        if iu >= nc or iv >= nc:
#            print('skip (%d,%d)' % (iu, iv))
            continue
        image_uv[iv, iu]    +=  vis[i]

    image_uv    /=  nvis
    image   =   np.real(fftshift(fft2(fftshift(image_uv))))

#    self.plot_uv(uv, cellsize, nc, 'uv_FFT.png')
#    self.plot_image(image, cellsize, nc, 'image_FFT.png')

    return uv, image

def plot_uv(self, uvw, cellsize, nc, name):

    uv  =   uvw[:, :2]

    urange  =   1. / util.mas2rad(cellsize)
    umin    =   -urange * 0.5  

    sc  =   1E6
    rc('font', size=13)
    plt.clf()
    fig =   plt.figure()
    fig.set_figwidth(5.5)
    fig.set_figheight(5)
    fig.subplots_adjust(left=0.15, right=0.97, top=0.95, bottom=0.12)
 
    ax  =   fig.add_subplot()
    ax.set_aspect('equal')
    plt.plot(uv[:, 0]/sc, uv[:, 1]/sc, marker = '.', c = 'steelblue', \
            ls = 'none', ms = 1)
    ax.set_xlabel('$u [10^6\lambda]$')
    ax.set_ylabel('$v [10^6\lambda]$')
    ax.set_xlim(umin/sc, -umin/sc) 
    ax.set_ylim(umin/sc, -umin/sc) 
    plt.savefig(name)

def plot_beam(self, beam, cellsize, nc, name, **kw):

#    cellsize    /=  1E3

    cs      =   cellsize
    hcs     =   cellsize * 0.5
    s       =   cellsize * nc * 0.5 
    vmin    =   np.min(beam)
    vmax    =   np.max(beam)

    if 'uv' in kw.keys():
        uv  =   kw['uv']
        ws  =   self.gen_weight(uv)
        bmaj, bmin, bpa =   util.calc_beam_param(uv, ws=ws)
    if 'beam_param' in kw.keys():
        bmaj, bmin, bpa =   kw['beam_param']

    plt.clf()
    fig =   plt.figure()

    plt.clf()
    rc('font', size=13)
 
    fig =   plt.figure()
    fig.set_figwidth(6.3)
    fig.set_figheight(5)
    fig.subplots_adjust(left=0.15, right=0.97, top=0.95, bottom=0.12)
 
    left    =   -(nc/2-1) * cs - hcs
    right   =   (nc/2) * cs + hcs
    bot     =   -(nc/2) * cs - hcs
    top     =   (nc/2-1) * cs + hcs
    extent  =   (left, right, bot, top)

    ax  =   fig.add_subplot()
    im  =   ax.imshow(beam, vmin = vmin, vmax = vmax, \
                origin = 'lower', cmap = plt.get_cmap('rainbow'), \
                extent = extent)

    if 'bmaj' in vars():
        bmaj    =   util.rad2mas(bmaj)
        bmin    =   util.rad2mas(bmin)
        print('plot_beam(): bmaj %f mas, bmin %f mas' % (bmaj, bmin))
        e   =   Ellipse(xy=[0,0], width=bmin, height=bmaj, \
            angle=bpa/np.pi*180., color='k', fill=False, lw = 0.5)
        ax.add_artist(e)

    ax.set_xlabel('X [mas]')
    ax.set_ylabel('Y [mas]')
#    cb  =   plt.colorbar(im, ax = ax)
#    cb.ax.set_ylabel('Flux [Jy]', rotation=90, va='bottom')
    cb  =   plt.colorbar(im, orientation='vertical')
    cb.ax.set_ylabel('Strength')

    plt.savefig(name)

def plot_image(self, image, cellsize, nc, name):

#    cellsize    /=  1E3
    image   *=  1E3

    hcs     =   cellsize * 0.5
    s       =   cellsize * nc * 0.5
    vmin    =   np.min(image)
    vmax    =   np.max(image)

    plt.clf()
    fig =   plt.figure()

    rc('font', size=13)
    fig.set_figwidth(6.3)
    fig.set_figheight(5)
    fig.subplots_adjust(left=0.15, right=0.97, top=0.95, bottom=0.12)
    ax  =   fig.add_subplot()
    im  =   ax.imshow(image, vmin = vmin, vmax = vmax, \
                origin = 'lower', cmap = plt.get_cmap('rainbow'), \
                extent = (-s, s, -s, s))
    ax.set_xlabel('X [mas]')
    ax.set_ylabel('Y [mas]')
#    cb  =   plt.colorbar(im, ax = ax)
#    cb.ax.set_ylabel('Flux [Jy]', rotation=90, va='bottom')
    cb  =   plt.colorbar(im, orientation='vertical')
    cb.ax.set_ylabel('Flux [mJy/beam]')

    plt.savefig(name)

# coords in rad, cs in mas
def plot_src(self, coords, cs, nc, name):

    plt.clf()
    fig =   plt.figure()
    ax  =   fig.add_subplot()

    hs  =   cs * nc * 0.5 / 60E3
    
    coords  *=  180./np.pi * 60. # to arcmin
    plt.plot(coords[:, 0], coords[:, 1], 'r+')

    plt.xlim(-hs, hs)
    plt.ylim(-hs, hs)

    ax.set_xlabel('X [arcmin]')
    ax.set_ylabel('Y [arcmin]')

    plt.savefig(name)

if __name__ == '__main__':
    main()
