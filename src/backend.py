#!/usr/bin/env python

import numpy as np
#import cupy as cp

from multiprocessing import Process, Queue

def worker_cupy(pid, npixel_seg, npixel, lmn1, uvw, vis, q0, q1):

    print('pid %d, set gpu ...' % (pid))
    cp.cuda.Device(pid).use()

    lmn1_d =   cp.array(lmn1)
    uvw_d  =   cp.array(uvw)
    vis_d  =   cp.array(vis)

    def kernel_cupy(_lmn1_d):
        lmn1_uvw_d  =   cp.einsum('hk,jk->hj', _lmn1_d, uvw_d)
        ph_d        =   2. * np.pi * (lmn1_uvw_d - cp.floor(lmn1_uvw_d))
        image1_d    =   cp.einsum('ij,j->i', cp.cos(ph_d), cp.real(vis_d))
        image2_d    =   cp.einsum('ij,j->i', cp.sin(ph_d), cp.imag(vis_d))
        image_d =   image1_d + image2_d
        return image_d.get()

    while not q0.empty():
        id_seg  =   q0.get()

        i0  =   id_seg * npixel_seg 
        i1  =   i0 + npixel_seg
        if i1 > npixel:
            i1  =   npixel
 
        print('pid %d, seg %d ...' % (pid, id_seg))
        buf    =   kernel_cupy(lmn1_d[i0:i1, :])
        q1.put((i0, i1, buf))

# lmn1: (npixel, 3), uvw: (nvis, 3), vis: nvis
def direct_image_cp(lmn1, uvw, vis, nc, s_mem_max):

    s_mem_max   =   10E9

    ngpu    =   2

    npixel  =   lmn1.shape[0] 
    nvis    =   uvw.shape[0]

    s_pixel     =   nvis * 8
    s_mem_tot   =   s_pixel * npixel
    
    npixel_seg  =   int(s_mem_max / s_pixel)
    nseg        =   int(np.ceil(npixel / npixel_seg))

    image       =   np.zeros(npixel, dtype=np.float32)

    print('############# direct_image_np ##############')
    print('Total Pixel num:     %d' % npixel)
    print('Vis num:             %d' % nvis)
    print('Max mem size:        %.3f GB' % (s_mem_max/1E9))
    print('Required mem size:   %.3f GB' % (s_mem_tot/1E9))
    print('Segment num:         %d' % (nseg))
    print('Pixels per seg:      %d' % (npixel_seg))
    print('############# direct_image_np ##############')

    q0  =   Queue()
    q1  =   Queue()
    for i in range(nseg):
        q0.put(i)

    procs   =   []
    for i in range(ngpu):
        proc    =   Process(target=worker_cupy, args=(i, npixel_seg, npixel,\
                            lmn1, uvw, vis, q0, q1))
        proc.start()
        procs.append(proc)

    for i in range(nseg):
        i0, i1, buf =   q1.get()
        image[i0:i1] =   buf[:]

    for proc in procs:
        proc.join()

#    for id_seg in range(nseg):
    for id_seg in []:
        print('seg %d ...' % (id_seg))
        i0  =   id_seg * npixel_seg 
        i1  =   i0 + npixel_seg
        if i1 > npixel:
            i1  =   npixel
        image[i0:i1]    =   kernel(lmn1_d[i0:i1, :])

    return image.reshape((nc, nc)) / nvis

# lmn1: (npixel, 3), uvw: (nvis, 3), vis: nvis
def direct_image_np(lmn1, uvw, vis, nc, s_mem_max):

    npixel  =   lmn1.shape[0] 
    nvis    =   uvw.shape[0]

    s_pixel     =   nvis * 8
    s_mem_tot   =   s_pixel * npixel
    
    npixel_seg  =   int(s_mem_max / s_pixel)
    nseg        =   int(np.ceil(npixel / npixel_seg))

    image       =   np.zeros(npixel, dtype=np.float32)

    print('############# direct_image_np ##############')
    print('Total Pixel num:     %d' % npixel)
    print('Vis num:             %d' % nvis)
    print('Max mem size:        %.3f GB' % (s_mem_max/1E9))
    print('Required mem size:   %.3f GB' % (s_mem_tot/1E9))
    print('Segment num:         %d' % (nseg))
    print('Pixels per seg:      %d' % (npixel_seg))
    print('############# direct_image_np ##############')

    def kernel(_lmn1):

        lmn1_uvw    =   np.einsum('hk,jk->hj', _lmn1, uvw)
        ph          =   -2. * np.pi * (lmn1_uvw - np.floor(lmn1_uvw))
        image1      =   np.einsum('ij,j->i', np.cos(ph), np.real(vis))
        image2      =   np.einsum('ij,j->i', np.sin(ph), np.imag(vis))
        return image1 + image2

    for id_seg in range(nseg):

        print('seg %d ...' % (id_seg))

        i0  =   id_seg * npixel_seg 
        i1  =   i0 + npixel_seg
        if i1 > npixel:
            i1  =   npixel
        image[i0:i1]    =   kernel(lmn1[i0:i1, :])

    return image.reshape((nc, nc)) / nvis
