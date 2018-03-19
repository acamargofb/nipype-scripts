#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 07:04:28 2017

@author: acamargo
"""


import time
from lib.fileIO import load
from lib.cuda_pcorrcoef_loop import cuda_pcorrcoef_loop
import numpy as np

tic = time.clock()



path = "/data/"




filename_tfn = 'aal_tpl'
filename_rfn = 'swrfunc'

data_tfn, affine_tfn, hdr_tfn = load("{}{}.nii".format(path, filename_tfn))
data_rfn, affine_rfn, hdr_rfn = load("{}{}.nii".format(path, filename_rfn))

print "Image Stack has dimensions {}".format(data_rfn.shape)

datashape_rfn = data_rfn.shape



rimg_prod = (datashape_rfn[0])*(datashape_rfn[1])*(datashape_rfn[2])

rimg = data_rfn.reshape((rimg_prod,datashape_rfn[3]), order = 'F').copy()


rimg = rimg.transpose()

tmp = np.unique(data_tfn)


numroi = tmp.shape[0] - 1

vector_tfn = (data_tfn.transpose()).flatten()


N = datashape_rfn[3]  

toc = time.clock()
print toc - tic, 'Elapsed time in seconds before for' 

tic = time.clock()

for cr in xrange(1, numroi + 1):
    ix =  np.nonzero(vector_tfn == cr)
    ts =  rimg[:, ix]
    ts = ts.squeeze()
    if cr == 1:
         meants_temp = np.mean(ts,axis=1)
         meants = meants_temp.reshape((N,1), order = 'F').copy()
    else:
        meants_temp = np.mean(ts,axis=1)
        meants_temp = meants_temp.reshape((N,1), order = 'F').copy()
        meants = np.append(meants, meants_temp, axis=1)   


  
toc = time.clock()
print toc - tic, 'Elapased time in seconds for the for loop' 
     
tic = time.clock()

matrix_test =  meants.transpose()
mx = np.ascontiguousarray(matrix_test, dtype=np.float32)

p_cuda, r_cuda = cuda_pcorrcoef_loop(mx)
    
toc = time.clock()
print toc - tic, 'Elapased time in seconds for the pearson correlation' 


#%%


