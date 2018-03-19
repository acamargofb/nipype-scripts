#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 17:08:20 2017

@author: acamargo
"""

__author__ = 'tattwei'


import time
from fileIO import load
from pcoeffcorr import pcorrcoef
from pcorrcoef_loop import pcorrcoef_loop
#import matplotlib.pyplot as plt
import numpy as np

tic = time.clock()

#path = "KETUM/"
#filename = 'C6442'

#path = "/home/acamargo/Documents/2017/July-Sept/GSOC2017/data/"
path = "/home/brain/neuro-csg-pipelines/Projects/Nipype/correlation/"
filename_tfn = 'aal_tpl'
filename_rfn = 'swrfunc'


filename_tfn = 'aal_tpl'
filename_rfn = 'swrfunc'

data_tfn, affine_tfn, hdr_tfn = load("{}{}.nii".format(path, filename_tfn))
data_rfn, affine_rfn, hdr_rfn = load("{}{}.nii".format(path, filename_rfn))

print "Image Stack has dimensions {}".format(data_rfn.shape)

datashape_rfn = data_rfn.shape

## MATLAB: rimg = reshape(rimg,prod(sz(1:3)),sz(4))';

rimg_prod = (datashape_rfn[0])*(datashape_rfn[1])*(datashape_rfn[2])

rimg = data_rfn.reshape((rimg_prod,datashape_rfn[3]), order = 'F').copy()

###% MATLAB: tmp = unique(timg(:));      % get ROI values within ROI image, 0 - background
rimg = rimg.transpose()

tmp = np.unique(data_tfn)


# MATLAB: numroi = length(tmp)-1;

numroi = tmp.shape[0] - 1

# MATLAB:
#meants = [];                % mean time series for each ROIs
#for ctr=1:numroi
#    ix = find(timg(:) == ctr);  % get index for each ROI
#    ts = rimg(:,ix);        % extract time series within the ROI

#end




vector_tfn = (data_tfn.transpose()).flatten()

#np.concatenate((a, b), axis=0)
#a[0:3][:,4:9]

N = datashape_rfn[3]  ## 260 for initial example

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


#R = np.corrcoef(meants.transpose()) 
#pR, RR = pcorrcoef(meants.transpose())  # The values obtained for p-value are not correct in comparison with the
# results of the p-value computed in MATLAB
p_loop, R_loop = pcorrcoef_loop(meants.transpose())

toc = time.clock()
print toc - tic, 'Elapased time in seconds' 