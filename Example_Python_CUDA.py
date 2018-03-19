#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 10:59:45 2017

@author: acamargo
"""

#%%
import os,sys,time
import pandas as pd
import numpy as np
import numba
from numba import cuda, float32
from numba import cuda, float32
import math

# Controls threads per block and shared memory usage.
# The computation will be done on blocks of TPBxTPB elements.
TPB = 16
bpg = (1,1) 
tpb = (1,3) 

my_gpu = numba.cuda.get_current_device()

n = 3
D = np.arange(n, dtype=np.float32)
E = np.arange(n, dtype=np.float32)
T = np.empty([1,2])   

thread_ct = my_gpu.WARP_SIZE
block_ct = int(math.ceil(float(n) / thread_ct))

print(block_ct)
@cuda.jit
def calcu_sum(D,E,T):

    ty = cuda.threadIdx.y
    bh = cuda.blockDim.y
    
    index_i = ty
    sbuf = cuda.shared.array(tpb, float32)

    L = len(D)
    su = 0
    while index_i < L:
        su += D[index_i]
        index_i +=bh

#    print('su:',su)

    sbuf[0,ty] = su
    cuda.syncthreads()

    if ty == 0:
        T[0,0] = 0
        T[0,1] = 0
        for i in range(0, bh):
            T[0,0] += sbuf[0,i]**2
            T[0,1] += sbuf[0,i]
#        print('T:',T[0,0])



#D = np.array([ 0.42487645,0.41607881,0.42027071,0.43751907,0.43512794,0.43656972,
#               0.43940639,0.43864551,0.43447691,0.43120232], dtype=np.float32)
#T = np.empty([1,1])
#print('D: ',D)

stream = cuda.stream()
with stream.auto_synchronize():
    dD = cuda.to_device(D, stream)
    dE = cuda.to_device(E, stream)
    dT= cuda.to_device(T, stream)
    calcu_sum[bpg, tpb, stream](dD,dE,dT)
    T = dT.copy_to_host()