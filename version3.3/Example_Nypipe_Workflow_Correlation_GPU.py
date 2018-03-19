#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 16:13:34 2017

@author: acamargo
"""
### This function computes the correlation matrix and the matrix of p-values
### from lib.cuda_rsfmri_correlation import cuda_pcorrcoef_loop

from nipype import Node, Function


imports = ['from lib.fileIO import load',
           'import numpy as np',
           'from lib.cuda_rsfmri_correlation import cuda_calc_fact_corr',
           'from lib.cuda_pcorrcoef_loop import cuda_pcorrcoef_loop',
           'import os'
]
def fmri_correlation(in_file_fmricorr):
    
    path = os.path.dirname(in_file_fmricorr)
    filename_tfn = '/aal_tpl'
    filename_rfn = '/' + os.path.basename(in_file_fmricorr)

    data_tfn, affine_tfn, hdr_tfn = load("{}{}.nii".format(path, filename_tfn))
    data_rfn, affine_rfn, hdr_rfn = load("{}{}".format(path, filename_rfn))

    datashape_rfn = data_rfn.shape
    
    rimg_prod = (datashape_rfn[0])*(datashape_rfn[1])*(datashape_rfn[2])
    
    rimg = data_rfn.reshape((rimg_prod,datashape_rfn[3]), order = 'F').copy()
    
    rimg = rimg.transpose()

    tmp = np.unique(data_tfn)


    numroi = tmp.shape[0] - 1


    vector_tfn = (data_tfn.transpose()).flatten()
    
    
    N = datashape_rfn[3]  ## 
   

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


 
    #p_loop, R_loop = pcorrcoef_loop(meants.transpose())
    matrix_test = meants.transpose()
    mx = np.ascontiguousarray(matrix_test, dtype=np.float32)
    p_cuda, r_cuda = cuda_pcorrcoef_loop(mx)
    
    fileout_corr_mat = path + '/cuda_corr_matrix.txt'
    fileout_pvalue_mat = path + '/cuda_pvalue_matrix.txt'

    np.savetxt(fileout_corr_mat, R_loop, delimiter=',') 
    np.savetxt(fileout_pvalue_mat, p_loop, delimiter=',') 
    
    return p_loop, R_loop
#%%
## Creating the node that executes the fmri_correlation function

fmricorr = Node(Function(input_names=["in_file_fmricorr"],
                         output_names=["p_loop", "R_loop" ],
                         function=fmri_correlation,
                         imports=imports),
                name='fmricorr')
#%%
# Testing the node function

### This is the parameters to run the program on the GPU
#TPB = 16  # Threads per block
#bpg = (1,1)  # Blocks per Grid
#tpb = (1,900)  # Threads per Block

fmricorr.inputs.in_file_fmricorr = "/data/swrfunc.nii"    
fmricorr.run()
print('Done')

results = fmricorr.result.outputs
print(results)
print('Done')
#%%
### Now we can test this node function in a workflow
 
from nipype import Node, Workflow
from nipype.interfaces import fsl

from os.path import abspath
in_file = abspath("/data/swrfunc.nii")

# Smooth process
smooth = Node(fsl.IsotropicSmooth(
    in_file="/data/swrfunc.nii",
    out_file="/data/smooth_swrfunc.nii.gz",
    fwhm=4),
    name = "smooth")


fmricorr = Node(Function(input_names=["in_file_fmricorr"],
                         output_names=["p_loop", "R_loop" ],
                         function=fmri_correlation,
                         imports=imports),
                name="fmricorr")

wf = Workflow(name="smoothflow_fmricorr")


wf.connect(smooth, "out_file", fmricorr, "in_file_fmricorr")

wf.write_graph("workflow_graph.dot")
from IPython.display import Image
Image(filename="workflow_graph.dot.png")


wf.write_graph(graph2use='flat')
from IPython.display import Image
Image(filename="graph_detailed.dot.png")

# Specify the base directory for the working directory
wf.base_dir = "working_dir"

# Execute the workflow
wf.run()
print('Done')
#%%


