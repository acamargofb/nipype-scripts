__author__ = 'tattwei'


import nibabel as nib
import pandas as pd
from dipy.align.reslice import reslice


# load NIFTI image
def load(filename):
    fobj = nib.load(filename)
    img = fobj.get_data()
    affine = fobj.affine
    header = fobj.header
    return img, affine, header

# save NIFTI image
def save(data, affine, filepath):
    img_nifti = nib.Nifti1Image(data, affine)
    nib.save(img_nifti, filepath)

def savewhdr(data, affine, hdr, filepath):
    img_nifti = nib.Nifti1Pair(data, affine)
    nib.save(img_nifti, filepath)

# read numpy array from excel
def loadAff(filename):
    df = pd.read_excel(filename)
    affine = pd.DataFrame.as_matrix(df)
    return affine


# write numpy array (affine matrix) to excel
def saveAff(filename, affine):
    maxCoords = affine.shape
    #print maxCoords
    df = pd.DataFrame(  data=affine[0:,0:],    # values
                    index=range(1,maxCoords[0]+1),    # 1st column as index
                    columns=range(1, maxCoords[1]+1))  # 1st row as the column names
    # Create a Pandas Excel writer using XlsxWriter as the engine.
    writer = pd.ExcelWriter(filename, engine="xlsxwriter")
    # Convert the dataframe to an XlsxWriter Excel object.
    df.to_excel(writer, sheet_name='Affine')
    # Close the Pandas Excel writer and output the Excel file.
    writer.save()


# read the image sampling
def imgStack(filename):
    img, affine, hdr = load(filename)
    zooms =  hdr.get_zooms()[:3]
    print "MRI stack {} has size {} and zooms {}".format(filename, img.shape, zooms)
    return zooms


# reslice the images according to a reference
def resliceStack(path, filename,new_zooms, interporder):
    img, affine, hdr = load("{}{}".format(path,filename))
    zooms = hdr.get_zooms()[:3]
    reslicedImg, reslicedAffine = reslice (img, affine, zooms, new_zooms, order = interporder, num_processes=0)
    #newfilename = "{}resliced_{}".format(path,filename)
    #save(reslicedImg, reslicedAffine, newfilename)
    return reslicedImg, reslicedAffine

