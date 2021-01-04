import numpy as np
from matplotlib import pyplot as plt
from numpy import exp,pi,sin,cos,sqrt,arctan2,tan   
import tifffile as tiff
ele = np.array([[[1,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]],[[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]],[[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]],[[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]])
ele2 = np.tile(ele, (40,40,40)) 
out_put = np.load('out_put.npy') 
PSF = tiff.imread('PSF BW.tif')
image_sams = np.zeros((512,512,512))
for i in range(353):
    ele3 = np.pad(ele2, ((i,352-i),(176,176),(176,176)), 'constant', constant_values=0) 
    out_put_sample = np.multiply(out_put,ele3)
    OTF = np.fft.fftshift(np.fft.fftn(PSF))
    sampleF = np.fft.fftshift(np.fft.fftn(out_put_sample))
    imageF = sampleF * OTF
    image = np.fft.fftshift(np.fft.ifftn(imageF))
    image_sams[i,:,:] = np.sum(np.abs(image),axis=0)
tiff.imwrite('image_move.tif',image_sams,photometric='minisblack')
