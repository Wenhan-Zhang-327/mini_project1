import numpy as np
from matplotlib import pyplot as plt
from numpy import exp,pi,sin,cos,sqrt,arctan2,tan   
import tifffile as tiff
w = 40
a = 10
b = 512-a*w
ele1 =  np.zeros((w, w, w))
ele1[0,0,0] = 1
ele1[w-1,w-1,w-1] = 1
ele2 = np.tile(ele1, (a,a,a)) 
ele3 = np.pad(ele2, ((0,b),(int(b/2),int(b/2)),(int(b/2),int(b/2))), 'constant', constant_values=0) 
plt.imshow(ele3[0,:,:],cmap = 'gray')
tiff.imwrite('sample_sep.tif',ele3,photometric='minisblack')

out_put = np.load('out_put.npy') 
PSF = tiff.imread('PSF BW.tif')
OTF = np.fft.fftshift(np.fft.fftn(PSF))
image_sams = np.zeros((512,512,512))
for i in range(353):
    sample = np.roll(ele3,i,axis=0)
    out_put_sample = np.multiply(out_put,sample)
    sampleF = np.fft.fftshift(np.fft.fftn(out_put_sample))
    imageF = sampleF * OTF
    image = np.fft.fftshift(np.fft.ifftn(imageF))
    image_sams[i,:,:] = np.sum(np.abs(image),axis=0)
tiff.imwrite('image_move.tif',image_sams,photometric='minisblack')
