import numpy as np
from matplotlib import pyplot as plt
from numpy import exp,pi,sin,cos,sqrt,arctan2,tan   
import tifffile as tiff
from skimage import io

def saveResult(array,title):
    array = array-np.amin(array)
    array = 65535*array/np.amax(array)
    svPath = title + '.tif'
    array = array.astype(np.uint16)
    io.imsave(svPath,array)
    return []

print('Generating sample...')
w = 40
a = 10
b = 512-a*w
ele1 =  np.zeros((w, w, w))
ele1[0,0,0] = 1
ele1[w-1,w-1,w-1] = 1
ele2 = np.tile(ele1, (a,a,a)) 
ele3 = np.pad(ele2, ((0,b),(int(b/2),int(b/2)),(int(b/2),int(b/2))), 'constant', constant_values=0) 


tiff.imwrite('sample_sep.tif',ele3,photometric='minisblack')



print('Loading detection PSF...')
PSF = tiff.imread('PSF BW.tif')
saveResult(PSF,'detection PSF')

print('Calculating detection OTF...')
OTF = (np.fft.fftn(PSF))
del PSF 
saveResult(np.abs(OTF),'detection OTF')

print('Loading excitation PSF...')
out_put = np.float32(np.load('out_put.npy')) 
saveResult(out_put,'excitation PSF')

print('Generating lightsheet...')
sheet = out_put[256,:,:]
del out_put
sheet = np.tile(sheet,(512, 1, 1))
sheet = np.moveaxis(sheet,0,1)
saveResult(sheet,'sheet')

print('Rolling sample')
temp = np.float32(np.zeros((512,512)))
sample = np.roll(ele3,20,axis=0)

print('Calculating fluoresence response')
temp = np.multiply(sheet,sample)
# temp = np.copy(sample) 
saveResult(temp,'response')

print('Calculating FT of fluoresence response')
temp = (np.fft.fftn(temp))

print('Clipping frequencies')
temp = np.multiply(temp,OTF)
saveResult(np.abs(temp),'F_spectrum')

print('Moving to image space')
temp = np.fft.fftshift(np.fft.ifftn(temp))

print('Calculating detected signal')
temp = np.sum(np.abs(temp),axis=0)

print('Saving results')
saveResult(temp,'result')
