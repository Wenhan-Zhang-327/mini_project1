import numpy as np
from matplotlib import pyplot as plt
from numpy import exp,pi,sin,cos,sqrt,arctan2,tan
import tifffile as tiff
from skimage import io
from scipy.ndimage import rotate,affine_transform


def saveResult(array,title):
    array = array-np.amin(array)
    array = 65535*array/np.amax(array)
    svPath = title + '.tif'
    array = array.astype(np.uint16)
    io.imsave(svPath,array)
    return []

print('Generating sample...')
w = 20
a = 5
b = 512-a*w
ele1 =  np.zeros((int(w),w, w))
ele1[0,0,0] = 100
ele1[int(w)-1,0,0] = 100
ele2 = np.tile(ele1, (a,a,a)) 
ele3 = np.pad(ele2, ((0,int(b)),(int(b/2+w/2),int(b/2-w/2)),(int(b/2+w/2),int(b/2-w/2))), 'constant', constant_values=0) 
tiff.imwrite('sample_sep.tif',ele3,photometric='minisblack')

print('Loading detection PSF...')
PSF = tiff.imread('PSF BW_1_512.tif')
PSF = np.moveaxis(PSF,1,2)
saveResult(PSF,'detection PSF')

print('Calculating detection OTF...')
OTF = (np.fft.fftn(PSF))
del PSF 
saveResult(np.abs(OTF),'detection OTF')

print('Loading excitation PSF...')
out_put = tiff.imread('result_512.tif')
# out_put = np.float32(np.load('out_put.npy')) 
saveResult(out_put,'excitation PSF')

print('Generating lightsheet...')
sheet = out_put[:,256,:]
del out_put
sheet = np.tile(sheet,(512, 1, 1))
sheet = np.moveaxis(sheet,0,2)
saveResult(sheet,'sheet')

nShift = 80
stepSize = 5
final = np.zeros((nShift,512,512))


for shift in range(0,nShift):

    print('Rolling sample')
    temp = np.float32(np.zeros((512,512)))
    sample = np.roll(ele3,10+stepSize*shift,axis=0)

    print('Calculating fluoresence response')
    temp = np.multiply(sheet,sample)
    saveResult(temp,'response')

    print('Calculating FT of fluoresence response')
    temp = (np.fft.fftn(temp))

    print('Clipping frequencies')
    temp = np.multiply(temp,OTF)
    saveResult(np.abs(temp),'F_spectrum')

    print('Moving to image space')
    temp = np.fft.fftshift(np.fft.ifftn(temp))

    print('Calculating detected signal')
    temp = np.abs(temp[256,:,:])

    final[shift,:,:] = temp
print('Saving results')
saveResult(final,'wf_result')
final1 = rotate(final,8,axes=(0,1),reshape=True)
saveResult(final1,'wf_result_rot')
deg=8
lmd = tan(np.deg2rad(deg))
scl = cos(np.deg2rad(deg))
mat_shear = np.array([[1,0,0,0],[lmd,1,0,0],[0,0,1,0],[0,0,0,1]])
mat_scale = np.array([[1,0,0,0],[0,1,0,0],[0,0,scl,0],[0,0,0,1]])
final2 = affine_transform(final1, np.matmul(mat_shear,mat_scale))
saveResult(final2,'wf_result_aff')
