import numpy as np
from scipy.ndimage import rotate

def getPhaseMask(w,z1,z2,z3,z4):
    """
    Calculates the phase mask to reproduce the 4 main aberations for high NA objectives
    ----------
    w: width of square mask
    
    z1: defocus
    z2, z3: lateral and vertical astigmatism
    z4: spherical aberration
    
    Returns
    -------
    mask:
        W x W square array containing the phase mask
    
    """
    lims = np.linspace(-1/2,1/2,w)
    x, y = np.meshgrid(lims,lims)
    rho = np.sqrt(x**2 + y**2) 
    phi = np.arctan2(y, x)
    
    mask = np.zeros([w,w])
    
    mask += z1* np.sqrt(3)*(2*(rho**2) - 1)
    mask += z2* np.sqrt(6)*(rho**2)*np.cos(2*phi)
    mask += z3* np.sqrt(6)*(rho**2)*np.sin(2*phi)
    mask += z4* np.sqrt(5)*(6*(rho**4) - 6*(rho**2) + 1)
   
    mask = mask % (2 * np.pi) -np.pi
        
    return mask

def addPhaseFeature(mask,feature,Z1,Z2):
    """
    Calculates the phase mask to reproduce typical PSF modification
    ----------
    mask: mask to be altered
    feature: Phase feature to add
    Z1: variable parameter
    Z2: variable parameter
    
    Returns
    -------
    mask:
        W x W square array containing the altered phase mask
    
    """
    w = mask.shape[0]
    lims = np.linspace(-1/2,1/2,w)
    x, y = np.meshgrid(lims,lims)
    rho = np.sqrt(x**2 + y**2) 
    phi = np.arctan2(y, x)
    if feature == 'STED':

        mask = mask+phi*Z1
  
    elif feature == 'bessel':

        mask = mask+np.pi*2*np.cos(Z1*rho)

    elif feature == 'bessel phase':

        mask = mask+Z1*rho

    elif feature == 'bottle':
        
        mask[rho<Z1] += np.pi/2
        mask[rho<Z2] += 0

    mask = mask % (2 * np.pi) -np.pi
        
    return mask

def getPupil(w,beam,Z1,Z2):
    """
    Calculates the beam amplitude at the back of the lens
    ----------
    w: size of output array
    beam: type of beam to use
    Z1: variable parameter
    Z2: variable parameter
    
    Returns
    -------
    mask:
        W x W square array containing the altered phase mask
    
    """
    lims = np.linspace(-1,1,w)
    x, y = np.meshgrid(lims,lims)
    rho = np.sqrt(x**2 + y**2) 
    phi = np.arctan2(y, x)

    pupil = np.zeros([w,w])

    if beam == 'Gaussian':

        pupil = np.exp(-rho**2/(2*Z1**2))
    
    elif beam == 'Airy':

        pupil[rho<Z1] = 1
        pupil[rho<Z2] = 0

    elif beam == 'SIM':

        pupil[rho<Z1] = 1
        pupil = np.roll(pupil,(int(Z2*w))) + np.roll(pupil,(-int(Z2*w)))

    elif beam =='SPIM':

        pupil[rho<Z1] = 1
        pupil = np.roll(pupil,(int(Z2*w)))

    else:
        pupil[rho<Z1] = 1

    pupil = pupil/np.amax(pupil)

    return pupil

def shiftSpot(pupil, beamCentre):

    w = pupil.shape[0]
    padLength = int(np.round(w/2))
    lims = np.linspace(-1,1,w)
    x, y = np.meshgrid(lims,lims)
    rho = np.sqrt(x**2 + y**2) 
    pupil[rho>1]=0
    pupil = pupil[::2,::2]
    newPupil = np.pad(pupil, ((padLength, padLength), (padLength, padLength)), 'constant')
    #newPupil = np.roll(newPupil,int(beamCentre*padLength))


    return newPupil

def addLinearPhase(mask,tilt):

    w = mask.shape[0]
    lims = np.linspace(-1,1,w)
    x, y = np.meshgrid(lims,lims)
    linPhase = x*tilt
    linPhase = linPhase % (2 * np.pi) -np.pi
    mask = mask + linPhase
    mask = mask % (2 * np.pi) -np.pi

    return mask

def rotate_nn(data, angle, axes):
    """
    Rotate a `data` based on rotating coordinates.
    """

    # Create grid of indices
    shape = data.shape
    d1, d2, d3 = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]

    # Rotate the indices
    d1r = rotate(d1, angle=angle, axes=axes)
    d2r = rotate(d2, angle=angle, axes=axes)
    d3r = rotate(d3, angle=angle, axes=axes)

    # Round to integer indices
    d1r = np.round(d1r)
    d2r = np.round(d2r)
    d3r = np.round(d3r)

    d1r = np.clip(d1r, 0, shape[0])
    d2r = np.clip(d2r, 0, shape[1])
    d3r = np.clip(d3r, 0, shape[2])

    return data[d1r, d2r, d3r]