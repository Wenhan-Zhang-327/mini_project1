import numpy as np
from matplotlib import pyplot as plt
# from scipy.integrate import nquad
# from scipy.special import jn
from numpy import exp,pi,sin,cos,sqrt,arctan2     
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
    lims = np.linspace(-1,1/2,w)
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
def debyr_integral(beam,alpha,lam,f,E_x,E_y): 
    lims = np.linspace(-1,1,512)
    x, y = np.meshgrid(lims,lims)
    z = f
    k = 2*pi/lam
    r = sqrt(x**2+y**2)
    phi = arctan2(y,x)
    intensity = np.zeros([512,512])
    

    def midpoint_double1(f, a, b, c, d, nx, ny):
        hx = (b - a)/float(nx)
        hy = (d - c)/float(ny)
        I = 0
        for i in range(nx):
            for j in range(ny):
                xi = a + hx/2 + i*hx
                yj = c + hy/2 + j*hy
                I += hx*hy*f(xi, yj)
        return I
        
    def e(theta,psi): 
        q1 = cos(theta) + 1
        q2 = cos(theta) - 1
        q3 = 2*sin(theta)
        q4 = 2*cos(theta)
            
        a_theta = 0.5*sqrt(cos(theta))
            
        ex = a_theta*(E_x*(q1+q2*cos(2*psi))+E_y*q2*sin(2*psi))*(-1j)/lam*f*exp(1j*(k*z*cos(theta)+k*r*sin(theta)*cos(psi-phi)))*beam
        ey = a_theta*(E_y*(q1-q2*cos(2*psi))+E_x*q2*sin(2*psi))*(-1j)/lam*f*exp(1j*(k*z*cos(theta)+k*r*sin(theta)*cos(psi-phi)))*beam
        ez = a_theta*(-q3)*(E_x*cos(psi)+E_y*sin(psi))*(-1j)/lam*f*exp(1j*(k*z*cos(theta)+k*r*sin(theta)*cos(psi-phi)))*beam
        return (ex,ey,ez)
        
    Ex = np.abs(midpoint_double1(lambda theta, psi: e(theta,psi)[0], 0, alpha, 0, 2*pi, 20, 20))
    Ey = np.abs(midpoint_double1(lambda theta, psi: e(theta,psi)[1], 0, alpha, 0, 2*pi, 20, 20))
    Ez = np.abs(midpoint_double1(lambda theta, psi: e(theta,psi)[2], 0, alpha, 0, 2*pi, 20, 20))
    
    intensity += (Ex**2 + Ey**2 + Ez**2)
    return intensity

mask = getPhaseMask(512,0,0,0,0)
mask = addPhaseFeature(mask,'STED',1,0)

pupil = getPupil(512,'Airy',0.8,0.4)

beam = pupil*np.exp(1j*mask)
sample = debyr_integral(beam,0.47,0.68,2,1,1)
plt.figure(figsize=(10, 10))


centre = np.shape(sample)[1]/2
lim_up = int(centre+255)
lim_down = int(centre-256)
plt.imshow(sample[lim_down:lim_up,lim_down:lim_up],cmap = 'gray')
