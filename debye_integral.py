import numpy
from scipy.integrate import nquad
from scipy.special import jn
from numpy import exp,pi,sin,cos,sqrt,arctan2
def debyr_integral_vector(point,alpha,lam,f,E_x,E_y):
    
    def debyr_integral(x,y,z,alpha,lam,f,E_x,E_y): 
        
        kr = 2*pi/lam*sqrt(x*x+y*y)
        kz = 2*pi/lam*z
        phi = arctan2(y,x)
        
        def complex_dblquadrature(func, a, b, c, d, **kwargs):
            
            def real_func(x,y):
                return numpy.real(func(x,y))
            def imag_func(x,y):
                return numpy.imag(func(x,y))
            def bounds_x(y):
                return [a, b]
            def bounds_y():
                return [c, d]
            
            real_integral = nquad(real_func, [bounds_x, bounds_y], **kwargs)
            imag_integral = nquad(imag_func, [bounds_x, bounds_y], **kwargs)
            
            return real_integral[0] + 1j*imag_integral[0]
        
        def e(theta,psi): 
            q1 = cos(theta) + 1
            q2 = cos(theta) - 1
            q3 = 2*sin(theta)
            q4 = 2*cos(theta)
            
            a_theta = 0.5*sqrt(cos(theta))
            
            ex = a_theta*(E_x*(q1+q2*cos(2*psi))+E_y*q2*sin(2*psi))*1j/lam*f*exp(1j*(kz*cos(theta)+kr*sin(theta)*cos(psi-phi)))
            ey = a_theta*(E_y*(q1-q2*cos(2*psi))+E_x*q2*sin(2*psi))*1j/lam*f*exp(1j*(kz*cos(theta)+kr*sin(theta)*cos(psi-phi)))
            ez = a_theta*(-q3)*(E_x*cos(psi)+E_y*sin(psi))*1j/lam*f*exp(1j*(kz*cos(theta)+kr*sin(theta)*cos(psi-phi)))
            
            return (ex,ey,ez)
        
        Ex = complex_dblquadrature(lambda theta, psi: e(theta,psi)[0], 0, alpha, 0, 2*pi)
        Ey = complex_dblquadrature(lambda theta, psi: e(theta,psi)[1], 0, alpha, 0, 2*pi)
        Ez = complex_dblquadrature(lambda theta, psi: e(theta,psi)[2], 0, alpha, 0, 2*pi)
        
        return (Ex,Ey,Ez)
    
    debyr_integral_v = numpy.vectorize(debyr_integral)
    p = numpy.transpose(point)
    
    return numpy.transpose(debyr_integral_v(p[0],p[1],p[2],alpha,lam,f,E_x,E_y))

points = [[1,2,3],[2,2,3],[3,2,3],[4,2,3]]
debyr_integral_vector(points,0.5,4,2,1,0)
