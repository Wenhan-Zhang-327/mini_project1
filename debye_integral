import numpy
from scipy.integrate import quad
from scipy.special import jn
from numpy import exp,pi,sin,cos,sqrt,arctan2
def debyr_integral(x,y,z,alpha,lam,f,ex,ey):
    
    kr = 2*pi/lam*sqrt(x*x+y*y)
    kz = 2*pi/lam*z
    phi = arctan2(y,x)
    
    def complex_quadrature(func, a, b, **kwargs):
        def real_func(x):
            return numpy.real(func(x))
        def imag_func(x):
            return numpy.imag(func(x))
        real_integral = quad(real_func, a, b, **kwargs)
        imag_integral = quad(imag_func, a, b, **kwargs)
        return real_integral[0] + 1j*imag_integral[0]
    
    def i0(θ):
        return sqrt(cos(θ))*sin(θ)*(cos(θ)+1)*jn(0,kr*sin(θ))*exp(1j*kz*cos(θ))
    def i1(θ):
        return sqrt(cos(θ))*sin(θ)*sin(θ)*jn(1,kr*sin(θ))*exp(1j*kz*cos(θ))
    def i2(θ):
        return sqrt(cos(θ))*sin(θ)*(cos(θ)-1)*jn(2,kr*sin(θ))*exp(1j*kz*cos(θ))
    
    I0 = complex_quadrature(lambda θ:i0(θ), 0, alpha)
    I1 = complex_quadrature(lambda θ:i1(θ), 0, alpha)
    I2 = complex_quadrature(lambda θ:i2(θ), 0, alpha)
    
    Ex = ex*(I0+I2*cos(2*phi))+ey*I2*sin(2*phi)*1j*2*pi/lam*f
    Ey = ey*(I0-I2*cos(2*phi))+ex*I2*sin(2*phi)*1j*2*pi/lam*f
    Ez = 2*I1*ex*cos(phi)+ey*sin(phi)*2*pi/lam*f
    
    return (Ex,Ey,Ez)

debyr_integral(1,2,3,0.5,9,5,1,0)
