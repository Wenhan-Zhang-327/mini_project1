import numpy
from scipy.integrate import nquad
from scipy.special import jn
from numpy import exp,pi,sin,cos,sqrt,arctan2
def debyr_integral_vector(point,alpha,lam,f,E_x,E_y):
    

    
    def debyr_integral(x,y,z,alpha,lam,f,E_x,E_y): 
        
        kr = 2*pi/lam*sqrt(x*x+y*y)
        kz = 2*pi/lam*z
        phi = arctan2(y,x)
        

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
            
            ex = a_theta*(E_x*(q1+q2*cos(2*psi))+E_y*q2*sin(2*psi))*1j/lam*f*exp(1j*(kz*cos(theta)+kr*sin(theta)*cos(psi-phi)))
            ey = a_theta*(E_y*(q1-q2*cos(2*psi))+E_x*q2*sin(2*psi))*1j/lam*f*exp(1j*(kz*cos(theta)+kr*sin(theta)*cos(psi-phi)))
            ez = a_theta*(-q3)*(E_x*cos(psi)+E_y*sin(psi))*1j/lam*f*exp(1j*(kz*cos(theta)+kr*sin(theta)*cos(psi-phi)))
            return (ex,ey,ez)
        
        Ex = midpoint_double1(lambda theta, psi: e(theta,psi)[0], 0, alpha, 0, 2*pi, 20, 20)
        Ey = midpoint_double1(lambda theta, psi: e(theta,psi)[1], 0, alpha, 0, 2*pi, 20, 20)
        Ez = midpoint_double1(lambda theta, psi: e(theta,psi)[2], 0, alpha, 0, 2*pi, 20, 20)
    
        
        return (Ex,Ey,Ez)
    debyr_integral_v = numpy.vectorize(debyr_integral)
    p = numpy.transpose(point)
    return numpy.transpose(debyr_integral_v(p[0],p[1],p[2],alpha,lam,f,E_x,E_y))
point = [[1,2,3],[2,2,3],[3,2,3],[4,2,3]]
debyr_integral_vector(point,0.5,4,2,1,0)
