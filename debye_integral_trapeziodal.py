import numpy
from scipy.integrate import nquad
from scipy.special import jn
from numpy import exp,pi,sin,cos,sqrt,arctan2
def debyr_integral_vector(point,alpha,lam,f,E_x,E_y):
    

    
    def debyr_integral(x,y,z,alpha,lam,f,E_x,E_y): 
        
        kr = 2*pi/lam*sqrt(x*x+y*y)
        kz = 2*pi/lam*z
        phi = arctan2(y,x)
        
        def trapezoidal_double(f, a, b, c, d, nx, ny):
            hx = (b - a)/float(nx)
            hy = (d - c)/float(ny)
            I = 0.25*(f(a, c) + f(a, d) + f(b, c) + f(b, d))
            Ix = 0
            for i in range(1, nx):
                xi = a + i*hx
                Ix += f(xi, c) + f(xi, d)
            I += 0.5*Ix
            Iy = 0
            for j in range(1, ny):
                yj = c + j*hy
                Iy += f(a, yj) + f(b, yj)
            I += 0.5*Iy
            Ixy = 0
            for i in range(1, nx):
                for j in range(1, ny):
                    xi = a + i*hx
                    yj = c + j*hy
                    Ixy += f(xi, yj)
            I += Ixy
            I *= hx*hy
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
        
        Ex = trapezoidal_double(lambda theta, psi: e(theta,psi)[0], 0, alpha, 0, 2*pi, 50, 50)
        Ey = trapezoidal_double(lambda theta, psi: e(theta,psi)[1], 0, alpha, 0, 2*pi, 50, 50)
        Ez = trapezoidal_double(lambda theta, psi: e(theta,psi)[2], 0, alpha, 0, 2*pi, 50, 50)
    
        
        return (Ex,Ey,Ez)
    debyr_integral_v = numpy.vectorize(debyr_integral)
    p = numpy.transpose(point)
    return numpy.transpose(debyr_integral_v(p[0],p[1],p[2],alpha,lam,f,E_x,E_y))
point = [[1,2,3],[2,2,3],[3,2,3],[4,2,3]]
debyr_integral_vector(point,0.5,4,2,1,0)
