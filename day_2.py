import numpy
import math
import matplotlib.pyplot as plt

l = 1
theta1 = 10
theta2 = 10
goal = numpy.array([[1],[1]])
beta = 0.5

def endeff(th1,th2,len):
    cth1 = numpy.cos(th1*(math.pi/180))
    sth1 = numpy.sin(th1*(math.pi/180))
    cths = numpy.cos((th1+th2)*(math.pi/180))
    sths = numpy.sin((th1+th2)*(math.pi/180))
    e = numpy.array([[(len*cth1+len*cths)],[(len*sth1+len*sths)]])
    return e

def jacobian(th1,th2,len):
    sth1 = numpy.sin(th1*(math.pi/180))
    cth1 = numpy.cos(th1*(math.pi/180))
    cths = numpy.cos((th1+th2)*(math.pi/180))
    sths = numpy.sin((th1+th2)*(math.pi/180))
    j = numpy.array([[-1*len*sth1,-1*len*sths],[len*cth1,len*cths]])
    return j

def inverse(jac):
    
    j_    = numpy.linalg.inv(jac)
    return j_

end   = endeff(theta1,theta2,l)
cap_e = goal - end
i = 0

while (cap_e[0][0]>0.1) or (cap_e[1][0]>0.1):
    jacob  = jacobian(theta1,theta2,l)
    j_inv  = inverse(jacob)
    print(jacob)
    print(jacob.shape)
    del_e  = beta*cap_e
    del_th = j_inv.dot(del_e)
    theta1 = theta1 + del_th[0]
    theta2 = theta2 + del_th[1]
    end   = endeff(theta1,theta2,l)
    cap_e = goal - end
    i = i+1
    

print(end)
print(i)