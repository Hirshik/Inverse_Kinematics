import numpy
import math
import matplotlib.pyplot as plt
import matplotlib.animation as ani

l = 1
theta1 = 10*(math.pi/180)
theta2 = 10*(math.pi/180)
o1 = theta1
o2 = theta2
goal = numpy.array([1,1])
beta = 0.1

def endeff(th1,th2,len):
    cth1 = numpy.cos(th1)
    sth1 = numpy.sin(th1)
    cths = numpy.cos((th1+th2))
    sths = numpy.sin((th1+th2))
    e = numpy.array([(len*cth1+len*cths),(len*sth1+len*sths)])
    return e

def jacobian(th1,th2,len):
    sth1 = numpy.sin(th1)
    cth1 = numpy.cos(th1)
    cths = numpy.cos((th1+th2))
    sths = numpy.sin((th1+th2))
    j = numpy.array([[-1*len*sth1,-1*len*sths],[len*cth1,len*cths]])
    return j

def inverse(jac):
    j_    = numpy.linalg.inv(jac)
    return j_

def mag(vect):
    d = (vect[0]**2 + vect[1]**2)**0.5
    distance = numpy.array([d])
    return distance

end   = endeff(theta1,theta2,l)
cap_e = goal - end
dist = mag(cap_e)
i = 0
path = end

while (cap_e[0]>0.001) or (cap_e[1]>0.001):
    jacob  = jacobian(theta1,theta2,l)
    j_inv  = inverse(jacob)
    del_e  = beta*cap_e
    del_th = j_inv.dot(del_e.transpose())
    theta1 = theta1 + del_th[0]
    theta2 = theta2 + del_th[1]
    end   = endeff(theta1,theta2,l)
    cap_e = goal - end
    dist  = numpy.concatenate((dist,mag(cap_e)), axis = 0)
    i = i+1
    path = numpy.concatenate((path,end), axis = 0)
    
path_re = path.reshape(i+1,2)

print(i)
print(path_re[-1])
print(theta1)
print(theta2)
print(dist[-1])

x_init = numpy.array([0,l*numpy.cos(o1),path_re[0][0]])
y_init = numpy.array([0,l*numpy.sin(o1),path_re[0][1]])
x_finl = numpy.array([0,l*numpy.cos(theta1),path_re[-1][0]])
y_finl = numpy.array([0,l*numpy.sin(theta1),path_re[-1][1]])

path_t = path_re.transpose()

plt.figure("Arm position and path")
plt.plot(path_t[0],path_t[1],'o',color = 'black', label = 'path')
plt.plot(x_init, y_init, label = 'initial arm position', color = 'red')
plt.plot(x_finl, y_finl, label = 'final arm position', color = 'blue')
plt.xlim([-2,2])
plt.ylim([-2,2])

steps = numpy.arange(i+1)

plt.figure("Distance from goal vs number of steps")
plt.plot(steps,dist)
plt.xlim([0,i+2])
plt.ylim([0,4])

plt.show()