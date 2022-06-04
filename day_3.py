import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import linalg
from matplotlib.patches import Ellipse

l = 1
theta1 = 45*(math.pi/180)
theta2 = 45*(math.pi/180)
o1 = theta1
o2 = theta2
goal = np.array([1.5,1])
beta = 0.1
v = 1

def fk(th1,th2,len):
    cth1 = np.cos(th1)
    sth1 = np.sin(th1)
    cths = np.cos((th1+th2))
    sths = np.sin((th1+th2))
    e = np.array([(len*cth1+len*cths),(len*sth1+len*sths)])
    return e

def jacobian(th1,th2,len):
    sth1 = np.sin(th1)
    cth1 = np.cos(th1)
    cths = np.cos((th1+th2))
    sths = np.sin((th1+th2))
    j = np.array([[-1*len*sth1,-1*len*sths],[len*cth1,len*cths]])
    return j

def inverse(jac):
    j_    = np.linalg.inv(jac)
    return j_

def mag(vect):
    d = (vect[0]**2 + vect[1]**2)**0.5
    distance = np.array([d])
    return distance

end   = fk(theta1,theta2,l)
cap_e = goal - end
dist = mag(cap_e)
i = 0
path = end
print(end)
while (cap_e[0]>0.001) or (cap_e[1]>0.001):
    jacob  = jacobian(theta1,theta2,l)
    j_inv  = inverse(jacob)
    cap_v  = (cap_e * v) / np.linalg.norm(cap_e)
    omega  = j_inv.dot(cap_v.transpose())
    cap_t  = cap_e[0] / cap_v[0]
    del_t  = beta*cap_t
    del_th = omega * del_t
    theta1 = theta1 + del_th[0]
    theta2 = theta2 + del_th[1]
    end   = fk(theta1,theta2,l)
    cap_e = goal - end
    dist  = np.concatenate((dist,mag(cap_e)), axis = 0)
    i = i+1
    path = np.concatenate((path,end), axis = 0)
    #print(path)

path_re = path.reshape(i+1,2)

print(i)
print(path_re[-1])
print(theta1)
print(theta2)
print(dist[-1])

x_init = np.array([0,l*np.cos(o1),path_re[0][0]])
y_init = np.array([0,l*np.sin(o1),path_re[0][1]])
x_finl = np.array([0,l*np.cos(theta1),path_re[-1][0]])
y_finl = np.array([0,l*np.sin(theta1),path_re[-1][1]])

V_init = jacobian(o1,o2,l)
ve_ini = V_init.dot(V_init.transpose())
eig_in = linalg.eigvals(ve_ini)
V_finl = jacobian(theta1,theta2,l)
ve_fin = V_finl.dot(V_finl.transpose())
eig_fi = linalg.eigvals(ve_fin)

path_t = path_re.transpose()
print(path_t)
plt.figure("Arm position and path")
ellipse_init = Ellipse((path_re[0][0], path_re[0][1]), eig_in[0]**0.5, eig_in[1]**0.5, angle=((o1+o2)*(180/math.pi)), alpha=0.5, color = 'red')
plt.gca().add_patch(ellipse_init)
ellipse_finl = Ellipse((path_re[-1][0], path_re[-1][1]), eig_fi[0]**0.5, eig_fi[1]**0.5, angle=((theta1+theta2)*(180/math.pi)), alpha=0.5, color = 'blue')
plt.gca().add_patch(ellipse_finl)
plt.plot(path_t[0],path_t[1],'o',color = 'black', label = 'path')
plt.plot(x_init, y_init, label = 'initial arm position', color = 'red')
plt.plot(x_finl, y_finl, label = 'final arm position', color = 'blue')
plt.xlim([-2,3])
plt.ylim([-2,3])

steps = np.arange(i+1)

plt.figure("Distance from goal vs number of steps")
plt.plot(steps,dist)
plt.xlim([0,i+2])
plt.ylim([0,4])

plt.show()