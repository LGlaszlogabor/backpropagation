import math
import numpy as np
import matplotlib.pyplot as plt
import random

def gakt(x):
    return 1/(1+math.exp(-1*x))

def gaktdx(x):
    return gakt(x)*(1 - gakt(x))

alpha = 0.1

weight_range_bottom = -2.0101
weight_range_top = 2.01000
#######preparing learning data######################
data_size = 51
dataX = np.zeros(data_size)
dataY = np.zeros(data_size)

for i in range(0, data_size):
    dataX[i] = (-2 + i*4/data_size)
    dataY[i] = dataX[i]**3 - dataX[i]**2 + np.random.normal(0,0.5,1)
    
plt.plot(dataX, dataY, 'ro')
plt.axis([-2, 2, -7, 5])
#######initialize weights##########################
w = np.zeros((7,2))
a = np.zeros(7)
a_value = np.zeros(7)
v = np.zeros((4, 7))
s = np.zeros(4)
s_value = np.zeros(4)
u = np.zeros(4)
for i in range(0,4):
    u[i] = weight_range_bottom + random.random()*(weight_range_top - weight_range_bottom)
for l in range(0, 4):
    for i in range(0,7):
        v[l][i] = weight_range_bottom + random.random()*(weight_range_top - weight_range_bottom)
for k in range(0,7):
    w[k][0] = weight_range_bottom + random.random()*(weight_range_top - weight_range_bottom)
    w[k][1] = weight_range_bottom + random.random()*(weight_range_top - weight_range_bottom)
dErr_du = np.zeros(4)
dErr_ds = np.zeros(4)
dErr_dv = np.zeros((4, 7))
dErr_da = np.zeros((4,7))
dErr_dw = np.empty((7,2))


totalError = 20
iteration = 0
while totalError/data_size > 0.10 and iteration < 1000:
    iteration = iteration + 1
    totalError = 0
    for index in range(0,data_size):
        #####feedforward#####################################
        x = dataX[index]
        y = dataY[index]
        
        #first layer
        for k in range(1,7):
            a_value[k] = w[k][0] + w[k][1]*x
            a[k] = gakt(a_value[k])
        #second layer
        for l in range(1, 4):
            s_value[l] = v[l][0] + v[l][1]*a[1] + v[l][2]*a[2] + v[l][3]*a[3] + v[l][4]*a[4] + v[l][5]*a[5] + v[l][6]*a[6]
            s[l] = gakt(s_value[l]) 
        #output layer
        fx = u[0] + u[1]*s[1] + u[2]*s[2] + u[3]*s[3]    
        #####backpropagation##############################
        outputError = y-fx
        totalError = totalError + abs(outputError)
        for i in range(0,4): #output layer parameter derivative
            dErr_du[i] = -1*outputError*s[i]    
        for i in range(1,4):
            dErr_ds[i] = -1*outputError*u[i]     
        for l in range(1, 4): #second layer parameter derivative
            for i in range(0,7):
                dErr_dv[l][i]= dErr_ds[l] * gaktdx(s_value[l])*a[i]
        for l in range(1, 4):
            for i in range(1,7):
                dErr_da[l][i]= dErr_ds[l] * gaktdx(s_value[l])*v[l][i]
        for k in range(1,7): #first layer parameter derivative
            dErr_dw[k][0] = dErr_da[2][k]*gaktdx(a_value[k])
            dErr_dw[k][1] = dErr_da[2][k]*gaktdx(a_value[k]) * x
        #substracting error contribution from weights
        for i in range(0,4):
            u[i] = u[i] - alpha*dErr_du[i]
        for l in range(1, 4):
            for i in range(0,7):
                v[l][i]= v[l][i] - alpha * dErr_dv[l][i]
        for k in range(1,7):
            w[k][0] = w[k][0] - alpha*dErr_dw[k][0]
            w[k][1] = w[k][1] - alpha*dErr_dw[k][1]    
    print('{}{}'.format("Average error per data points:", totalError/data_size))
##############evaluating network###################################  
plot_size = 100
plotX = np.zeros(plot_size)
plotY = np.zeros(plot_size)

for i in range(0, plot_size):
    plotX[i] = (-2 + i*4/plot_size)
    x = (-2 + i*4/plot_size)
    #first layer
    for k in range(1,7):
        a_value[k] = w[k][0] + w[k][1]*x
        a[k] = gakt(a_value[k])
    
    #second layer
    for l in range(1, 4):
        s_value[l] = v[l][0] + v[l][1]*a[1] + v[l][2]*a[2] + v[l][3]*a[3] + v[l][4]*a[4] + v[l][5]*a[5] + v[l][6]*a[6]
        s[l] = gakt(s_value[l]) 
    #output layer
    #print(s)
    fx = u[0] + u[1]*s[1] + u[2]*s[2] + u[3]*s[3]        
    plotY[i] = fx
plt.plot(plotX, plotY)
plt.show()