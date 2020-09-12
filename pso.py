import numpy as np
import pandas as pd
import random
import math as m
size=int(input("enter the swarm size\n")) #declares the size
iterations=int(input("enter the number of iterations\n")) #declares no of iterations
c=[0,0]
c[0]=float(input("enter the value of c1\n"))
c[1]=float(input("enter the value of c2\n"))
particles=np.random.rand(size,10) #declaring a particle swarm in each row will represent each particle and each column represent respective weights
particles=(2*particles-1)*10 #making this operation so the weights are bw -1 to 1
v=np.zeros((size,10)) #initalising the velocities to zero


f=open("DS1_input.txt","r")
time=f.readlines()  #reading time from the file
for i in range(len(time)): #converting the time values from strings to float
    time[i]=float(time[i])
g=open("DS1_output.txt","r") #reading actual output from the file
ce=g.readlines()  #ce stands for cummulative error
for j in range(len(ce)):    #converting the string of data into integers
    ce[j]=float(ce[j])

def normalise(x): #function to normalise the values into values in between 0 and 1
    ma=max(x)
    mi=min(x)
    for i in range(len(x)):
        x[i]=(x[i]-mi)/(ma-mi)

normalise(time) #normalising the time values of the data set given
normalise(ce) #normalising the out put values


def cost(t,ce,w):   #defination of cost function
    '''takes time list actual error list and expected weights as input in the given order and returns error as output'''
    cost=0
    outs=0
    for i in range(len(t)):
        try:
            out=round(w[4]*(1-m.exp(-w[0]*t[i]))+w[5]*(1-(1+w[1]*t[i])*(m.exp(-w[1]*t[i])))+(w[6]*(1-(m.exp(-w[2]*t[i]))))/(1+(w[8]*(m.exp(w[2]*t[i]))))+w[7]/(1+w[9]*(m.exp(-w[3]*t[i]))),2)
        except OverflowError:
            out = float('inf')
        cost=cost+((out-ce[i])**2)
        outs+=out**2
    cost=m.sqrt(cost/outs)
    return cost

p_best=particles.copy() #creating pbest array each row defining personal best of each particle since initial position is initial personal best position

g_best=particles[0].copy()  #out of randomly genreated weights looking for global best
for j in range(size):
    if(cost(time,ce,g_best)>cost(time,ce,p_best[j])):
        g_best=p_best[j].copy()
w=float(input("enter the value of w\n")) #initalising the inertia factor value

for iter in range(iterations):
    for j in range(size):
        r=np.random.rand(1,2)
        v[j]=w*v[j]-c[0]*(r[0,0])*(particles[j]-p_best[j])-c[1]*(r[0,1])*(particles[j]-g_best) #updating velocity first
        particles[j]=particles[j]+v[j] #updating position of particles
        #if(particles[j,8]>1):
         #   particles[j,8]=1
          #  v[j,8]=0
        #if(particles[j,9]>1):
         #   particles[j,9]=1
          #  v[j,9]=0
        #if(particles[j,9]<0):
         #   particles[j,9]=0
          #  v[j,9]=0
        #if(particles[j,8]<0):
         #   particles[j,8]=0
          #  v[j,8]=0
        if(cost(time,ce,particles[j])<cost(time,ce,p_best[j])): #updating their personal bests
            p_best[j]=particles[j].copy()
        if(cost(time,ce,p_best[j])<cost(time,ce,g_best)): #updating the global bests
            g_best=p_best[j].copy()

    w=w-((0.5)/(size-1))*iter #updation of inertia factor after each and every itertation


G_best = tuple((g_best))
print(g_best)

def avg_error(t,ce,w):
    ae=0
    for i in range(2,len(t)):
        try:
            out=w[4]*(1-m.exp(-w[0]*t[i]))+w[5]*(1-(1+w[1]*t[i])*(m.exp(-w[1]*t[i])))+(w[6]*(1-(m.exp(-w[2]*t[i]))))/(1+(w[8]*(m.exp(w[2]*t[i]))))+w[7]/(1+w[9]*(m.exp(-w[3]*t[i])))
        except OverflowError:
            out = float('inf')
        re=(out-ce[i])/(ce[i])
        re=abs(re)*100
        ae+=re

    ae=(ae/(len(t)))
    return ae
print("the average error of the fitting performance is\n")

print(avg_error(time,ce,g_best)/100)

# to predict the prediction power calculating the average error of the predicted values

ope=open("test1.txt","r")
test_time=ope.readlines()
for i in range(len(test_time)):
    test_time[i]=float(test_time[i])
ope2=open("test1o.txt","r")
test_ce=ope2.readlines()
for i in range(len(test_time)): #converting the time values from strings to float
    test_ce[i]=float(test_ce[i])

print(avg_error(test_time,test_ce,g_best)/100)

# to predict the end point prediction
def relative_error(t1,ce1,w):
    out=w[4]*(1-m.exp(-w[0]*t1))+w[5]*(1-(1+w[1]*t1)*(m.exp(-w[1]*t1)))+(w[6]*(1-(m.exp(-w[2]*t1))))/(1+(w[8]*(m.exp(w[2]*t1))))+w[7]/(1+w[9]*(m.exp(-w[3]*t1)))
    re=(out-ce1)/(ce1)
    re=abs(re)*100
    return re

print(relative_error(test_time[0],test_ce[0],g_best)/100)
