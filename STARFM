# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 23:35:07 2020

@author: shx
"""
import math
import numpy as np
from osgeo import gdal
from skimage.measure import compare_psnr
from skimage.measure import compare_ssim
from skimage.measure  import compare_nrmse
from skimage.measure  import compare_mse
import matplotlib.pyplot as plt
import numba as nb
import copy

##coordination distance
@nb.vectorize()
def xydistance(x,y,xc,yc):
    d=1+np.sqrt(   math.pow(   x-xc ,2)   +      math.pow(  y-yc  ,2)        )/(window//2)
    return d
###because there are only one windows,we only  caculate one distance matrix 
def windowdistance(w):
    x=np.arange(w)[:,None]+np.zeros(shape=w)[None,:]
    y=np.arange(w)[None,:]+np.zeros(shape=w)[:,None]
    xc=np.ones(shape=(w,w))*(w//2)
    yc=np.ones(shape=(w,w))*(w//2)
    d=xydistance(x,y,xc,yc)
    return d.reshape(w*w)
###get similar mask-matrix
@nb.vectorize()
def similar(l0,d):
    if abs(l0)<=d: 
        kk=1
    else:
        kk=0
    return kk
###make pixels' indexs of the x-th window
def makeindexs(x):
    return (np.arange(x,x+window)[:,None]+long*np.arange(window)[None,:]).T.reshape(window*window) 

####get all center's windowindexs
@nb.jit()
def loopindex(number1,number2,size):
    ll=[]
    t0=int(number1//size)
    ti=int(number2//size)
    r0=number1%size
    r1=number2%size
    numindex=[]
    #numindex=list(range(t0*size+t0*(window-1),t0*size+t0*(window-1)+size-r0))
    for i in range(t0,ti):
        numindex=numindex+list( range(i*size+i*(window-1),(i+1)*size+i*(window-1)) ) 
    #numindex=numindex+list(range((ti)*size+(ti-1)*(window-1),(ti)*size+(ti-1)*(window-1)+r1))
    for i in numindex:
        ll.append(makeindexs(i))
    return np.array(ll)

###### padding
def makeadd(array,number):
    up=array.shape[0]
    right=array.shape[1]
    a1=np.ones(shape=(up+window-1,right+window-1) )*number
    a1[window//2:window//2+up,window//2:window//2+right]=copy.deepcopy(array)   
    return a1
#####make divide parts
def makedivid(numbers,times,size):
    d1=int(numbers//times)
    d2=(d1//size)*size
    left=[]
    right=[]
    for i in range(times-1):
        left.append(i*d2)
        right.append((i+1)*d2)
    left.append((times-1)*d2)
    right.append(numbers)    
    return left,right

#### main starfm
def starfm(landsat0, modis0,modis1,w,times):
    #window size(window=w,easy for writing)
    #image shape
    right=landsat0.shape[1]
    up=landsat0.shape[0]
    #padding:extend image to get the same window pixels for border center-pixels
    newup=up+w-1
    newright=right+w-1
    ####just choosesome constant making the padding pixels dont meet the similar-pixles-condition.if you dont like it,defining a binary mask-matrix is esrier
    l0=makeadd(landsat0,-50000).reshape(newup*newright)
    m0=makeadd(modis0,20000).reshape(newup*newright)
    m1=makeadd(modis1,160000).reshape(newup*newright)
    #####divide several parts if you dont have enough memory
    left,one=makedivid(up*right,times,up)
    d=[0.1,0.1,0.1]
    l1=dividstarfm(one[0],left[0],l0,m0,m1,right,up,newright,newup,w,d)
    if times>1:
        for i in range(1,times):
            l1=np.hstack((l1,dividstarfm(one[i],left[i],l0,m0,m1,right,up,newright,newup,w,d)))
            
    ##reconstruct image
    if l1.shape[0]!=up*right:
        print('error')
    else:
        landsat1=l1.reshape(up,right)  
    return landsat1


######part starfm
def dividstarfm(one,left,l0,m0,m1,right,up,newright,newup,w,d):
    '''
    always delete vars, even it is python 
    '''
    print('get windowindexs')
    ##get windowindexs
    windowindex=loopindex(left,one,up)    
    
    
    print('define window features')
    ##define window features
    f1=(abs(m1-m0))[windowindex]
    f2=(abs(l0-m0))[windowindex]
    f3=(l0)[windowindex]
    
    
    print('define center features')
    ##define center features
    l0r=landsat0.reshape(up*right)[left:one]
    m0r=modis0.reshape(up*right)[left:one]
    m1r=modis1.reshape(up*right)[left:one]
    c1=(abs(m1r-m0r))[:,None]+np.zeros(shape=w*w)[None,:]
    c2=(abs(l0r-m0r))[:,None]+np.zeros(shape=w*w)[None,:]
    c3=l0r[:,None]+np.zeros(shape=w*w)[None,:]
    
    
    print('select similar pixels')
    ##select similar pixels
    d1=d[0]
    d2=d[1]
    d3=d[2]
    mask0=similar(f1-c1,d1*c1)
    mask1=similar(f2-c2,d2*c2)
    mask2=similar(f3-c3,d3*c3)
    mask=mask0*mask1*mask2
    del mask1,mask2,mask0
    del c1,c2,c3
    
    
    print('caculate weights')
    ##caculate weights 
    f12=np.log(abs( f1)+2)
    f22=np.log(abs( f2)+2)
    del f1,f2
    distance=windowdistance(w)
    catt=f12*f22*distance
    weight= 1/catt
    del f12,f22,distance
    weight=weight*mask
    data=((l0+m1-m0)[windowindex])*mask
    del mask,f3
    normweight=weight/np.sum(weight,axis=1).reshape(weight.shape[0],1)
    
    
    print('caculate aim landsat')
    ##caculate aim landsat
    l1=np.sum(np.multiply(data,normweight),axis=1)
    return l1


##read data(after pre-step:same projection,same .....),if dont,please use gdal,py6s..... 
landsatfir='L72000306_SZ_B432_30m.tif'
modisfir='MOD09_2000306_SZ_B214_250m.tif'
aimlandsatfir='L72002311_SZ_B432_30m.tif'
aimmodisfir='MOD09_2002311_SZ_B214_250m.tif'

landsat0=gdal.Open(landsatfir).ReadAsArray()[0]
modis0=gdal.Open(modisfir).ReadAsArray()[0] 
landsat1=gdal.Open(aimlandsatfir).ReadAsArray()[0] 
modis1=gdal.Open(aimmodisfir).ReadAsArray()[0] 

##define some global constant for better useing nb.vectorize() acc speed
global long,window
window=49
long=landsat0.shape[1]+window-1


##use starfm fusion landsat and modis
l1=starfm(landsat0, modis0,modis1,window,3)


##plot image
plt.figure('starfm based landsat1')
plt.imshow(l1.reshape(landsat0.shape)/10000,cmap=plt.cm.gray)

plt.figure('real landsat1')
plt.imshow(landsat1/10000,cmap=plt.cm.gray)
