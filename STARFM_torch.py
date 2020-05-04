# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 15:15:36 2020

@author: Administrator
"""


import numpy as np
import torch
import torch.nn as nn
import time
#import skimage.measure as  sm
import skimage.metrics  as  sm
import cv2
from osgeo import gdal
import matplotlib.pyplot as plt

###img read tool###############################################################
def imgread(file,mode='gdal'):
    if mode=='cv2':
        img=cv2.imread(file,-1)/10000.
    if mode=='gdal':
        img=gdal.Open(file).ReadAsArray()/10000.
    return img

###weight caculate tools######################################################
def weight_caculate(data):
    return  torch.log((abs(data)*10000+1.00001))

def caculate_weight(l1m1,m1m2):
    #atmos difference
    wl1m1=weight_caculate(l1m1 )
    #time deference
    wm1m2=weight_caculate(m1m2 )
    return  wl1m1*wm1m2

###space distance caculate tool################################################
def indexdistance(window):
    #one window, one distance weight matrix
    [distx,disty]=np.meshgrid(np.arange(window[0]),np.arange(window[1]))
    centerlocx,centerlocy=(window[0]-1)//2,(window[1]-1)//2
    dist=1+(((distx-centerlocx)**2+(disty-centerlocy)**2)**0.5)/((window[0]-1)//2)
    return  dist

###threshold select tool######################################################
def weight_bythreshold(weight,data,threshold):
    #make weight tensor
    weight[data<=threshold]=1
    return  weight

###initial similar pixels tools################################################
def spectral_similar_threshold(clusters,NIR,red):
    thresholdNIR=NIR.std()*2//clusters
    thresholdred=red.std()*2//clusters
    return  (thresholdNIR,thresholdred)  

def caculate_similar(l1,threshold,window):
    #read l1
    l1=torch.from_numpy(l1.reshape(1,1,l1.shape[0],l1.shape[1]))
    device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
    l1=l1.to(device)
    l1=nn.functional.unfold(l1,window)    
    #caculate similar
    weight=torch.zeros(l1.shape).to(device)  
    centerloc=( l1.size()[1]-1)//2
    weight=weight_bythreshold(weight,abs(l1-l1[:,centerloc:centerloc+1,:]) ,threshold)
    return weight

def classifier(l1):
    '''not used'''
    return

###similar pixels filter tools#################################################
def allband_arrayindex(array,indexarray,rawindexshape):
    shape=array.shape
    newarray=np.zeros(rawindexshape)
    for band in range(shape[0]):
        newarray[band]=array[band][indexarray]
    return newarray

def similar_filter(l1,m1,m2,sital,sitam):
    shape=l1.shape
    l1=torch.from_numpy(l1.reshape(1,shape[0],shape[1],shape[2]))
    m1=torch.from_numpy(m1.reshape(1,shape[0],shape[1],shape[2]))
    m2=torch.from_numpy(m2.reshape(1,shape[0],shape[1],shape[2]))
    device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
    l1=l1.to(device)
    m1=m1.to(device)
    m2=m2.to(device)
    l1m1=abs(l1-m1)
    m1m2=abs(m2-m1)
    #####
    l1m1=nn.functional.unfold(l1m1,(1,1)).max(1)[0]+(sital**2+sitam**2)**0.5
    m1m2=nn.functional.unfold(m1m2,(1,1)).max(1)[0]+(sitam**2+sitam**2)**0.5
    return (l1m1,m1m2)

###starfm for onepart##########################################################
def starfm_onepart(l1,m1,m2,similar,thresholdmax,window,outshape):
    #####pytorch GPU mode 
    outshape=outshape
    shape=l1.shape
    l1=torch.from_numpy(l1.reshape(1,1,shape[0],shape[1]))
    m1=torch.from_numpy(m1.reshape(1,1,shape[0],shape[1]))
    m2=torch.from_numpy(m2.reshape(1,1,shape[0],shape[1]))
    device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
    l1=l1.to(device)
    m1=m1.to(device)
    m2=m2.to(device)
    
    #####img to col
    l1=nn.functional.unfold(l1,window)
    m1=nn.functional.unfold(m1,window)
    m2=nn.functional.unfold(m2,window)
    l1m1=abs(l1-m1)
    m1m2=abs(m2-m1)
    #####caculate weights
    #distance weight
    dist=nn.functional.unfold(torch.from_numpy(indexdistance(window)).reshape(1,1,window[0],window[1]),window).to(device)
    #time and space weight
    w=caculate_weight(l1m1,m1m2)
    w=1/(w*dist)
    #similar pixels: 1:by threshold 2:by classifier
    wmask=torch.zeros(l1.shape).to(device)  
    wmask=weight_bythreshold(wmask,l1m1,thresholdmax[0]) 
    wmask=weight_bythreshold(wmask,m1m2,thresholdmax[1])
    #mask
    w=w*wmask*similar
    #normili
    w=w/(w.sum(1))
    #####predicte and trans
    #predicte l2
    l2=(l1+m2-m1)*w
    l2=l2.sum(1).reshape(1,1,l2.shape[2])
    #col to img
    l2=nn.functional.fold(l2,outshape,(1,1))
    #tensor to numpy
    if device.type=='cuda':
        l2_numpy=l2[0,0,:,:].cpu().numpy()
    else:
        l2_numpy=l2[0,0,:,:].numpy()
    return l2_numpy
###starfm for allpart#########################################################
def starfm_main(l1r,m1r,m2r,param):
    #get start time
    time_start=time.time()
    #read parameters
    parts_shape=param['part_shape']
    window=param['window_size']
    clusters=param['clusters']
    NIRindex=param['NIRindex']
    redindex=param['redindex']
    sital=param['sital']
    sitam=param['sitam']
    #caculate initial similar pixels threshold
    threshold=spectral_similar_threshold(clusters,l1r[NIRindex],l1r[redindex])    
    ####shape
    imageshape=l1r.shape
    row=imageshape[1]//parts_shape[0]+1
    col=imageshape[2]//parts_shape[1]+1
    padrow=window[0]//2
    padcol=window[1]//2
    #get output data
    l2_fake=np.zeros(imageshape)   
    #####padding constant for conv;STARFM use Inverse distance weight(1/w),better to avoid 0 and NAN(1/0),or you can use another distance measure
    constant1=10
    constant2=20
    constant3=30
    l1=np.pad( l1r,((0,0),(padrow,padcol),(padrow,padcol)),'constant', constant_values=(constant1,constant1))
    m1=np.pad( m1r,((0,0),(padrow,padcol),(padrow,padcol)),'constant', constant_values=(constant2,constant2))
    m2=np.pad( m2r,((0,0),(padrow,padcol),(padrow,padcol)),'constant', constant_values=(constant3,constant3))
    #split parts , get index and  run for every part
    row_part=np.array_split( np.arange(imageshape[1]), row , axis = 0) 
    col_part=np.array_split( np.arange(imageshape[2]),  col, axis = 0)  
    print('Split into {} parts,row number: {},col number: {}'.format(len(row_part)*len(row_part),len(row_part),len(row_part)))
    
    for rnumber,row_index in enumerate(row_part):
        for cnumber,col_index in enumerate(col_part):
            ####run for part: (rnumber,cnumber)
            print('now for part{}'.format((rnumber,cnumber)))
            ####output index
            rawindex=np.meshgrid(row_index,col_index)
            ####output shape
            rawindexshape=(col_index.shape[0],row_index.shape[0])
            ####the real parts_index ,for reading the padded data 
            row_pad=np.arange(row_index[0],row_index[len(row_index)-1]+window[0])
            col_pad=np.arange(col_index[0],col_index[len(col_index)-1]+window[1])    
            padindex=np.meshgrid(row_pad,col_pad)
            ####caculate initial similar pixels
            NIR_similar=caculate_similar(l1[NIRindex][ padindex ],threshold[0],window)   
            red_similar=caculate_similar(l1[redindex][ padindex ],threshold[1],window)  
            similar=NIR_similar*red_similar       
            ####caculate threshold used for similar_pixels_filter  
            thresholdmax=similar_filter( allband_arrayindex(l1r,rawindex,(imageshape[0],rawindexshape[0],rawindexshape[1])),
                                        allband_arrayindex(m1r,rawindex,(imageshape[0],rawindexshape[0],rawindexshape[1])) , 
                                        allband_arrayindex(m2r,rawindex,(imageshape[0],rawindexshape[0],rawindexshape[1])),sital,sitam)
            ####run for each band
            for band in range(imageshape[0]):    
                l2_fake[band][rawindex ]=starfm_onepart(l1[band][ padindex ],m1[band][padindex ],m2[band][ padindex ],similar,thresholdmax,window,rawindexshape)
    #time cost
    time_end=time.time()    
    print('now over,use time {:.4f}'.format(time_end-time_start))  
    return l2_fake


if __name__ == "__main__":
    ##three band datas(sorry,just find them at home,i cant recognise the spectral response range of each band,'NIR' and 'red' are only examples)
    l1file='L72000306_SZ_B432_30m.tif'
    l2file='L72002311_SZ_B432_30m.tif'
    m1file='MOD09_2000306_SZ_B214_250m.tif'
    m2file='MOD09_2002311_SZ_B214_250m.tif'
    ##param
    param={'part_shape':(100,100),
           'window_size':(31,31),
           'clusters':5,
           'NIRindex':1,'redindex':0,
           'sital':0.001,'sitam':0.001}
    #read images from files
    l1=imgread(l1file)
    m1=imgread(m1file)
    m2=imgread(m2file)
    l2_gt=imgread(l2file)    
    ##predicte
    l2_fake=starfm_main(l1,m1,m2,param)
    
    ##show results 
    #transform:(chanel,H,W) to (H,W,chanel)
    l2_fake=l2_fake.transpose(1,2,0)
    l2_gt=l2_gt.transpose(1,2,0)
    l1=l1.transpose(1,2,0)
    m1=m1.transpose(1,2,0)
    m2=m2.transpose(1,2,0)
    #plot
    plt.figure('landsat:t1')
    plt.imshow(l1)    
    plt.figure('landsat:t2_fake')
    plt.imshow(l2_fake)
    plt.figure('landsat:t2_groundtrue')
    plt.imshow(l2_gt)    
    
    ##evaluation
    ssim1=sm.structural_similarity(l2_fake,l2_gt,data_range=1,multichannel=True)
    ssim2=sm.structural_similarity(l1,l2_gt,data_range=1,multichannel=True)
    ssim3=sm.structural_similarity(l1+m2-m1,l2_gt,data_range=1,multichannel=True)
    print('with-similarpixels ssim: {:.4f};landsat_t1 ssim: {:.4f};non-similarpixels ssim: {:.4f}'.format(ssim1,ssim2,ssim3))

    
    
    
    
    
    
    
    
    
    
    
