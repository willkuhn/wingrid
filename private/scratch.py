# -*- coding: utf-8 -*-
"""
Created on Thu Feb 09 14:55:15 2017

@author: Will
"""

import cv2,os
import numpy as np
import matplotlib.pyplot as plt
import time

os.chdir('D:\Dropbox\dev\github-repos\wingrid')

#%% 
# Grid Recipes
import os,cv2
import numpy as np
os.chdir('D:\Dropbox\dev\github-repos\wingrid')
from wingrid import Grid,Analyze

# Fit a grid to an image and view it
fn = os.listdir('images')[5]
fn = 'WRK-WS-01951D.png'
im = cv2.imread(os.path.join('images',fn))[:,:,::-1]
g = Grid(10,8,use_chrom=True,background='black',max_dim=1000)
g.fit(im).plot_grid(im,use_chrom=True)

# Get features from several images
filenames = os.listdir('images')
g = Grid(12,10,use_chrom=True,background='black',max_dim=1000)
features = np.zeros((len(filenames),g.n_features_),dtype=float)
f_mask = np.zeros((len(filenames),g.n_features_),dtype=bool)

for i in range(23,len(filenames)):
    fn = filenames[::-1][i]
    im = cv2.imread(os.path.join('images',fn))[:,:,::-1]
    if i<30:
        g.fit(im).plot_grid(im,use_chrom=True)

    f,fm = g.fit_sample(im)
    features[i] = f
    f_mask[i] = fm


# Analyze some features with PCA
an = Analyze(features,f_mask)
an.fit_pca()
an.plot(labels,comps=(1,3))

# Analyze some featuers with LDA
an = Analyze(features,f_mask)
an.fit_lda(labels)
an.plot_comps(labels)

# Make loadings bar chart
fn = 'WRK-WS-01922D.png'#'WRK-WS-01951D.png'
im = cv2.imread(os.path.join('images',fn))[:,:,::-1]
g = Grid(12,10,use_chrom=True,background='black',max_dim=1000)
g.fit(im)
an = Analyze(features,f_mask)
an.fit_pca()
an.loadings_plot_bar(g,comp=1)

# Make loadings 2D plot
fn = 'WRK-WS-01922D.png'#'WRK-WS-01951D.png'
im = cv2.imread(os.path.join('images',fn))[:,:,::-1]
g = Grid(12,10,use_chrom=True,background='black',max_dim=1000)
g.fit(im)
an = Analyze(features,f_mask)
an.fit_pca()
an.loadings_plot_2d(g,comps=(1,2),n_highest=10)

# Make loadings/image plot
fn = 'WRK-WS-01922D.png'#'WRK-WS-01951D.png'
im = cv2.imread(os.path.join('images',fn))[:,:,::-1]
g = Grid(12,10,use_chrom=True,background='black',max_dim=1000)
g.fit(im)
an = Analyze(features,f_mask)
an.fit_pca()
an.loadings_image_overlay(im,g,comps=[1],show_cell_nums=False)
#%% Figure out which images Jack needs to fix:
import os,cv2
import numpy as np
os.chdir('D:\Dropbox\dev\github-repos\wingrid')
from wingrid import Grid

def check(fn):
    try:
        im = cv2.imread(os.path.join('images',fn))[:,:,::-1]
        g = Grid(10,10,background='black')
        g.fit(im);
    except RuntimeError:
        print fn

path = 'D:\Dropbox\Mimicry\Peru Butterflies&Polythore' 
filenames = [i for i in os.listdir(path) if i.endswith('.png')]
filenames = map(lambda i: os.path.join(path,i),filenames)
fix_these1 = map(check,filenames)

path = 'D:\Dropbox\Jack - butterfly mimicry rings\Clearwing & Orange Tip' 
filenames = [i for i in os.listdir(path) if i.endswith('.tif')]
filenames = map(lambda i: os.path.join(path,i),filenames)
fix_these2 = map(check,filenames)

#%%
