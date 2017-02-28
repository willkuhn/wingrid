# -*- coding: utf-8 -*-

"""Helpful examples of Wingrid usage"""
### Install package
git clone https://...wingrid
cd wingrid
python setup.py


### Import the package
import os,cv2
import numpy as np
os.chdir('D:\Dropbox\dev\github-repos\wingrid')
from wingrid import Grid,Analyze


### Fit a 10x10 grid to an image with black background and display it
# get the image
fn = '681.png'
im = cv2.imread(os.path.join('images',fn))

# reverse order of image's color channels
# (cv2 imports BGR but we need RGB)
im = im[:,:,::-1]

# set up and fit grid
g = Grid(10,10,background='black')
g.fit_sample(im);

# plot the fitted grid onto the image
g.plot_grid(image=im,
            show_gridlines=True,   # plot gridlines
            show_cell_nums=False,   # number grid cells
            show_tri=True,         # show minimum-enclosing triangle
            show_edge_cells=True,  # gray out 'edge cells'
            use_chrom=True)       # plot image in chromatic coordinates (not RGB)



### Fit an 8x5 grid to an image with white background and plot the grid
fn = '2255D.png'
im = cv2.imread(os.path.join('images',fn))[:,:,::-1]
g = Grid(8,5,background='white')
g.fit_sample(im);
g.plot_grid(image=im, show_gridlines=True, show_cell_nums=True, show_tri=True,
            show_edge_cells=True, use_chrom=False)


### Sample color features from an image
fn = '681.png'
im = cv2.imread(os.path.join('images',fn))[:,:,::-1]
g = Grid(8,5,background='black')
g.fit_sample(im)
#(array([ 111.31605184,  116.19882777,  123.32599119,  110.49406229,
#         119.67112299,  109.53637771,  112.9796173 ,  131.9183147 ,
#         ...,             4.34933523,   11.90519015,   13.93916276,
#          10.11596208,   21.73463472,   12.86757322,           nan]),
#array([False, False, False, False, False, False,  True,  True,  True,
#       False, False,  True,  True, False, False, False, False, False,
#         ...,  True,  True,  True, False, False,  True,  True, False,
#       False, False, False, False, False, False], dtype=bool))


### Check through images to determine if any need to be re-Photoshopped
filenames = ['WRK-WS-02259D.png','WRK-WS-00684_Polythore_mutata_M.png',
             'WRK-WS-01927D.png','WRK-WS-02272D.png']
for fn in filenames:
    try:
        im = cv2.imread(os.path.join('images',fn))[:,:,::-1]
        g = Grid(10,10,background='black')
        g.fit(im);
    except RuntimeError:
        print fn
#1927D bad.png

### Sample several images with a 12x10 grid
import pandas as pd # load pandas
import time
md = pd.read_csv(os.path.join('images','metadata.csv'),header=0) # read a table

# Initialize the grid
g = Grid(12,10,background='black')

# Set up dataframes for features and f_mask
features = pd.DataFrame(data=[],columns=g.f_labels_)
f_mask = pd.DataFrame(data=[],columns=g.f_labels_)

# Sample each image and save results
for fn in md['filename']:
    im = cv2.imread(os.path.join('images',fn))[:,:,::-1]
    t0 = time.clock()
    f,fm = g.fit_sample(im)
    t1 = time.clock()
    features.loc[fn] = f
    f_mask.loc[fn] = fm
    print 'Finished {} in {:.3f}s'.format(fn,t1-t0)
print features.shape, f_mask.shape
#(15, 720) (15, 720)


### Save sampled features and f_masks
# Make a temporary directory (if it doesn't yet exist) to put data
if not os.path.isdir('temp'):
    os.mkdir('temp')

# Export pandas dataframes as CSV files
fn = os.path.join('temp','features.csv')
features.to_csv(fn,header=True,index=False)
fn = os.path.join('temp','f_mask.csv')
f_mask.to_csv(fn,header=True,index=False)

# Import CSV files to pandas dataframes
fn = os.path.join('temp','features.csv')
features = pd.read_csv(fn,header=0)
fn = os.path.join('temp','f_mask.csv')
f_mask = pd.read_csv(fn,header=0)


### Run the PCA
an = Analyze(features,f_mask) # initialize analysis
an.fit_pca() # do the PCA

### Run an LDA on features and plot LD1 against LD2
# Get labels
labels = md['label']
indiv_labels = np.array(range(len(labels)),dtype=str) # generate indiv labels

# Run LDA using `labels`
an = Analyze(features,f_mask) # initialize analysis
an.fit_lda(labels) # do the LDA

# Get transformed data
features_transformed = an.features_transformed_
print features_transformed.shape
#(15, 7)

# Plot components (LD1 vs LD2)
title = '{} non-edge cells ({} features)'.format(an.n_features_masked_/6,
                                                 an.n_features_masked_)
an.plot_comps(labels,comps=[1,2],indiv_labels=indiv_labels,title=title)

# Get indices and labels for the 5 features that contribute most to LD1
print an.loadings_top_contrib(comps=[1],n_highest=5,grid=None)
print an.loadings_top_contrib(comps=[1],n_highest=5,grid=g)
#[ 78 135 120  84 121]
#['gm22' 'gs82' 'gs32' 'gm32' 'gs33']




# Plot PC1 vs PC2
an.plot_comps(labels,comps=[1,2],indiv_labels=None,title=title)

### View the loadings on PC1
an.loadings_plot_bar(g,comp=1)


### Plot the loadings of PC1 and 2, highlighting the top 5 contributors
an.loadings_plot_2d(grid=g,comps=[1,2],n_highest=5)


### Visualize LD1's loadings on an image
fn = '672.png'#'WRK-WS-02272D.png'
im = cv2.imread(os.path.join('images',fn))[:,:,::-1]
g = Grid(12,10,background='black')
g.fit(im)
an.loadings_image_overlay(im,grid=g,comps=[1])
