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
fn = 'WRK-WS-01951D.png'

# read an image file
im = cv2.imread(os.path.join('images',fn))

# reverse order of color channels (cv2 imports as BGR but we need RGB)
im = im[:,:,::-1]

g = Grid(10,10,background='black') # initialize the grid
g.fit_sample(im); # fit the grid to image and sample it

# plot the fitted grid on the image
g.plot_grid(image=im, show_gridlines=True, show_cell_nums=True, show_tri=True, 
            show_edge_cells=True, use_chrom=False)


### Fit an 8x5 grid to an image with white background and display it
fn = '109-2.tif'
im = cv2.imread(os.path.join('images',fn))[:,:,::-1]
g = Grid(8,5,background='white')
g.fit_sample(im);
g.plot_grid(image=im, show_gridlines=True, show_cell_nums=True, show_tri=True, 
            show_edge_cells=True, use_chrom=False)


### Sample color features from an image
fn = '109-2.tif'
im = cv2.imread(os.path.join('images',fn))[:,:,::-1]
g = Grid(8,5,background='white')
features,f_mask = g.fit_sample(im)


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
#WRK-WS-01927D.png
 
       
### Sample 13 images with a 12x10 grid
filenames = ['WRK-WS-00672_Polythore_mutata_F.png',
             'WRK-WS-00674_Polythore_mutata_M.png',
             'WRK-WS-00675_Polythore_mutata_M.png',
             'WRK-WS-00676_Polythore_mutata_F.png',
             'WRK-WS-00677_Polythore_mutata_M.png',
             'WRK-WS-00681_Polythore_mutata_F.png',
             'WRK-WS-00684_Polythore_mutata_M.png',
             'WRK-WS-00685_Polythore_mutata_M.png',
             'WRK-WS-00686_Polythore_mutata_F.png',
             'WRK-WS-01922D.png','WRK-WS-01949D.png',
             'WRK-WS-02259D.png','WRK-WS-02272D.png']
# Initialize the grid
g = Grid(8,5,background='black')

# Set up arrays for features and f_mask
features = np.zeros((len(filenames),g.n_features_),dtype=float)
f_mask = np.zeros((len(filenames),g.n_features_),dtype=bool)

# Sample each image and save results
for i in range(len(filenames)):
    fn = filenames[i]
    im = cv2.imread(os.path.join('images',fn))[:,:,::-1]
    f,fm = g.fit_sample(im)
    features[i] = f
    f_mask[i] = fm
    print "Just finished: %s" % fn


### Save sampled features and f_masks
if not os.path.isdir('temp'):
    os.mkdir('temp') # make a temporary directory

fn = os.path.join('temp','features.npy')
np.save(fn,features) # save features to NumPy binary

fn = os.path.join('temp','f_mask.npy')
np.save(fn,f_mask) # save f_mask to NumPy binary


### Import sampled features and f_masks
fn = os.path.join('temp','features.npy')
features = np.load(fn) # read features from NumPy binary

fn = os.path.join('temp','f_mask.npy')
f_mask = np.load(fn) # read f_mask from NumPy binary


### Run an LDA on features and plot LD1 against LD2
labels = ['P mutata F','P mutata M','P mutata M','P mutata F','P mutata M',          
          'P mutata F','P mutata M','P mutata M','P mutata F','M petreus M',
          'I agnosia F','C tutia','Ci aurorina aurorina']
          
an = Analyze(features,f_mask) # initialize analysis
an.fit_lda(labels) # do the LDA

# Generate labels for individuals
indiv_labels = np.array(range(len(labels)),dtype=str)

an.plot_comps(labels,comps=[1,2],indiv_labels=indiv_labels)

### Run a PCA and plot PC1 against PC2
an = Analyze(features,f_mask) # initialize analysis

an.fit_pca() # do the PCA

an.plot_comps(labels,comps=[1,2],indiv_labels=None)


### View the loadings on PC1
an.loadings_plot_bar(g,comp=1)


### Get the labels of the 5 features that contribute most to PCs1-5
an.loadings_top_contrib(comps=[1],n_highest=5,grid=g)


### Plot the loadings of PC1 and 2, highlighting the top 5 contributors
an.loadings_plot_2d(grid=g,comps=[1,2],n_highest=5)


### Visualize PC1's loadings on an image
fn = 'WRK-WS-00672_Polythore_mutata_F.png'#'WRK-WS-02272D.png'
im = cv2.imread(os.path.join('images',fn))[:,:,::-1]
g = Grid(8,8,background='black')
g.fit(im)
an = Analyze(features,f_mask) # initialize analysis
an.fit_pca()
an.loadings_image_overlay(im,grid=g,comps='all')
