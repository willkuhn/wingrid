# -*- coding: utf-8 -*-
"""
Wingrid Insect Wing Color Analysis Package
"""

# Author: William R. Kuhn

# License: GNU GPL License v3

from .helpers import *

import cv2 # MUST BE version 3.0 or higher
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from matplotlib.collections import PatchCollection
from matplotlib import lines
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

################################################################################
# GRID SAMPLER CLASS

class Grid():
    """Grid Sampler

    Takes an image that contains a pair of wings, fits a stretchable grid to
    the wings in the image, and samples color features from the grid.

    For now, images are assumed to have had their backgrounds removed
    beforehand with photo-editing, such as GIMP. See Image Preparation Guide
    for details.

    IMPORTANT: Images must be RGB, not BGR. OpenCV imports images as BGR, so if
    if is used to import an image for grid-fitting, the channel order must be
    reversed beforehand, like so:
        >>> im = cv2.imread(im_filename)[:,:,::-1]

    The steps for fitting a grid to an image are as follows:
      (1) Detect wings in the image with simple thresholding.
      (2) Fit a minimum-enclosing triangle ABC to the wings, so that AB is line
          along the anterior margin of the set of wings, AC is a line along
          the posterior margin and A is a vertex at the basal edges of the
          wings.
      (3) Define n_grid_rows-1 evenly-spaced lines between AB and AC that all
          intersect vertex A. Take the slopes (in degrees) of these lines, and
          those of AB and AC, as the variable `thetas`.
      (4) Find the closest and farthest wing pixel values from vertex A and
          take their distance from A as the beginning and end of a list of
          `radii`. Calculate n_grid_cols-1 additional evenly-spaced radii
          between these bounds and add them to `radii`.
      (5) Grid cells can now be formed, where rows are lines radiating from
          vertex A at the angles defined in `thetas` and columns are arcs around
          vertex A at distances defined in 'radii'.

    Parameters
    ----------
    n_grid_rows,n_grid_cols : int, optional (default 10 and 10)
        Desired number of rows and columns in the grid. Each must be >=1.

    background : str, optional
        Background color in the image (background must be converted to this
        color in Photoshop, for example, prior to sampling), possible values:
          - 'black' (default)
          - 'white'

    use_chrom : bool, optional (default True)
        Whether to transform the RGB image pixel values to chromatic coordinates
        prior to sampling the image.

    blur : bool, optional (default True)
        Passed to `RGB2chrom()`; whether blurring is applied before an image is
        converted to chromatic coordinates.

    max_dim : int, optional (default 1000)
        Maximum dimension for an input image. If the image excedes this amount
        in height or width, the image will be downsampled so that the maximum of
        those equals `max_dim`

    min_object_size : int, optional (default 20)
        Used to remove small objects during masking. Minimum size of an object
        (in terms of no. of edge pixels) that will be accepted by
        `Grid._get_mask()`. Smaller objects will be filtered out. Increasing
        this value means that larger objects will be removed during masking.

    Attributes
    ----------
    n_cells_ : int
        Number of cells in the grid (n_grid_rows*n_grid_cols).

    n_features_ : int
        Number of features that will be produces (3*2*n_cells_).

    thetas_ : array, shape (n_grid_rows+1,)
        Slopes (in degrees) used to draw rows of the grid.

    radii_ : array, shape (n_grid_cols+1,)
        Radii used to draw columns of the grid.

    mask_ : bool array, same shape as input image
        Boolean mask of input image.

    hull_pts_ : array, shape (arbitrary,2)
        Coordinates of edge pixels in the convex image of `mask_`.

    tri_ : array, shape (3,2)
        Coordinates of minimum-enclosing triangle around the wings in the input
        image.

    features_ : array, shape (n_features,)
        Features calculated by sampling input image. `np.nan` is returned for
        features from 'edge cells' (see `f_mask_` description below).

    f_mask_ : bool array, shape (n_features_,)
        List describing condition of the cell where each feature is derived.
        `False` means that the cell contains some of the input image's mask
        and/or that the cell is out of the bounds of the image. Cells that
        meet either/both of these conditions are called 'edge cells'. `True`
        means that the cell is not an 'edge cell'. This mask is used later for
        filtering out edge cells among a set features extracted from multiple
        images.

    f_labels_ : array, shape (n_features_,)
        List of strings labels describing each feature in the form
        [channel: r,g,b][method: m,s][cell number]. Ex, the label 'gs12'
        describes a feature derived from the stdev of cell 12 in the green
        channel.

    cell_px_coords_ : list of arrays, length (n_grid_cols*n_grid_rows)
        (x,y) coordinates of all pixels in each grid cell.

    Examples
    --------
    TODO:
    >>> from wingrid import Grid
    >>> import cv2
    >>> grid = Grid(n_lines=5,n_arcs=8)
    >>> im = imread('image.jpg')
    >>> im = im[:,:,::-1] # convert from openCV's BGR to RGB
    >>> grid.fit(im)
    >>> features,f_mask = grid.coeffs_, grid.f_mask_
    [1]
    """

    def __init__(self, n_grid_rows=5, n_grid_cols=8, background='black',
                 use_chrom=True, blur=True, max_dim=1000, min_object_size=20):

        self.n_grid_rows     = n_grid_rows
        self.n_grid_cols     = n_grid_cols
        self.background      = background
        self.use_chrom       = use_chrom
        self.blur            = blur
        self.max_dim         = max_dim
        self.min_object_size = min_object_size

        # Check `n_grid_rows` and `n_grid_cols`
        if n_grid_rows<1:
            raise ValueError("`n_grid_rows` must be >=1")
        elif n_grid_cols<1:
            raise ValueError("`n_grid_cols` must be >=1")

        # Check that cv2 includes minEnclosingTriangle
        self._check_opencv_version()

        # Calculate the number of cells in the grid
        self.n_cells_ = n_grid_rows*n_grid_cols

        # Calculate the no. of features to be extracted
        self.n_features_ = n_grid_rows*n_grid_cols*2*3

        # Make feature labels
        self._get_f_labels()


    @staticmethod
    def _check_opencv_version():
        """OpenCV versions before 3.1 are missing `cv2.minEnclosingTriangle`,
        which is required for `Grid._calc_triangle()`. This returns an error if
        that function is missing."""

        if not 'minEnclosingTriangle' in dir(cv2):
            msg =("OpenCV version >=3.1 (with `minEnclosingTriangle`) is"
                     " required. Found v{}".format(cv2.__version__))
            raise RuntimeError(msg)


    def _get_f_labels(self):
        """Creates feature labels.

        Labels are strings in the form [channel][statistic][cell no.], where
        channel is 'r', 'g', or 'b', statistic is 'm' for mean or 's' for
        standard deviation and cells are numbered left to right, top to bottom
        in the grid, starting with 0."""

        n_cells = self.n_cells_
        labels = np.array([ch+st+str(c) for ch in ('rgb') for st in ('ms') \
                           for c in range(n_cells)])
        labels = np.array(labels)
        self.f_labels_ = labels


    def _get_mask(self,image):
        """Masks input image.

        Masks by thresholding using the specified `background` then looking for
        large objects (i.e. wings) in the thresholded image."""

        min_object_size = self.min_object_size
        background      = self.background

        # Check image first
        if image is None:
            raise ValueError('Input image is `None`.')

        # Downsample large images
        im = downscaleIf(image, max_dim=self.max_dim)
        gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)

        # Threshold image
        if background=='white':
            _,thresh = cv2.threshold(gray,254,255,type=cv2.THRESH_BINARY)
        elif background=='black':
            _,thresh = cv2.threshold(gray,1,255,type=cv2.THRESH_BINARY_INV)
        else:
            raise ValueError("Unrecognized value for `background`. Available options: ['black','white']")

        # Find objects in the image
        _,cnts,_ = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

        # Add to a blank mask all objects that are at least 500 edge pixels in size
        mask = np.ones(thresh.shape, dtype=np.uint8)*255

        # Check that at least 1 object is exceeds `min_object_size`
        if np.all([len(c)<min_object_size for c in cnts]):
            raise RuntimeError('`_get_mask` failed to detect wings in image. Try lowering `min_object_size`.')

        for c in cnts:
        	if len(c)>=min_object_size: # only include 'large' objects
        		cv2.drawContours(mask, [c], -1, 0, -1)

        mask = 255 - mask # Invert mask (so white = wings, black = background)
        mask = mask.astype(bool)

        # Verify mask shape
        if image.shape[:2] != mask.shape:
            raise RuntimeError('Mask shape does not match image shape: %s,%s'
                                % ( str(mask.shape)),str(image.shape[:2]) )

        # Check that mask doesn't include too much of the image's perimeter
        perim = np.concatenate([mask[:8].ravel(),
                                mask[-8:].ravel(),
                                mask[:,:8].ravel(),
                                mask[:,-8:].ravel()])
        if perim.tolist().count(True)/float(len(perim))>0.15:
            raise RuntimeError("Image was incorrectly masked. Make sure that"
            " the image's background has been removed and replaced with black"
            " or white.")

        self.mask_ = mask

    def _get_hull_pts(self):
        """Get coordinates of edge pixels in mask"""

        mask = self.mask_
        thresh = mask.astype(np.uint8)*255

        # Get contours
        _,cnts,_ = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        if len(cnts)>1: # if more than one object in image
            cnts = np.concatenate(cnts) # join cnts to get points for all objects
        elif len(cnts)==1: # else take the 1st object in cnts, cast as array
            cnts = np.array(cnts[0])
        else:
            raise RuntimeError('Failed to detect wings in image.')

        # Get hull points
        hull_pts = cv2.convexHull(cnts)

        self.hull_pts_ = hull_pts


    def _calc_triangle(self):
        """Calculates the minimum-enclosing triangle around the wings."""

        hull_pts = self.hull_pts_

        # CALCULATE TRIANGLE
        area,tri = cv2.minEnclosingTriangle(hull_pts) # Min triangle
        tri = tri.reshape((3,2))

        # REDEFINE TRIANGLE VERTICES (ABC)
        # A: left-most: lowest x-valued points
        a = np.argmin(tri[:,0])
        # A: upperleft-most: lowest y-valued of the 2 lowest x-valued points
        #a = np.argmin(tri[np.argsort(tri[:,0])[:2],1])
        # A: closest point to origin (0,0)
        #a = np.argmin([np.sqrt([pt[0]**2 + pt[1]**2) for pt in tri])
        # TODO: refine strategy for determining vertex A

        # B: uppermost (lowest y-value [in img coord sys]) of the remaining 2
        remaining_two = list({0,1,2}-{a})
        b = remaining_two[np.argmin(tri[remaining_two,1])]

        # C: remaining vertex
        c = list({0,1,2} - {a,b})[0]

        A,B,C = [tri[i] for i in [a,b,c]]
        tri_new = np.array((A,B,C))

        self.tri_ = tri_new


    def _define_grid(self):
        """Calculate slopes for grid rows and radii for columns."""

        A,B,C = self.tri_
        hull_pts = self.hull_pts_
        n_grid_rows = self.n_grid_rows
        n_grid_cols = self.n_grid_cols

        # Randomly sample a subset of hull points if there lots
        if len(hull_pts)>200:
            hull_pts = random.sample(hull_pts,200)
        else:
            hull_pts = hull_pts

        # Calculate radii
        dists = [euclidean(A,i) for i in hull_pts]
        radii = np.linspace(min(dists), max(dists),n_grid_cols+1)

        # Calculate slopes:
        # Note: Since we're in image coordinates (y-axis is flipped),
        # upper slope is typically neg & lower is typically pos.
        slopeAB = slope(A,B) # slope of the upper edge
        slopeAC = slope(A,C) # slope of the lower edge
        thetas = np.linspace(slopeAB, slopeAC,n_grid_rows+1)

        self.radii_  = radii
        self.thetas_ = thetas


    @ staticmethod
    def _get_px_coords_for_cell(cell_row,cell_col,tri,radii,thetas,
                                step=2):
        """Calculates the (x,y) coordinates of pixels in a grid cell.

        Used in self._get_coords().

        Parameters
        ----------
        cell_row,cell_col : ints
            Index of cell, where 0<=cell_row<=n_grid_rows and
            0<=cell_col<=n_grid_cols

        tri,radii,thetas : params
            Parameters of the grid

        step : int (default 2)
            Computation-saving parameter, will return every `skip` pixel
            in the row and column direction. With `step=2`, coordinates
            are returned for every other pixel, row-wise and column-wise.
            With `step=1`, all pixel coordinates are returned.

        Returns
        -------
        coords : array, shape (~n_pixels_in_cell/(step**2),2)
            (x,y) coordinates for pixels in the specified cell
        """

        A,B,C = tri
        row = cell_row
        col = cell_col

        # Get cell's corners (coords of the intesections of its 2 lines and 2 arcs)
        rL,rR = radii[[col,col+1]] # radii of left and right bounding arcs
        tU,tD = thetas[[row,row+1]] # slope of upper and lower bounding lines
        corner_UL = polar2cart(rL,tU,originXY=A) # upper left corner (on image)
        corner_LL = polar2cart(rL,tD,originXY=A) # lower left corner (on image)
        corner_UR = polar2cart(rR,tU,originXY=A) # upper right corner (on image)
        corner_LR = polar2cart(rR,tD,originXY=A) # lower right corner (on image)
        corner_UL,corner_LL,corner_UR,corner_LR = \
            np.array([corner_UL,corner_LL,corner_UR,corner_LR],dtype=int)

        # Calculate cell's bounding box as (left,right,top,bottom)
        if min(tU,tD)<0<max(tU,tD): # if cell crosses the x-axis
            bbL = min(corner_UL[0], corner_LL[0])
            # right bound may be on arc, which crosses x-axis at (rR+A[0],0)
            bbR = max(corner_UR[0], corner_LR[0], int(rR+A[0]))
            bbU = min(corner_UL[1], corner_UR[1])
            bbD = max(corner_LL[1], corner_LR[1])
        else:
            bbL = min(corner_UL[0], corner_LL[0])
            bbR = max(corner_UR[0], corner_LR[0])
            bbU = min(corner_UL[1], corner_UR[1])
            bbD = max(corner_LL[1], corner_LR[1])

        # List the coordinates of every `step` pixel in the cell's bounding box
        coords = np.array([(x,y) for y in range(bbU,bbD,step) \
                           for x in range(bbL,bbR,step)])

        # Convert those (x,y) pixel coordinates to (r,theta) polar coords
        coords_pol = np.array(map(lambda i: cart2polar(*i,originXY=A),
                              coords))

        # Find pixel coordinates within grid cell
        rWise =     np.bitwise_and(rL<=coords_pol[:,0],
                                   coords_pol[:,0]<rR)
        thetaWise = np.bitwise_and(tU<=coords_pol[:,1],
                                   coords_pol[:,1]<tD)
        coords =    coords[np.bitwise_and(rWise,thetaWise)]

        return coords


    @staticmethod
    def _safe_extract(x,y,image,mask,h,w):
        """Extracts the pixel value from an image at coordinate (x,y),
        returning np.nan if (x,y) is out-of-bounds or off the mask.

        Used in `self._sample_grid()`.

        Parameters
        ----------
        x,y : int
            Pixel coordinates, where (x,y) corresponds to the pixel at
            image[y,x]

        image : array
            Input image

        mask : array
            Image mask from fitted grid model

        h,w : int
            The height and width, respectively, `image`
        """
        if x<0 or y<0: # nan if (x,y) is out of left or top bounds
            return np.nan
        elif x>w-1 or y>h-1: # nan if (x,y) is out of right or bottom bounds
            return np.nan
        elif not mask[y,x]: # nan if pixel is outside of object's mask
            return np.nan
        else: # otherwise return the pixel value at coord (x,y)
            return image[y,x]


    @staticmethod
    def _sample_grid(self,image,mask,coords):
        """Samples image using pre-fitted grid and image mask.

        Used in `self.fit_sample()`.

        Parameters
        ----------
        image : array
            Input image

        mask,coords : params
            Parameters from fitted grid

        Returns
        -------
        features : array, shape (n_features,)
            Features calculated by sampling input image. `np.nan` is returned
            for features from 'edge cells' (see `f_mask_` description below).

        f_mask : bool array, shape (n_features_,)
            List describing condition of the cell where each feature is derived.
            `False` means that the cell contains some of the input image's mask
            and/or that the cell is out of the bounds of the image. Cells that
            meet either/both of these conditions are called 'edge cells'. `True`
            means that the cell is not an 'edge cell'. This mask is used later
            for filtering out edge cells among a set features extracted from
            multiple images.
        """

        f_mask = [] # image's feature mask
        features = [] # image featured

        h,w = image.shape[:2]

        for channel in cv2.split(image): # for each color channel: (r,b,g)

            # Wrapper for _safe_extract()
            extract = lambda px: self._safe_extract(px[0],px[1],channel,
                                                    mask,h,w)

            means = [] # grid's cell means
            stDevs = [] # grid's cell standard deviations
            f_m = [] # holds grid's feature mask

            for cell in coords: # for each cell in the grid

                # extract pixel values while also checking that each pixel is
                # within the image and within the image mask
                vals = np.array(map(extract,cell))

                # drop any nans
                vals_no_nan = vals[~np.isnan(vals)]

                # if sampling cell returned any nans, f_mask=False, else True
                if len(vals_no_nan)<len(vals):
                    f_m.append(False)
                else:
                    f_m.append(True)

                # Calculate the means & std of non-nan values in the cell
                if len(vals_no_nan)==0:
                    m =  np.nan
                    sd = np.nan
                else:
                    m =  np.mean(vals_no_nan)
                    sd = np.std(vals_no_nan)

                means.append(m)
                stDevs.append(sd)

            features.extend(means)
            features.extend(stDevs)
            f_mask.extend(f_m+f_m) # extend f_mask once for mean & once for std

        return np.array(features),np.array(f_mask)


    def _get_coords(self):
        """Calculates pixel coordinates for all cells in the grid."""

        n_grid_rows = self.n_grid_rows
        n_grid_cols = self.n_grid_cols
        tri         = self.tri_
        radii       = self.radii_
        thetas      = self.thetas_

        # Get row,col indices for all cells in the grid
        cells = [(row,col) for row in range(n_grid_rows) \
                 for col in range(n_grid_cols)]

        # Find the px coordinates in each cell
        coords = [self._get_px_coords_for_cell(r,c,tri,radii,thetas,
                  step=2) for r,c in cells]

        self.cell_px_coords_ = coords


    def fit(self,image):
        """Fit grid to an input image without sampling the image, returning
        `self`.

        Parameters
        ----------
        image : array
            Input image
        """

        # Check image first
        if image is None:
            raise ValueError('Input image is `None`.')

        # Downsample large images
        im = downscaleIf(image, max_dim=self.max_dim)

        self._get_mask(im) # mask the image
        self._get_hull_pts() # convex hull of the image mask
        self._calc_triangle() # find the min-enclosing triange
        self._define_grid() # define radii & thetas
        self._get_coords() # find the coordinates of pixels in each grid cell

        return self # allows this method to be chained to others


    def fit_sample(self,image):
        """Fit grid to an input image, and then sample
        color features from the image.

        Parameters
        ----------
        image : array
            Input image

        Returns
        -------
        features : array, shape (n_features,)
            Features calculated by sampling input image. `np.nan` is returned
            for features from 'edge cells' (see `f_mask_` description below).

        f_mask : bool array, shape (n_features_,)
            List describing condition of the cell where each feature is derived.
            `False` means that the cell contains some of the input image's mask
            and/or that the cell is out of the bounds of the image. Cells that
            meet either/both of these conditions are called 'edge cells'. `True`
            means that the cell is not an 'edge cell'. This mask is used later
            for filtering out edge cells among a set features extracted from
            multiple images.
        """

        # Check image first
        if image is None:
            raise ValueError('Input image is `None`.')

        # Downsample large images
        im = downscaleIf(image, max_dim=self.max_dim)

        # Fit grid to image
        self.fit(im)

        # Convert image from RGB to chromatic coordinates
        if self.use_chrom:
            im = RGB2chrom(im,blur=self.blur)

        features,f_mask = self._sample_grid(self,im,self.mask_,
                                            self.cell_px_coords_)

        self.features_ = features
        self.f_mask_   = f_mask

        return features,f_mask


    def get_params(self):
        """Get a dictionary of the grid model's current parameters."""

        keys = ['background',
                'cell_px_coords_',
                'f_labels_',
                'f_mask_',
                'features_',
                'hull_pts_',
                'mask_',
                'max_dim',
                'min_object_size',
                'n_cells_',
                'n_features_',
                'n_grid_cols',
                'n_grid_rows',
                'radii_',
                'thetas_',
                'tri_',
                'use_chrom']

        d = {}
        for k in keys:
            if hasattr(self,k):
                d[k] = getattr(self,k)

        return d


    def plot_grid(self,image=None,show_gridlines=True,show_cell_nums=True,
                  show_tri=True,show_edge_cells=True,use_chrom=False):
        """Vizualize a fitted and sampled grid.

        Raises RuntimeError if grid has not been fitted and sampled.

        Parameters
        ----------
        image : array, optional (default None)
            Input image on which the grid has been fitted. If no image is
            provided, grid will be plotted on a canvas matching `background`.

        show_gridlines : bool, optional (default True)
            Whether to show the gridlines of the fitted grid.

        show_cell_nums : bool, optional (default True)
            Whether to show the cell numbers of the fitted grid.

        show_tri : bool, optional (default True)
            Whether to show the minimum-enclosing triangle used to fit the grid.

        show_edge_cells : bool, optional (default True)
            Whether to show 'edge cells' (for which f_mask is False).

        use_chrom : bool, optional (default False)
            If an image is provided, whether to convert it to chromatic
            coordinates before plotting it.
        """
        if not hasattr(self,'f_mask_'):
            raise RuntimeError('Grid has not yet been fit to an image. Try'
                               ' `g=Grid().fit_sample(image);g.plot_grid()`.')

        A,B,C = tri = self.tri_
        thetas      = self.thetas_
        radii       = self.radii_
        coords      = self.cell_px_coords_
        background  = self.background

        # Set up plot colors
        colors = [(103, 255,  30), # for triangle
                  (103, 255,  30), # for triangle vertices
                  (  0,  95, 254), # for grid lines
                  (240,  59,   0), # for grid cell numbers
                  (191, 191, 191), # for edge cells, gray
                  ]

        # Rescale values to between 0 and 1
        colors = [(r/255.,g/255.,b/255.) for r,g,b in colors]

        # Grid linewidth
        glw = 1.

        # Font dictionary
        fontdict={'weight':'bold',
                  'fontsize':12}

        if image is not None:
            # Downsample large images
            im = downscaleIf(image, max_dim=self.max_dim)
        else:
            im = self.mask_ # used only to get image shape for adjusting bounds

        if use_chrom and image is not None:
            # Convert image to chromatic coordinates
            im = RGB2chrom(im)

        if show_edge_cells:
            f_mask      = self.f_mask_
            n_grid_rows = self.n_grid_rows
            n_grid_cols = self.n_grid_cols

            # Get channel,statistic,row,col indices for all cells in the grid
            cells = [(ch,st,row,col) for ch in range(3) for st in range(2)\
            for row in range(n_grid_rows) for col in range(n_grid_cols)]

            # Get indices of non-contributing cells ('edge cells')
            cells_noncontrib = np.array(cells)[~f_mask]


        # PLOT
        fig, ax = plt.subplots()

        # Add image
        if image is not None:
            ax.imshow(im)

        # Change color of figure background to match image background
        if background=='white':
            ax.set_axis_bgcolor('white')
        elif background=='black':
            ax.set_axis_bgcolor('black')

        # Plot edge cells
        if show_edge_cells:
                patches = []
                for (_,_,r,c) in cells_noncontrib:
                    w = Wedge(A,
                              radii[c+1], # outer radius
                              thetas[r], # theta1 (in deg)
                              thetas[r+1], # theta2 (in deg)
                              width=radii[c+1]-radii[c]) # outer - inner radius
                    patches.append(w)
                p = PatchCollection(patches,
                                    facecolor=colors[4],
                                    edgecolor=None,
                                    alpha=0.2)
                ax.add_collection(p)

        # Plot triangle
        if show_tri:
            #Draw points of triangle
            plt.plot(*tri[[0,1,2,0]].T,
                     color=colors[0],
                     marker='o',
                     markersize=5,
                     lw=glw,
                     markeredgecolor='none')

            #Label triangle points
            for lbl,coord in zip(('ABC'),tri):
                ax.annotate(lbl,
                            xy=(0,0),
                            xytext=coord+(10,10),
                            color=colors[1],
                            **fontdict)

        # Show grid
        if show_gridlines:
            #Draw arcs as PatchCollection wedges
            patches = []
            for r in radii:
                w = Wedge(A,r,thetas[0],thetas[-1], width=0.001)
                patches.append(w)
            p = PatchCollection(patches,
                                facecolor='none',
                                edgecolor=colors[2],
                                lw=glw)
            ax.add_collection(p)

            #Draw lines
            for t in np.deg2rad(thetas):
                x0 = (radii[0] * np.cos(t)) + A[0]
                y0 = (radii[0] * np.sin(t)) + A[1]
                x1 = (radii[-1] * np.cos(t)) + A[0]
                y1 = (radii[-1] * np.sin(t)) + A[1]
                line = lines.Line2D((x0,x1),(y0,y1),
                                    color=colors[2],
                                    lw=glw)
                ax.add_line(line)

        # Annotate cells with their cell number
        if show_cell_nums:
            # Get centroids of each cell
            cell_centers = map(lambda c: np.mean(c,axis=0), coords)

            for i,(x,y) in enumerate(cell_centers):
                ax.text(x,y,str(i),
                        color=colors[3],
                        ha='center',
                        va='center',
                        **fontdict)

        # Adjust the plot's boundaries
        buf = (0.02*radii[-1]) # buffer
        xmin = min(0, # left side of image
                   A[0]-buf) # A's x-value
        xmax = max(A[0]+radii[-1]+buf, # right side of grid
                   B[0]+buf, # B's x-value
                   C[0]+buf, # C's x-value
                   im.shape[1]) # right side of image
        ymax = min(0, # top of image
                   B[1]-buf, # B's y-value
                   # top corner of grid
                   radii[-1]*np.sin(np.deg2rad(thetas[0]))+A[1]-buf)
        ymin = max(im.shape[0], # bottom of image
                   C[1]+buf, # C's y-value
                   # bottom corner of grid
                   radii[-1]*np.sin(np.deg2rad(thetas[-1]))+A[1]+buf)
        ax.set_xlim(xmin,xmax)
        ax.set_ylim(ymin,ymax)

        # adjust plot padding
        plt.subplots_adjust(bottom=0.06,
                            left=  0.06,
                            right= 0.96,
                            top=   0.96
                            )

        plt.show()

        # TODO: Add option to write figure to file.
        # TODO: Add option to manually provide alternative colors for items.


################################################################################
# GRID ANALYZER class

class Analyze():
    """Grid Features Analyzer

    Analyzes features extracted from multiple wing images using
    `Grid.fit_sample()`.

    Parameters
    ----------
    features : array, shape (n_features,n_samples)
        Dataset of features extracted from images using `Grid.fit_sample()`.

    f_mask : bool array, shape (n_features,n_samples), optional
        Feature mask determined using `Grid.fit_sample()`. It is strongly
        recommended that this be provided.

    scale_features : bool, optional (default True)
        Whether or not to scale features with
        `sklearn.preprocessing.StandardScaler` prior to transformation.

    Attributes
    ----------
    n_features_ : int
        The number of features in the dataset (no. columns of `features`)

    n_samples_ : int
        The number of individuals in the dataset (no. rows of `features`)

    f_mask_common_ : array of bools, shape (n_features,)
        Common feature mask: for each feature, if all values for that feature
        in `f_mask` are True, then the corresponding value in `f_mask_common_`
        is True, otherwise it is False. Features with "True" in this list
        were derived from cells that were consistently non-edge among all the
        grids fitted to images in the dataset.

    n_mask_features_ : int
        The number of features in the dataset after masking features

    n_components_ : int
        The number of components in the transformed dataset.
        - For PCA, n_components = min(n_features,n_samples)
        - For LDA, n_components = n_classes-1

    n_features_masked_ : int
        The number of features in the dataset after masking. This is equal to
        the number of 'True's in `f_mask_common_`.

    features_masked_ : array
        Features after masking.

    features_transformed_ : array, shape (n_samples,n_components)
        Features transformed with PCA or LDA.

    loadings_ : array
        The matrix used in the PCA or LDA to transform data points, i.e. the
        'loading' values. Rows of this represent the contribution that each
        feature makes to a principal component or linear discriminant.
        - For PCA, an array of shape (n_components,n_features)
        - For LDA, an array of shape (n_classes,n_features)

    explained_variance_ratio_ : array, shape (n_components,)
        The proportion of the total variance in features that is captured in
        each component

    classes_ : array, shape (n_classes)
        For LDA only, a list of all classes on which the LDA was fitted

    method_ : str
        String describing which fit method has been run. Options: ['pca','lda']

    Notes
    -----
    Notes text...

    TODO: Examples
    --------
    >>> from wingrid import Grid,Analyze
    >>> import cv2
    [1]
    """

    def __init__(self,features,f_mask=None,scale_features=True):

        self.features       = features
        self.f_mask         = f_mask
        self.scale_features = scale_features
        self.n_samples_     = features.shape[0]
        self.n_features_    = features.shape[1]

        if f_mask is not None:
            # Check shapes of features and f_mask match
            if features.shape != f_mask.shape:
                raise ValueError("`features` and `f_mask` shapes must match."
                                 "Got %s and %s." % (str(features.shape),
                                 str(f_mask.shape)))

            self._mask_features() # Mask features


    def _mask_features(self):
        """Finds a common set of features from consistently non-edge cells."""

        features = self.features
        f_mask   = self.f_mask

        f_mask_common = np.all(f_mask,axis=0)
        features_masked = features.T[f_mask_common].T

        self.f_mask_common_     = f_mask_common
        self.features_masked_   = features_masked
        self.n_features_masked_ = features_masked.shape[1]


    def fit_pca(self):
        """Perform a principal component analysis (PCA) on the
        features."""

        # Mask features
        if self.f_mask is not None:
            X = self.features_masked_
        else:
            X = self.features

        # Rescale features
        if self.scale_features:
            X = StandardScaler().fit_transform(X)

        # Do PCA
        pca = PCA(n_components=None,whiten=True)
        features_transformed = pca.fit_transform(X)

        self.features_transformed_     = features_transformed
        self.n_samples_                = features_transformed.shape[0]
        self.n_components_             = features_transformed.shape[1]
        self.loadings_                 = pca.components_
        self.explained_variance_ratio_ = pca.explained_variance_ratio_
        self.method_                   = 'pca'

        return self

    def fit_lda(self,labels):
        """Perform a linear discriminant analysis (LDA) on the features.

        Parameters
        ----------
        labels : array, shape (n_samples,)
            Target values, i.e. a variable that groups samples into some
            arbitrary subsets.
        """
        if self.f_mask is not None:
            X = self.features_masked_
        else:
            X = self.features

        # Rescale features
        if self.scale_features:
            X = StandardScaler().fit_transform(X)

        # Do LDA
        lda = LinearDiscriminantAnalysis(n_components=None)
        features_transformed = lda.fit_transform(X,labels)

        self.features_transformed_     = features_transformed
        self.n_samples_                = features_transformed.shape[0]
        self.n_components_             = features_transformed.shape[1]
        self.loadings_                 = lda.coef_
        self.explained_variance_ratio_ = lda.explained_variance_ratio_
        self.classes_                  = np.unique(labels)
        self.method_                   = 'lda'

        return self

    def loadings_top_contrib(self,comps=[1],n_highest=-1,grid=None):
        """Get the top-contributing features for 1+ components.

        Returns either the indices or labels of the features that contributed
        most to the specified components.

        Parameters
        ----------
        comps : list or 'all', optional (default [1])
            List of 1 or more components with which to calculate the highest-
            contributing features. '[1]' means the first component/discriminant
            will be used to score feature contributions. Values must be between
            1 and n_components. If 'all' is passed, highest-contributing
            features are determined for all components.

        n_highest : int, optional (default -1)
            Number of highest-contributing features to return. Must be between
            1 and n_features. If -1, returns indices for all features, sorted
            from highest to lowest contribution.

        grid : `Grid` instance, optional (default None)
            An initialized instance of `Grid`, thus having the attribute
            `f_labels_`. The grid in `grid` must match the grid used to sample
            `features`.

        Returns
        -------

        """
        # Check that a model has been fitted already
        if not hasattr(self,'method_'):
            raise RuntimeError('A model has not yet been fit to the features. '
                            'Run `Analyze.fit_pca()` or `Analyze.fit_lda()`.')

        # Check that comps is a list and values are within proper range
        if comps is 'all':
            comps = range(1,self.n_components_+1)
        elif not type(comps)==list:
            raise ValueError("`comps` must be list or 'all'. Got type: {}"\
                             .format(type(comps)))
        else:
            for c in comps:
                if not 1<=c<=self.n_components_:
                    raise ValueError('`comps` values must be between 1 &'
                                     ' n_components')

        loadings = self.loadings_ # Loadings

        if self.f_mask is not None:
            n_f = self.n_features_masked_
            # Check 'n_highest' values
            if not 1<=n_highest<=n_f:
                raise ValueError('`n_highest` must be <=n_features_masked.')
        else:
            n_f = self.n_features_
            # Check 'n_highest' values
            if not 1<=n_highest<=n_f:
                raise ValueError('`n_highest` must be <=n_features.')

        # Calculate highest-contributing loading values by their distance from origin
        loading_dists = np.zeros(n_f)
        for i in range(n_f):
            # calculate Euclidean distance from loadings to origin:
            loading_dists[i] = np.sqrt(np.sum([loadings[j,i]**2 for j in [c-1 for c in comps]]))

        # get indices for `loading_dists` sorted highest to lowest
        hc = np.argsort(loading_dists)[::-1]
        hc = hc[:n_highest] # indices of the highest contributors

        # if something is provided for `labels`
        if grid is not None:

            # if it's a `grid` instance, get f_labels
            if hasattr(grid,'f_labels_'):
                f_labels = grid.f_labels_
            else:
                raise ValueError('`grid` has not been initialized.')

            # check length of labels
            if not len(f_labels)==self.n_features_:
                raise ValueError('The grid size in `grid` and n_features do '
                'not match.')

            if self.f_mask is not None:
                # Mask labels
                f_labels = f_labels[self.f_mask_common_]

            return f_labels[hc]

        # otherwise return indices of the highest contributing features
        return hc


    def plot_comps(self,labels,comps=[1,2],filter_by=None,color_by=None,
                   label_classes=True,indiv_labels=None,title=None,
                   center_at_origin=True):
        """Draw a 2D scatterplot of transformed data.

        Parameters
        ----------
        labels : list-like of shape (n_samples,)
            Labels used to determine classes. Points for a class
            are connected by lines extending from the class' centroid.
        comps : list of 2 ints, optional (default [1,2])
            Which two PCs or LDs to plot. [1,2] means the first two PCs or LDs.
            Values must be between 1 and n_components.
        filter_by : list-like of shape (n_samples,), optional
            A Boolean list used to filter the plot to a subset of the samples
            for which `filter_by`==True. Default is None (no filtering).
            This is helpful for highlighting a subset of the data, while
            holding the plot limits constant.
        color_by : list-like of shape (n_samples,), optional
            Alternative grouping variable for coloring individual points.
            If not provided, indivuals are colored by class (as determined
            using `labels`)
        label_classes : bool, optional (default True)
            If True, class labels are placed at each class' centroid.
        indiv_labels : list-like of shape (n_samples,), optional
            List of labels for individuals (e.g., filenames or sample codes).
            Labels are placed alongside each individual's coordinate.
            If provided, must be of length n_samples even if `filter_by` is
            also provided. If None, individuals are not labeled.
        title : str (default None), optional
            A string to be added to the end of the plot's title.
        center_at_origin : bool (default True), optional
            If True, plot is centered at (0,0); otherwise, it's centered at
            the centroid of the data. Centering at the origin looks nice, but
            is not alway convenient, particularly when `filter_by` is used.
        """
        # Check that a model has been fitted already
        if not hasattr(self,'method_'):
            raise RuntimeError('A model has not yet been fit to the features. '
                            'Run `Analyze.fit_pca()` or `Analyze.fit_lda()`.')

        X                        = self.features_transformed_
        method                   = self.method_
        n_components             = self.n_components_
        explained_variance_ratio = self.explained_variance_ratio_

        # Check `comps`
        if len(comps)!=2:
            raise ValueError('`comps` must be a list of length 2.')
        elif not (1<=comps[0]<=n_components or 1<=comps[1]<=n_components):
            raise ValueError('`comps` values must be between 1 & n_components.')

        # Check `labels`
        if not len(labels)==len(X):
            raise ValueError('`labels` must be of length n_samples.')
        else:
            labels = np.array(labels) # make sure labels is an array

        # Check `indiv_labels`
        if indiv_labels is not None: # if individual labels are provided
            # Check length of indiv_labels
            if not len(indiv_labels)==len(X):
                raise ValueError("`indiv_labels` must be of length n_samples.")

        # Check `color_by`
        if color_by is not None: # if individual labels are provided
            # Check length of color_by
            if not len(color_by)==len(X):
                raise ValueError("`color_by` must be of length n_samples.")

        # Check & apply `filter_by`
        if filter_by is not None:
            filter_by = np.array(filter_by) # Make sure its an array

            if not len(filter_by)==len(X):
                raise ValueError('`filter_by` must be of length n_samples.')
            elif not filter_by.dtype is np.dtype('bool'):
                raise ValueError('`filter_by` must be Boolean')

            else: # Apply `filter_by`
                X      = X[filter_by]
                labels = labels[filter_by]
                if indiv_labels is not None:
                    indiv_labels = indiv_labels[filter_by]


        # PRE-PLOT
        classes = np.unique(labels) # get classes for connecting indivs
        if color_by is not None:
            # Get iterator list specifying color for each indiv
            groups = np.unique(color_by)
            group_colors = plt.cm.hsv(np.linspace(0.,0.75,len(groups)))
            gc_dict = dict(zip(groups,group_colors))
            group_colors_list = np.array([gc_dict[g] for g in color_by])
        else:
            # Get list for coloring by class
            class_colors = plt.cm.hsv(np.linspace(0.,0.75,len(classes)))

        plot_colors = [(  0,  0,  0), # class labels color: solid black
                       ( 89, 89, 89), # indiv labels color: 35% gray
                       ]

        # Rescale values to between 0 and 1
        plot_colors = [(r/255.,g/255.,b/255.) for r,g,b in plot_colors]

        # Get axes limits
        if center_at_origin:
            buf = 0.2*np.abs(X[:,[comps[0]-1,comps[1]-1]]).std()
            mmax = np.abs(X[:,[comps[0]-1,comps[1]-1]]).max() + buf
            xmax = ymax = mmax
            xmin = ymin = -mmax
        elif not center_at_origin: # TODO: 'off' doesn't seem to work here
            center = X.mean(axis=0) # find centroid
            resid = X - center # center data at centroid
            buf = 0.2*np.abs(resid[:,[comps[0]-1,comps[1]-1]]).std()
            mmax = np.abs(resid[:,[comps[0]-1,comps[1]-1]]).max() + buf
            xmax = mmax + center[0]
            ymax = mmax + center[1]
            xmin = -mmax + center[0]
            ymin = -mmax + center[1]
        else:
            raise ValueError('`center_at_origin` must be Boolean.')


        # PLOT
        fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(12,9))

        # draw lines at origin
        ax.vlines(0.,ymin,ymax,
                  colors='k',
                  linestyle='solid',
                  alpha=0.4)
        ax.hlines(0.,xmin,xmax,
                  colors='k',
                  linestyle='solid',
                  alpha=0.4)

        # Main plotting loop:
        if color_by is not None: # If color_by is provided:
            for cl in classes: #For each class
                points = np.atleast_2d(X[labels==cl])
                points = points[:,[comps[0]-1,comps[1]-1]]
                colors = np.atleast_2d(group_colors_list[labels==cl])

                # Calculate centroid
                mx,my = points.mean(axis=0)

                # Plot centroid (gray)
                ax.plot(mx,my,'+',
                        alpha=0.5,
                        ms=10,
                        color='k',
                        markeredgewidth=1)

                # Draw lines from all points to centroid (gray)
                for x,y in points: #For each indiv with `bn`
                    ax.plot((x,mx),(y,my),
                            linestyle='solid',
                            color='k',
                            alpha=0.3)

                # Plot all points (lookup colors from group_colors_list)
                plt.scatter(points.T[0],points.T[1],
                         marker='o',
                         c=colors,
                         linewidths=0,
                         s=100,
                         alpha=0.5)

        else: # If color_by is not provided:
            for cl,color in zip(classes,class_colors): #For each class
                points = np.atleast_2d(X[labels==cl])
                points = points[:,[comps[0]-1,comps[1]-1]]

                # Calculate centroid
                mx,my = points.mean(axis=0)

                # Plot centroid
                ax.plot(mx,my,'+',
                        alpha=0.5,
                        ms=10,
                        color=color,
                        markeredgewidth=1)

                # Draw lines from all points to centroid
                for x,y in points: #For each indiv with `bn`
                    ax.plot((x,mx),(y,my),
                            linestyle='solid',
                            color=color,
                            alpha=0.3)

                # Plot all points
                plt.scatter(points.T[0],points.T[1],
                         marker='o',
                         c=color,
                         linewidths=0,
                         s=100,
                         alpha=0.5)

        # Label individuals
        if indiv_labels is not None: # if individual labels are provided

            for (x,y),il in zip(X[:,[comps[0]-1,comps[1]-1]],indiv_labels):
                ax.text(x+0.15*buf,y-0.15*buf, # offset coordinates by 15%
                        il,
                        color=plot_colors[1],
                        fontsize=10,
                        ha='left',
                        va='top')

        # Label classes at their centroids
        # This is separate & below individual labels so that class labels are on top
        if label_classes:
            for cl in classes: #For each class
                points = np.atleast_2d(X[labels==cl])
                points = points[:,[comps[0]-1,comps[1]-1]]

                # Calculate centroid
                mx,my = points.mean(axis=0)

                # Label centroid with class label
                ax.text(mx+0.1*buf,my+0.1*buf, # offset coordinates by 10%
                        cl,
                        color=plot_colors[0],
                        fontsize=14,
                        ha='left',
                        va='bottom')

        # Label axes
        if method=='pca':
            xlabel = 'PC{} ({:.2f}% of total var)'.format(comps[0],explained_variance_ratio[0]*100)
            ylabel = 'PC{} ({:.2f}% of total var)'.format(comps[1],explained_variance_ratio[1]*100)
        elif method=='lda':
            xlabel = 'LD{} ({:.2f}% of total var)'.format(comps[0],explained_variance_ratio[0]*100)
            ylabel = 'LD{} ({:.2f}% of total var)'.format(comps[1],explained_variance_ratio[1]*100)

        # Set axes limits to center origin
        ax.set_xlim(xmin,xmax)
        ax.set_ylim(ymin,ymax)

        # Change color of the background outside plot to white
        ax.set_facecolor('white')

        # Label axes
        plt.xlabel(xlabel,fontsize=12)
        plt.ylabel(ylabel,fontsize=12)

        # Title the plot
        if method=='pca':
            plot_title = 'PC{} & {}'.format(comps[0],comps[1])
        elif method=='lda':
            plot_title = 'LD{} & {}'.format(comps[0],comps[1])
        if title is not None:
            plot_title += (' - '+title)
        plt.title(plot_title,fontsize=14)

        # adjust plot padding
        plt.subplots_adjust(bottom=0.06,
                            left=  0.06,
                            right= 0.96,
                            top=   0.96
                            )

        plt.show()


    def loadings_plot_bar(self,grid,comp=1):
        """Draw bar plot of loadings for a component of transformed features.

        Parameters
        ----------
        comp : int, optional (default 1)
            Principal component or linear discriminant to display loadings for.
            '1' means the first component/discriminant. Must be between
            1 and n_components.

        grid : `Grid` instance
            An instance of `Grid` that has been initialized, so that it
            has the attribute `f_labels_`. The grid parameter's must
            match those used to extract the the features.
        """
        # Check that a model has been fitted already
        if not hasattr(self,'method_'):
            raise RuntimeError('A model has not yet been fit to the features. '
                            'Run `Analyze.fit_pca()` or `Analyze.fit_lda()`.')

        # Check grid
        if not hasattr(grid,'f_labels_'):
            raise ValueError("Entry for `grid` not recognized.")

        # Check comp
        if not 1<=comp<=self.n_components_:
            raise ValueError('`comp` must be between 1 & n_components.')

        f_labels                 = grid.f_labels_ # Feature labels
        loadings                 = self.loadings_ # Loadings
        f_mask_common            = self.f_mask_common_ # Feature mask
        explained_variance_ratio = self.explained_variance_ratio_
        method                   = self.method_

        # check that grid size and features length match up
        if not len(f_mask_common)==len(f_labels):
            raise ValueError("It looks like the number of features in "
                "`f_mask` don't match the grid size in `grid`.")

        if self.f_mask is not None:
            # Mask feature labels
            f_labels = f_labels[f_mask_common]

        # Get variance for specified component
        var = explained_variance_ratio[comp-1]

        # Set up plot and subplots
        fig,ax = plt.subplots(nrows=1,ncols=1,facecolor='w',figsize=(15,3))

        # Add horizontal grid
        ax.grid(color='k',
                alpha=0.3,
                which='major',
                axis='y',
                linewidth=0.5,
                linestyle='solid')

        # Plot loadings (Axes.vlines() looks nicer than Axes.bar())
        ax.vlines(x=range(len(loadings[0])), # x-value for each line
                  ymin=0,
                  ymax=loadings[comp-1],        # length of each line
                  linestyles='solid',
                  color='k',
                  linewidths=1.2)

        # Set axis limits
        ymin = loadings[comp-1].min()-0.1*loadings[comp-1].std()
        ymax = loadings[comp-1].max()+0.1*loadings[comp-1].std()
        ax.set_xlim(-2,loadings.shape[1]+1) # add a little space above 1st bar and below last bar
        ax.set_ylim(ymin,ymax)

        # Label y-axis
        if method=='pca':
            ax.set_ylabel('Loadings on PC{}\n({:.2f}% of total var)'.format(comp,var*100))
        elif method=='lda':
            ax.set_ylabel('Loadings on LD{}\n({:.2f}% of total var)'.format(comp,var*100))

        # rotate the x-axis labels to vertical so they don't overlap
        plt.setp(ax.xaxis.get_majorticklabels(),rotation='vertical')

        # Label the x-ticks using the f_labels:
        ax.set_xticks(np.arange(len(loadings[0])))
        ax.set_xticklabels(f_labels)

        # adjust x-axis
        ax.tick_params(axis='x',
                       labelsize=10, # sets fontsize for x- and y-labels
                       length=0 # effectively removes x-tick marks from plot
                       )

        # Change color of the background outside plot to white
        ax.set_axis_bgcolor('white')

        # adjust plot padding
        plt.subplots_adjust(bottom=0.25,
                            left=  0.09,
                            right= 0.97,
                            top=   0.99
                            )

        plt.show()

        # TODO: This plot looks cramped when there are more than ~100 features
        # Perhaps only label the top-contributing features when n_features>100

    def loadings_plot_2d(self,grid,comps=[1,2],n_highest=10,title=None):
        """Scatterplot that shows, in 2 dimensions, the contribution that
        each feature makes toward the selected components. Thus, points that
        are relatively far away from the origin represent high contributing-
        features. The top `n_highest` contributing features are highlighted
        and labeled.

        This plot is adapted from one in Valentine Svensson's 29 November 2012
        blog post entitled, "Loadings with scikit-learn PCA"
        (http://www.nxn.se/valent/loadings-with-scikit-learn-pca and
        https://gist.github.com/vals/4172579).

        Parameters
        ----------
        comps : list of ints, optional (default [1,2])
            Principal components or linear discriminants to display loadings
            for. [1,2] means the first two PCs/LDs. Must be between 1 and
            n_components.

        grid : `Grid` instance
            An instance of `Grid` that has been initialized, so that it
            has the attribute `f_labels_`. The grid parameter's must
            match those used to extract the the features.

        n_highest : int, optional (default 10)
            Number of highest-contributing features to highlight in the plot.
            If `f_mask` was provided, must be between 1 and n_features_masked,
            otherwise must be between 1 and n_features.

        title : str or None (default), optional
            A string that will be added to the end of the plot's title.
        """

        # Check that a model has been fitted already
        if not hasattr(self,'method_'):
            raise RuntimeError('A model has not yet been fit to the features. '
                            'Run `Analyze.fit_pca()` or `Analyze.fit_lda()`.')

        # Check grid
        if not hasattr(grid,'f_labels_'):
            raise ValueError("Entry for `grid` not recognized.")

        # Check `comps` values
        if len(comps)!=2:
            raise ValueError('`comps` must be a list of length 2.')
        elif not (1<=comps[0]<=self.n_components_ or 1<=comps[1]<=self.n_components_):
            raise ValueError('`comps` values must be <=n_components.')

        f_labels                 = grid.f_labels_ # Feature labels
        loadings                 = self.loadings_ # Loadings
        explained_variance_ratio = self.explained_variance_ratio_
        method                   = self.method_

        # Set number of features and check `n_highest` value
        if self.f_mask is not None:
            n_f = self.n_features_masked_
            if not 1<=n_highest<=n_f:
                raise ValueError('`n_highest` must be <=n_features_masked.')
        else:
            n_f = self.n_features_
            if not 1<=n_highest<=n_f:
                raise ValueError('`n_highest` must be <=n_features.')

        # Get variance for specified component
        var = explained_variance_ratio[[c-1 for c in comps]]

        # Mask feature labels
        if self.f_mask is not None:
            f_labels = f_labels[self.f_mask_common_]

        # Calculate highest-contributing loading values by their distance from origin
        loading_dists = np.zeros(n_f)
        for i in range(n_f):
            # calculate Euclidean distance from loadings to origin (0,0):
            loading_dists[i] = np.sqrt(loadings[comps[0]-1,i]**2 + \
                                        loadings[comps[1]-1,i]**2)

        # get indices for `loading_dists` sorted highest to lowest
        hc = np.argsort(loading_dists)[::-1]
        hc = hc[:n_highest] # indices of the highest contributors

        # Needed to set axis limits and draw lines at origin
        mmax = np.abs(loadings[[comps[0]-1,comps[1]-1]]).max()+\
               0.2*loadings[[comps[0]-1,comps[1]-1]].std()

        # Set up plot and subplots
        fig,ax = plt.subplots(nrows=1,ncols=1,facecolor='w',figsize=(9,9))

        # draw lines at origin
        ax.vlines(0.,-mmax,mmax,
                  colors='k',
                  linestyle='solid',
                  alpha=0.5)
        ax.hlines(0.,-mmax,mmax,
                  colors='k',
                  linestyle='solid',
                  alpha=0.5)

        # Plot loadings for each PC as a separate bar plot:
        ax.plot(*loadings[[comps[0]-1,comps[1]-1]],
                marker='o',
                markersize=14,
                markeredgecolor='None',
                markerfacecolor='k',
                linestyle='None',
                alpha=0.3)

        # circle highest-contributing features
        # (plot them in reverse order so top contrib is on top)
        color = iter(plt.cm.viridis_r(np.linspace(0,1,len(hc))))
            # using viridis_r matches colors with loadings_image_overlay
            # (top contrib is yellow)
        for i,c in zip(hc,color)[::-1]:
            ax.plot(*loadings[[comps[0]-1,comps[1]-1],i],
                    marker='o',
                    markersize=18,
                    linestyle='None',
                    markeredgewidth=2.5,
                    markerfacecolor='None',
                    markeredgecolor=c,
                    alpha=0.6)

        # annotate the highest-contributing features
        for h in hc:
            ax.annotate(f_labels[h],# Annotate with coefficient label
                        xy=(0,0),
                        xycoords='data', #dummy coords
                        xytext=loadings[[c-1 for c in comps],h],
                        textcoords='data',
                        alpha=0.8,
                        fontsize=14)

        # Label x-axis
        if method=='pca':
            ax.set_xlabel('PC{} ({:.2f}% of total var)'.format(comps[0],var[0]*100))
            ax.set_ylabel('PC{} ({:.2f}% of total var)'.format(comps[1],var[1]*100))

        elif method=='lda':
            ax.set_xlabel('LD{} ({:.2f}% of total var)'.format(comps[0],var[0]*100))
            ax.set_ylabel('LD{} ({:.2f}% of total var)'.format(comps[1],var[1]*100))

        # rotate the x-axis labels by 45 deg so they don't overlap
        plt.setp(ax.xaxis.get_majorticklabels(),rotation=45)

        # set axis limits
        ax.set_xlim(-mmax,mmax)
        ax.set_ylim(-mmax,mmax)

        # Change color of the background outside plot to white
        ax.set_axis_bgcolor('white')

        # Title the plot
        if method=='pca':
            plot_title = 'Loadings on PC{} & {}'.format(comps[0],comps[1])
        elif method=='lda':
            plot_title = 'Loadings on LD{} & {}'.format(comps[0],comps[1])
        if title is not None:
            plot_title += (' - '+title)
        plt.title(plot_title,fontsize=14)

        # adjust plot padding
        plt.subplots_adjust(bottom=0.08,
                            left=  0.08,
                            right= 0.96,
                            top=   0.96
                            )

        plt.show()
        # TODO: add colorbar (use code from `loadings_image_overlay`)


    def loadings_image_overlay(self,image,grid,comps=[1],show_cell_nums=False,
                               use_chrom=True,title=None):
        """Visualize the loadings of all features on a grid-fitted image.

        This produces a 2x3 plot, where columns represent color channels (rgb),
        rows represent statistic (mean or stdev), and each subplot shows the
        fitted grid plotted on the appropriate channel of `image`.

        Parameters
        ----------
        image : array
            Input image on which to overlay a representation of loading values

        grid : `Grid` instance
            An instance of `Grid` fitted to the input image. The grid in `grid`
            must match the grid used to sample `features`.

        comps : list or 'all', optional (default [1])
            List of 1 or more components with which to calculate the highest-
            contributing features. '[1]' means the first component/discriminant
            will be used to score feature contributions. Values must be between
            1 and n_components. If 'all' is passed, highest-contributing
            features are determined for all components.
            Note: 'all' gives results which seem to vary everytime a PCA is fit
            to the same data. Use with caution.

        title : str or None (default), optional
            A string that will be added to the end of the plot's title.
        """
        # Check that a model has been fitted already
        if not hasattr(self,'method_'):
            raise RuntimeError('A model has not yet been fit to the features. '
                            'Run `Analyze.fit_pca()` or `Analyze.fit_lda()`.')

        # Check that comps is a list and values are within proper range
        if comps is 'all':
            Comps = range(1,self.n_components_+1)
        else:
            Comps = comps
        if not type(Comps)==list:
            raise ValueError("`comps` must be a list or 'all'. Got type: {}"\
                             .format(type(Comps)))
        else:
            for c in Comps:
                if not 1<=c<=self.n_components_:
                    raise ValueError('`comps` values must be between 1 &'
                                     ' n_components')

        # check that `grid` has been fitted
        if not hasattr(grid,'radii_'):
            raise ValueError('`grid` must be a fitted instance of'
                             ' `core.Grid`')

        # Get necessary variables from `grid`
        A,B,C         = grid.tri_
        thetas        = grid.thetas_
        radii         = grid.radii_
        coords        = grid.cell_px_coords_
        max_dim       = grid.max_dim
        background    = grid.background # image background color
        n_grid_rows   = grid.n_grid_rows
        n_grid_cols   = grid.n_grid_cols

        loadings      = self.loadings_ # Loadings

        # check that grid size and features length match up
        if not self.n_features_==n_grid_rows*n_grid_cols*2*3:
            raise ValueError('The grid size in `grid` and n_features do '
            'not match.')

        if self.f_mask is not None:
            n_f           = self.n_features_masked_
            f_mask_common = self.f_mask_common_
        else:
            n_f           = self.n_features_

        # CONTRIBUTIONS
        # Calculate highest-contributing loading values by their distance from origin
        loading_dists = np.zeros(n_f)
        for i in range(n_f):
            # calculate Euclidean distance from loadings to origin (0,0):
            loading_dists[i] = np.sqrt(np.sum([loadings[j,i]**2 for j in [c-1 for c in Comps]]))

        # rescale contributions to between 0 and 1
        contrib_resc_vals = (loading_dists-loading_dists.min())/ \
                            (loading_dists.max()-loading_dists.min()) # unsorted

        # Get channel,statistic,row,col indices for all cells in the grid
        cells = [(ch,st,row,col) for ch in range(3) for st in range(2)\
        for row in range(n_grid_rows) for col in range(n_grid_cols)]

        if self.f_mask is not None:
            # Get indices of contributing cells
            cells_contrib = np.array(cells)[f_mask_common]

            # Get indices of non-contributing cells ('edge cells')
            cells_noncontrib = np.array(cells)[~f_mask_common]
        else:
            cells_contrib = cells
            cells_noncontrib = np.array([])


        # PRE-PLOT
        # Set up plot colors
        colors = [(103, 255,  30), # for triangle
                  (103, 255,  30), # for triangle vertices
                  (  0,  95, 254), # for grid lines
                  (240,  59,   0), # for grid cell numbers
                  (191, 191, 191), # for edge cells, gray
                  ]

        # Rescale values to between 0 and 1
        colors = [(r/255.,g/255.,b/255.) for r,g,b in colors]

        # Color function for coloring contributing cells
        contrib_color_func = plt.cm.viridis

        # Grid linewidth
        glw = 0.5

        # Prepare image
        im = downscaleIf(image,max_dim=max_dim)
        if use_chrom:
            im = RGB2chrom(im)

        # check that `grid` has been fitted to `image`, specifically
        if not grid.mask_.shape == im.shape[:2]:
            raise ValueError('`grid` does not appear to have been fitted on'
                             ' `image`.')

        # Make (then clear) dummy image for colorbar
        dummy = np.atleast_2d(contrib_resc_vals)
        cbar_dummy = plt.imshow(dummy,cmap=contrib_color_func)
        plt.close()


        # PLOT
        # Set up plot: 2 rows (mean,std) x 3 cols (r,g,b)
        fig,axes = plt.subplots(nrows=2,ncols=3,figsize=(12,9),sharex=True,sharey=True)

        # Plot the r,g, & b image channels along columns
        for ch,col in enumerate(axes.T):
            for ax in col:
                ax.imshow(im[:,:,ch],cmap=plt.cm.gray)

                # Change color of figure background to match image background
                if background=='white':
                    ax.set_axis_bgcolor('white')
                elif background=='black':
                    ax.set_axis_bgcolor('black')

        # Color cells by contribution
        for st,row in enumerate(axes):   # rows: [mean,std]
            for ch,ax in enumerate(row): # cols: [r,g,b]

                # For contributing cells in that axis
                pertinent_cells = cells_contrib[\
                  np.bitwise_and(cells_contrib[:,0]==ch,cells_contrib[:,1]==st)]
                pertinent_contribs = contrib_resc_vals[\
                  np.bitwise_and(cells_contrib[:,0]==ch,cells_contrib[:,1]==st)]

                for (_,_,r,c),cont in zip(pertinent_cells,pertinent_contribs):
                    patch = [Wedge(A,
                              radii[c+1], # outer radius
                              thetas[r], # theta1 (in deg)
                              thetas[r+1], # theta2 (in deg)
                              width=radii[c+1]-radii[c], # outer - inner radius
                              )]
                    p = PatchCollection(patch,
                                        facecolor=contrib_color_func(cont),
                                        edgecolor=None,
                                        alpha=0.7)
                    ax.add_collection(p)

                # For non-contributing cells in that axis
                pertinent_cells = cells_noncontrib[\
                  np.bitwise_and(cells_noncontrib[:,0]==ch,
                                 cells_noncontrib[:,1]==st)]

                patches = []
                for (_,_,r,c) in pertinent_cells:
                    w = Wedge(A,
                              radii[c+1], # outer radius
                              thetas[r], # theta1 (in deg)
                              thetas[r+1], # theta2 (in deg)
                              width=radii[c+1]-radii[c]) # outer - inner radius
                    patches.append(w)
                p = PatchCollection(patches,
                                    facecolor=colors[4],
                                    edgecolor=None,
                                    alpha=0.5)
                ax.add_collection(p)


        # Draw grid and label cells
        for ax in axes.flat:

            # Draw grid
            #Draw arcs as PatchCollection wedges
            patches = []
            for r in radii:
                w = Wedge(A,r,thetas[0],thetas[-1], width=0.001)
                patches.append(w)
            p = PatchCollection(patches,
                                facecolor='none',
                                edgecolor=colors[2],
                                lw=glw)
            ax.add_collection(p)

            #Draw lines
            for t in np.deg2rad(thetas):
                x0 = (radii[0] * np.cos(t)) + A[0]
                y0 = (radii[0] * np.sin(t)) + A[1]
                x1 = (radii[-1] * np.cos(t)) + A[0]
                y1 = (radii[-1] * np.sin(t)) + A[1]
                line = lines.Line2D((x0,x1),(y0,y1),
                                    color=colors[2],
                                    lw=glw)
                ax.add_line(line)

            # Annotate cells with their cell number
            if show_cell_nums:
                # Get centroids of each cell
                cell_centers = map(lambda c: np.mean(c,axis=0), coords)

                # Contrast with image background
                if background=='white':
                    color = 'black'
                elif background=='black':
                    color = 'white'

                for i,(x,y) in enumerate(cell_centers):
                    ax.text(x,y,str(i),
                            color=color,
                            fontsize=12,
                            ha='center',
                            va='center')

        # Adjust axes
        buf = (0.01*radii[-1])
        xmin = min(0, # image left edge
                   # upper left corner of grid
                   radii[0]*np.cos(np.deg2rad(thetas[0]))+A[0]-buf,
                   # lower left corner of grid
                   radii[0]*np.cos(np.deg2rad(thetas[-1]))+A[0]-buf)
        xmax = A[0]+radii[-1]+buf # right side of grid
        ymax = min(0, # top of image
                   # top corner of grid
                   radii[-1]*np.sin(np.deg2rad(thetas[0]))+A[1]-buf)
        ymin = max(im.shape[0], # bottom of image
                   # bottom corner of grid
                   radii[-1]*np.sin(np.deg2rad(thetas[-1]))+A[1]+buf)
        ax.set_xlim(xmin,xmax)
        ax.set_ylim(ymin,ymax)

        # turn off axes ticks
        ax.set_xticks([])
        ax.set_yticks([])
        #ax.set_axis_off()

        # Title the plot
        if self.method_=='pca':
            if comps=='all':
                plot_title = "Loadings on all PCs".format(comps)
            elif len(comps)==1:
                plot_title = "Loadings on PC {}".format(comps)
            else:
                plot_title = "Loadings on PCs {}".format(comps)
        elif self.method_=='lda':
            if comps=='all':
                plot_title = "Loadings on all LDs".format(comps)
            elif len(comps)==1:
                plot_title = "Loadings on LD {}".format(comps)
            else:
                plot_title = "Loadings on LDs {}".format(comps)

        if title is not None:
            plot_title += (' - '+title)

        fig.suptitle(plot_title,fontsize=14)

        # Label rows and columns with channel and statistic
        for st,row in enumerate(axes):   # rows: [mean,std]
            for channel,ax in zip(['red','green','blue'],row): # cols: [r,g,b]
                if st==1: # bottom row
                    ax.set_xlabel(channel+' channel',color=channel,fontsize=12)
                if channel=='red': # first column
                    if st==0:
                        ax.set_ylabel("mean px. value",fontsize=12)
                    elif st==1:
                        ax.set_ylabel("stdev of px. values",fontsize=12)

        # adjust plot padding and spacing between plots
        plt.subplots_adjust(hspace=0.001,
                            wspace=0.001,
                            # padding on sides
                            bottom=0.04,
                            left=0.04,
                            right=0.97,
                            top=0.95,
                            )

        # Add colorbar
        cbar = fig.colorbar(cbar_dummy,
                            ax=axes.ravel().tolist(),#cax=cax,
                            ticks=[0.,1.], # add ticks at min and max
                            pad=0.005, # padding b/w plot and cbar
                            fraction=0.06) # how much of the image to use
        cbar.ax.set_yticklabels(['least', 'most']) # relabel ticks

        plt.show()
