# -*- coding: utf-8 -*-
"""
Wingrid Insect Wing Color Analysis Package
"""

# Author: William R. Kuhn

# License: GNU GPL License v3

import cv2 # MUST BE version 3.0 or higher
import numpy as np

################################################################################
# HELPER FUNCTIONS

def downscaleIf(image,max_dim=1000,interpolation=cv2.INTER_LINEAR):
    """Reduce large images.

    An large image is downsampled so that it's maximum dimension (h or w) is
    `max_dim`. If the image's max dim does not excede that value, it's returned
    unaltered.

    Parameters
    ----------
    image : array
        Input image. Can be grayscale or color

    max_dim : int (default 1000)
        Maximum dimension (height or width) allowed for image

    interpolation : long (default ``cv2.INTER_LINEAR``)
        Interpolation function used in ``cv2.resize``. See openCV's
        documentation for available options
    """
    h,w = image.shape[:2]
    if max(h,w)>max_dim: # if largest dim is greater than 1000 px
        if max(h,w)==h: # if height is largest dimension
            f = max_dim/float(h)
            dsize = (int(f*w),int(f*h))
        else: # if width is largest dimension
            f = max_dim/float(w)
            dsize = (int(f*w),int(f*h))
        image = cv2.resize(image,dsize=dsize,interpolation=interpolation)
    return image


def pad(image,padding=1,color='black'):
    """Pad an image with white or black pixels."""
    p = int(padding)

    if color=='black':
        value = [0,0,0]
    elif color=='white':
        value = [255,255,255]
    else:
        raise ValueError("Argument for `color` not recognized. Available "
                        "options: ['black','white']")

    return cv2.copyMakeBorder(image,p,p,p,p,cv2.BORDER_CONSTANT,value=value)


def slope(pt1,pt2):
    """Calculate the slope of a line (in deg) between 2 (x,y) points."""
    x0,y0 = pt1; x1,y1 = pt2
    try: return np.rad2deg(np.arctan((y1-y0)/(x1-x0)))
    except ZeroDivisionError: return 0.


def polar2cart(r,theta,originXY=(0,0)): # -> (x,y)
    """Converts polar (r,theta) coordinates to Cartesian (x,y).

    The origin can be specified as a point other than (0,0).
    `theta` should be in degrees."""
    ox,oy = originXY
    th = np.deg2rad(theta)
    return [(r*np.cos(th))+ox, (r*np.sin(th))+oy]


def cart2polar(x,y,originXY=(0,0)): # -> (r,theta)
    """Converts Cartesian (x,y) coordinates to polar (r,theta).

    The origin can be specified as a point other than (0,0). `theta` is
    returned in degrees."""
    ox,oy = originXY
    r = np.sqrt((x-ox)**2 + (y-oy)**2)
    theta = np.arctan((y-oy)/float((x-ox)))
    theta = np.rad2deg(theta)
    return [r,theta]


def safe_div(num,denom):
    """Safely divide 2 arrays, despite zeros.

    Safely do `num`/`denom`, where `num` and `denom` are numpy arrays, and
    when denom contains zeros."""
    d = denom.copy()
    d[d==0.] = np.inf
    return num/d


def RGB2chrom(image,blur=True):
    """Convert an image from RGB to chromatic coordinates.

    8-bit RGB color pixel values (R,G,B) are converted to
    chromatic coordinates (r,b,g) with the transformation:
        r = R / (R+G+B)
        g = G / (R+G+B)
        b = B / (R+G+B)

    Parameters
    ----------
    image : ndarray
        A uint8 RGB input image. Can also be RGBA.

    blur : bool, optional (default True)
        Whether to add 3x3 Gaussian blur to image prior to conversion. This
        helps with noisy images.

    Returns
    -------
    output : ndarray
        3-channel image with transformed pixel values.

    References
    ----------
    [1] Gillespie AR, Kahle AB, Walker RE (1987) Color enhancement of highly
        correlated images. II. Channel ratio and "chromaticity" transformation
        techniques. Remote Sens Environ 22: 343-365.
        doi:10.1016/0034-4257(87)90088-5.
    [2] Woebbecke DM, Meyer GE, Von Bargen K, Mortensen DA (1995) Color indices
        for weed identification under various soil, residue, and lighting
        conditions. Trans ASAE 38: 259-269.
    [3] Sonnentag O, Hufkens K Teshera-Sterne C, Young AM, Friedl M, Braswell
        BH, et al. (2012) Digital repeat photography for phenological research
        in forest ecosystems. Agric For Meteorol 152: 159-177.
        doi:10.1016/j.agrformet.2011.09.009.
    """

    img = image.copy()[:,:,:3]

    if blur:
        img = cv2.blur(img,(3,3)) # Blur image slightly to account for noise

    img = img.astype(np.float)
    channels = cv2.split(img) # Split image into channels
    tot = sum(channels) # Sum pixel values across channels
    channels_new = list(map(lambda channel: safe_div(channel,tot),channels))
    new = cv2.merge(channels_new) # Merge transformed channels
    new[np.isnan(new)] = 0. # Convert any pixel values where x/0. (NaNs) to 0.
    new *= 255.0/new.max() # Scale pixel values from (0-1.) to (0.-255.)
    new = new.astype(np.uint8)
    return new
