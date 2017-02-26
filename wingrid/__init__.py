# -*- coding: utf-8 -*-
"""
wingrid Insect Wing Color Analysis Package (:mod: `wingrid`)
============================================================

Wingrid for Python is a package for:
  (1) fitting a stretchable grid to images of insect wings,
  (2) sampling color features from grid-fitted images, and
  (3) analyzing and visualizing color features sampled from images.
"""

# Author: William R. Kuhn

# License: MIT License

from .core import *
from .helpers import *

__version__ = '0.1.0'
__all__ = ['Grid',
           'Analyze',
           'cart2polar',
           'downscaleIf',
           'pad',
           'polar2cart',
           'safe_div',
           'slope']
