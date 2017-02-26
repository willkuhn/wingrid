
#Copyright Will Kuhn 2016-2017

#IMPORT PACKAGES, DEFINE CUSTOM FUNCTIONS, & SET DIRECTORIES:
#------------------------------------------------------
#PACKAGES
import os
import numpy as np
import scipy.ndimage as nd
import scipy.misc, scipy.spatial
import random
import skimage.color, skimage.transform, skimage.feature, skimage.measure
import skimage.morphology as morph
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from matplotlib.collections import PatchCollection
from matplotlib import lines
import cPickle as pickle
from sklearn.decomposition import PCA
from sklearn.lda import LDA
from sklearn.preprocessing import normalize
from astropy.io import ascii
from collections import Counter #For binarizing image

#IMPORT CUSTOM FUNCTIONS
def maskImage(image):
    """Creates basic mask over wings in a image.
    Parameters
    ----------
    image : array
        RGB input image

    Returns
    -------
    output : array
        bool array where background pixels are false, foreground are True
    """
    mask = image.copy()
    min_size = int(mask.size*0.05)#Objects sized <5% of image will be removed
    #Get most common pixel value (i.e. background value), not alway 0.
    counts = Counter(list(np.ravel(image)))
    background = counts.most_common(1)[0][0]

    mask[mask[:,:] > background] = 1.#Any px greater than thresh goes to 1
    mask[mask[:,:] <= background] = 0.
    mask = mask.astype(np.bool)
    mask = morph.remove_small_objects(mask,min_size=min_size)#Delete small components
    mask = nd.binary_fill_holes(mask)#Fill holes in mask
    return mask

def color2chrom(image):
    """Convert an RGB image pixel values to chromatic coordinates.
    Parameters
    ----------
    image : ndarray
        RGB input image

    Returns
    -------
    output : ndarray
        3-channel image with transformed pixel values.

    Notes
    -----
    Floating point (0.0-1.0) RGB color pixel values (R,G,B) are converted to
    chromatic coordinates (r,b,g) with the transformation:
        r = R / (R+G+B)
        g = G / (R+G+B)
        b = B / (R+G+B)

    References
    ----------
    [1] Gillespie AR, Kahle AB, Walker RE (1987) Color enhancement of highly
        correlated images. II. Channel ratio and “chromaticity” transformation
        techniques. Remote Sens Environ 22: 343–365.
        doi:10.1016/0034-4257(87)90088-5.
    [2] Woebbecke DM, Meyer GE, Von Bargen K, Mortensen DA (1995) Color indices
        for weed identification under various soil, residue, and lighting
        conditions. Trans ASAE 38: 259–269.
    [3] Sonnentag O, Hufkens K, Teshera-Sterne C, Young AM, Friedl M, Braswell
        BH, et al. (2012) Digital repeat photography for phenological research
        in forest ecosystems. Agric For Meteorol 152: 159–177.
        doi:10.1016/j.agrformet.2011.09.009.
    """
    try:
        img = image.astype(np.float)
        chans = cv2.split(img) #Split image into channels
        tot = sum(chans) #Sum up pixel values across channels
        chans_new = map(lambda channel: channel/tot,chans) #transform channels
        new = cv2.merge(chans_new) #merge transformed channels
        new[np.isnan(new)] = 0. #convert any pixel values where x/0. (NaNs) to 0.
#        new *= 255.0/new.max() #convert pixel values from (0-1.) to (0.-255.)
#        new = new.astype(np.uint8)
        return new
    except: print "Error during conversion to chromaticity."


def findHullPoints(image):
    """Get indices of pixels of a convex hull image.
    Parameters
    ----------
    image : array
        input image

    Returns
    -------
    output : array
        list of (x,y) coordinates

    Notes
    -----
    See skimage.morphology.convex_hull_image for more details.
    """
    hull = morph.convex_hull_image(image)#Compute hull image
    hull = skimage.feature.canny(hull)#Get edges of hull
    (y,x) = np.where(hull)#Get edge coords
    return np.transpose((x,y))

def findMinTriangle(image):
    """Find minimum enclosing triangle for the object in an image.
    Parameters
    ----------
    image : array
        input image

    Returns
    -------
    output : array
        list of (x,y) coordinates

    Notes
    -----
    See cv2.minEnclosingTriangle for more details.
    """
    pts = findHullPoints(image) #Get coords of pixels in convex hull image
    pts = np.asarray([[i] for i in pts])#Reshape for minEclosingTriangle
    area,tri = cv2.minEnclosingTriangle(pts)#Min triangle
    return tri.reshape((3,2))

def slope(pt0,pt1):
    '''Returns the slope (in degrees) of line between *pt0* and *pt1*.'''
    x0,y0 = pt0
    x1,y1 = pt1
    den = (x1-x0)
    if den==0.:
        s = 0.
    else:
        s = (y1-y0)/den
        s = np.arctan(s)
    return np.rad2deg(s)

def intercept(slope,point):
    '''Returns the intercept of a line with *slope* that intersects *pt*.'''
    x,y = point
    return y - (slope*x)

def rotateAboutPoint(point,centerPoint,angle):
    '''Returns coordinates for *point* rotated by *angle* around *centerPoint*.
    Parameters
    ----------
    point : list or array
        point to be rotated in form (x,y)
    centerPoint : list or array
        point in form (x,y) about which *point* will be rotated
    angle : float
        angle in radians

    Returns
    -------
    output : list
        (x,y) coordinate of rotated point
    '''
    x,y = point
    xCent,yCent = centerPoint
    xRot = xCent + np.cos(angle) * (x - xCent) - np.sin(angle) * (y - yCent)
    yRot = yCent + np.sin(angle) * (x - xCent) + np.cos(angle) * (y - yCent)
    return xRot,yRot

def aTan(x,y): #Does arcTan(y/x) even when x=0
    if x==0.: return 0.
    else: return np.arctan(y/x)

def rescaleLandmarks(lms,size0,size1):
    """Returns landmarks that have been rescaled according to size parameters.
    Parameters
    ----------
    lms : list or array
        (x,y) coordinate tuples of landmarks
    size0 : tuple
        (rows,cols[,dim]) of original image; values for dim are ignored
    size1 : tuple
        (rows,cols[,dim]) of resized image; values for dim are ignored

    Returns
    -------
    lms_rescaled : array
        (x,y) coordinate tuples, rescaled by proportion resc_size/orig_size
    """
    if 2>len(size0) or len(size0)>3 or 2>len(size1) or len(size1)>3:
        raise RuntimeError('dims0 and dims1 must be 2- or 3-length tuples.')
    s0=np.asarray(size0[:2],dtype=float)
    s1=np.asarray(size1[:2],dtype=float)
    return (s1/s0) * np.asarray(lms)

def resizedImageAndLandmarks(img, landmarks, new_size):
    '''Returns an image that is resized and a set of landmarks that are
    rescaled according to *new_size*.
    Parameters
    ----------
    img : array
        Image array
    landmarks : list or array
        (x,y) coordinate tuples of landmarks on the input image
    new_size : int,float, tuple or list
        Parameter by which image should be resized & landmarks rescaled
        If int, treated as desired width of resized image
        If float, treated as proportion for resizing image
        If (row,col) tuple or list, treated as desired size of resized image
        Landmarks are rescaled following the same interpretation

    Returns
    -------
    img1 : PIL.Image.Image instance
        Output image, resized to *new_size*
    landmarks1 : array
        (x,y) coordinate tuples, rescaled to proportion new_size/orig_size
    '''
    #Get image size:
    if hasattr(img,'shape'): #For numpy array
        size0 = img.shape[:2]
    elif hasattr(img,'size'): #For PIL image object
        size0 = img.size
    else:
        RuntimeError('Error reading image size/shape.')
    #Interpret *new_size*:
    if type(new_size) == int: #Resize by image width
        w = float(new_size)
        size1 = tuple((w/size0[0] * np.asarray(size0)).astype(np.int))
    elif type(new_size) == float: #Proportional resizing
        size1 = tuple((new_size * np.asarray(size0)).astype(np.int))
    elif type(new_size) in (tuple,list):
        size1 = tuple(new_size)[:2]
    else:
        RuntimeError('Error interpreting new_size: %s.') % new_size

#    img1 = img.resize(size1)
    h,w,d = img.shape
    img1 = scipy.misc.imresize(img,(size1[0],size1[1],d))
    landmarks1 = rescaleLandmarks(landmarks,size0,size1)
    return (img1,landmarks1)

def checkRotate(image,triangle,cornerPts,angle):
    '''Rotates wings so that anterior of forewing is horizontal, while
    padding sides as necessary to prevent cropping the wings during rotation, then
    conditions image into a square.

    Parameters
    ----------
    coordsList : array
        bank of possible coordinates
    radii : list
        distance from (0,0) for each coordinate in *coordsList*
    thetas : list
        angle above 0 radians for each coordinate in *coordsList*
    thetaU : float
        theta in radians corresponding to the upper bounding ray
    thetaD : float
        theta in radians corresponding to the lower bounding ray
    rL : float
        radius from (0,0) corresponding to left bounding arc
    rR : float
        radius from (0,0) corresponding to right bounding arc

    Returns
    -------
    inPxs : list
        coordinates from *coordsList* that fall within polygon bounds
    '''
    triangle = np.round(triangle)
    A,B,C = triangle
    #Rotate corner points:
    pts = np.asarray([rotateAboutPoint(i,A,np.deg2rad(-angle)) for i in cornerPts])
    #Pad or crop left:
    xMin = round(min(pts[:,0]))
    if (xMin<A[0]) & (xMin<0): #Pad left
        image = np.pad(image,((0,0),(int(-xMin),0),(0,0)),'constant',constant_values=0)
        triangle = triangle - (xMin,0)
        pts = pts - (xMin,0)
    elif (xMin<A[0]) & (xMin>0): #Crop left
        image = image[:,xMin:]
        triangle = triangle - (xMin,0)
        pts = pts - (xMin,0)
    elif (xMin>A[0]) & (A[0]<0): #Pad left
        image = np.pad(image,((0,0),(int(-A[0]),0),(0,0)),'constant',constant_values=0)
        triangle = triangle - (A[0],0)
        pts = pts - (A[0],0)
    elif (xMin>A[0]) & (A[0]>0): #Crop left
        image = image[:,A[0]:]
        triangle = triangle - (A[0],0)
        pts = pts - (A[0],0)
    #Pad right:
    xMax = round(max(pts[:,0]))
    h,w,d = image.shape
    if xMax > w:
        image = np.pad(image,((0,0),(0,int(xMax-w)),(0,0)),'constant',constant_values=0)
    #Pad bottom:
    yMax = round(max(pts[:,1]))
    if yMax > h:
        image = np.pad(image,((0,int(yMax-h)),(0,0),(0,0)),'constant',constant_values=0)
    #Pad top:
    image = np.pad(image,((10,0),(0,0),(0,0)),'constant',constant_values=0)
    triangle = triangle + (0,10)
    pts = pts + (0,10)
    #Rotate image and triangle:
    h,w,d = image.shape
    A,B,C = triangle
    M = cv2.getRotationMatrix2D(tuple(A),angle,1)
    image = cv2.warpAffine(image,M,(w,h))
    triangle = np.asarray([rotateAboutPoint(i,A,np.deg2rad(-angle)) for i in triangle])
    A,B,C = triangle
    #Crop top, leaving a slight padding:
    a = A[1]-5
    if a > 0:
        image = image[a:,:]
        triangle = triangle - (0,a)
        pts = pts - (0,a)
    h,w,d = image.shape
    #Crop right:
    if w>pts[2,0]:
        image = image[:,:pts[2,0]]
    #Make square:
    h,w,d = image.shape
    if h<w: #Pad bottom
        image = np.pad(image,((0,int(w-h)),(0,0),(0,0)),'constant',constant_values=0)
    if h>w: #Crop bottom
        image = image[:w,:]
#    plt.imshow(image)
#    plt.plot(pts[:,0],pts[:,1],'bo')
    return image,triangle

def buildSquareWithGrid(image,mask,n_lines=8,n_circles=5,sq_size=500):
    #Apply mask:
    img0 = image * mask[:,:,np.newaxis]
    #Convex hull:
    pts = findHullPoints(mask)#Get coords of pixels in convex hull image
    tri = findMinTriangle(mask)#Get min enclosing triangle from mask
    #Define vertices A,B and C of triangle:
    a = np.argmin(tri[:,0])#vertex a has the lowest x-value
    b = np.argmin(tri[:,1])#vertex b has the lowest y-value (in image coord system)
    c = list(set((0,1,2)) - set((a,b)))[0]#the remaining vertex is c
    A,B,C = [tri[i] for i in a,b,c]
    tri = np.array((A,B,C))
    #Define radii for arcs:
    dists = [scipy.spatial.distance.euclidean(A,i) for i in random.sample(pts,500)]
    radii = np.arange(min(dists),max(dists),(max(dists)-min(dists))/n_circles)
    radii = np.append(radii,(max(dists)))

    #Rotate & crop image and triangle:
    thetaUp = slope(A,B)
    thetaDown = slope(A,C)
    pts1 = np.float32([[radii[0]*np.cos(np.deg2rad(thetaUp)),radii[0]*np.sin(np.deg2rad(thetaUp))],
            [radii[0]*np.cos(np.deg2rad(thetaDown)),radii[0]*np.sin(np.deg2rad(thetaDown))],
            [radii[-1]*np.cos(np.deg2rad(thetaUp)),radii[-1]*np.sin(np.deg2rad(thetaUp))],
            [radii[-1]*np.cos(np.deg2rad(thetaDown)),radii[-1]*np.sin(np.deg2rad(thetaDown))]])+A
    img0,tri = checkRotate(img0,tri,pts1,thetaUp)#Rotate, after padding as needed
    img0,tri = resizedImageAndLandmarks(img0, tri, sq_size)#Resize to sq_size
    #Convert RGB to Chromatic Coordinates:
    img0 = color2chrom(img0)

    #For grid sampling:
    A,B,C = tri
    radii = radii*((sq_size-A[0])/radii[-1])#Adjust radii to max sq_size
    thetaUp = slope(A,B)
    thetaDown = slope(A,C)
    if thetaDown<0:
        thetaDown = 180+thetaDown
    thetas = np.arange(thetaUp,thetaDown,(thetaDown-thetaUp)/n_lines)
    thetas = np.append(thetas,(thetaDown))
    return img0,radii,thetas,tri

def plotGrid(image,radii,thetas,tri,plotTriPts='False',axis='on'):
    '''BUILD FIGURE'''
    A,B,C = tri
    fig, ax = plt.subplots()
    ax.imshow(image)
    #Draw circles as wedges using PatchCollection:
    patches = []
    for r in radii:
        patches.append( Wedge(A, r, thetas[0],thetas[-1], width=0.001) )
    p = PatchCollection(patches,facecolor='none',edgecolor='b')
    ax.add_collection(p)
    #Draw lines:
    for t in np.deg2rad(thetas):
        x0 = (radii[0] * np.cos(t)) + A[0]
        y0 = (radii[0] * np.sin(t)) + A[1]
        x1 = (radii[-1] * np.cos(t)) + A[0]
        y1 = (radii[-1] * np.sin(t)) + A[1]
        line = lines.Line2D((x0,x1),(y0,y1),c='b')
        ax.add_line(line)
    if plotTriPts=='True':
        #Draw points of triangle:
        tris = np.append(tri,np.array([A]),axis=0)
        ax.plot(tris[:,0],tris[:,1],'bo',markersize=8)
        #Label triangle points:
        for lbl,coord in zip(('A','B','C'),tri):
            ax.annotate(lbl, xy=(0,0), xytext=coord+(10,10), color='b')
    plt.axis(axis)
#    plt.show()

def pixelsInPolygon(coordsList,radii,thetas,thetaU,thetaD,rL,rR):
    '''Selects coordinates from *coordsList* that fall within a polygon bounded
    above & below by a ray, & left & right by an arc.

    Parameters
    ----------
    coordsList : array
        bank of possible coordinates
    radii : list
        distance from (0,0) for each coordinate in *coordsList*
    thetas : list
        angle above 0 radians for each coordinate in *coordsList*
    thetaU : float
        theta in radians corresponding to the upper bounding ray
    thetaD : float
        theta in radians corresponding to the lower bounding ray
    rL : float
        radius from (0,0) corresponding to left bounding arc
    rR : float
        radius from (0,0) corresponding to right bounding arc

    Returns
    -------
    inPxs : list
        coordinates from *coordsList* that fall within polygon bounds
    '''
    inPxs = []
    for i in range(len(coordsList)):
        if bool(radii[i]>rL) & bool(radii[i]<=rR) & bool(thetas[i]<=thetaD) & bool(thetas[i]>thetaU):
            inPxs.append(coordsList[i])
    return inPxs

def applyToNonzeros(List,func):#Removes zeros from list, then if list is empty
#returns 0., or applies function to list of not empty
    l = [i for i in List if i>0]
    if len(l)>0: return func(l)
    else: return 0.#np.NaN

def dist(pt0,pt1):#Euclidean distance between 2 points
    return np.sqrt((pt1[0]-pt0[0])**2+(pt1[1]-pt0[1])**2)

def extractGridValues(image,triangle,radii,thetas,n_lines=8,n_circles=5,keep_polyPts='False'): #Extract features from image given grid parameters
    #List of pixel indices to sample:
    gridPoints = []
    h,w,d = image.shape
    for x in range(0,w,2):
        for y in range(0,h,2):
            gridPoints.append((x,y))
    A = triangle[0]
    ths = [slope(A+1,pt+1) for pt in np.float32(gridPoints)]
    rs = [dist(A+1,pt+1) for pt in np.float32(gridPoints)]
    #Make list of indices for all combos of (theta,radius):
    sets = []
    for i in range(len(thetas)-1):
        for j in range(len(radii)-1):
            sets.append((i,j))
    #Find the gridPoints within each grid polygon:
    if keep_polyPts:
        global polyPts
    polyPts = range(len(sets))
    for i,(t,r) in enumerate(sets):
        polyPts[i] = pixelsInPolygon(gridPoints,rs,ths,thetas[t],thetas[t+1],radii[r],radii[r+1])
    extractedValues = [[],[],[]]
    for d in range(3):
        for p in range(len(polyPts)):
            z = []
            for i,j in polyPts[p]:
                z.append(image[j,i,d])
            extractedValues[d].append(z)
    meanValues = np.zeros((3,n_lines*n_circles))
    for d in range(3):
        for i in range(n_lines*n_circles):
            meanValues[d,i] = applyToNonzeros(extractedValues[d][i],np.mean)
    meanValues = np.ravel(meanValues)
    stdevValues = np.zeros((3,n_lines*n_circles))
    for d in range(3):
        for i in range(n_lines*n_circles):
            stdevValues[d,i] = applyToNonzeros(extractedValues[d][i],np.std)
    stdevValues = np.ravel(stdevValues)
    features = np.concatenate((meanValues,stdevValues))
    v = features#normalize(features)[0]
    return list(v)

#Directories
base_path = 'D:\Dropbox\Mimicry'
image_path = 'D:\\Dropbox\\Mimicry\\Resized images\\Butterflies' #Contains source images
grid_exam_path = 'D:\Dropbox\Mimicry\Grid examples' #Gridded images are exported here
filenames = [os.path.join(image_path,i) for i in os.listdir(image_path) if i.endswith('.png')]

#%%


#IMAGE PROCESSING & FEATURE EXTRACTION:
#--------------------------------------
#Find previously-saved data, and exclude previously-processed images from cue:
if os.path.isfile(os.path.join(base_path,'extractedData.pkl')):
    featureDict = pickle.load(open(os.path.join(base_path,'extractedData.pkl'),'rb'))
    cue = [fn for fn in filenames if os.path.basename(fn).split('.')[0] not in featureDict.keys()]
else:
    featureDict = {} #If extractedData doesn't exists, initiate new dictionary
    cue = filenames #And include all filenames in processing cue
print 'Specimens remaining for pre-processing: ',len(cue)

#Process and extract coefficients from images in batch:
for fn in cue: #Do this for every file in 'cue'
    f = os.path.basename(fn).split('.')[0]
    try:
        img = nd.imread(fn) #Read image file
        img = img[:,:,:3] #Take only R, G, & B channels
        h,w,d = img.shape
        img = scipy.misc.imresize(img,(int(1000*h/w),1000,d)) #Resize to 1000px width
        mask = maskImage(skimage.color.rgb2gray(img)) #Create mask for grayscaled img
        #Reconfigure image, then calculate grid:
        img1,radii,thetas,tri = buildSquareWithGrid(img,mask)
        #Plot image with grid overlaid:
        plotGrid(img1,radii,thetas,tri,axis='off')
        plt.savefig(os.path.join(grid_exam_path,os.path.basename(fn))) # Save plot
        plt.close() # Don't display plot, just save it to file

        #Extract coefficients from image according to grid:
        featureDict[f] = grid.extractGridValues(img1,radii,thetas)
    except: featureDict[f] = None #In case of image-processing or extraction error

#Save (pickle) updated featureDict:
pickle.dump(featureDict,open('extractedData.pkl','wb'),-1)

#%%


#GET SPECIES METADATA:
#---------------------
#('species metadata.dat' should be a tab-delimited table with 1st row as column names #and 1st column, 'File', as filenames that match names of images. Other columns can be #'Species', 'Sex','Country',etc.)

l = ascii.read(os.path.join(base_path,'species metadata.dat')) #Import metadata file
#Make dictionary of metadata:
mdata = {}
for i in range(len(l)):
    tempDict = {}
    for key in l.keys()[1:]:
        tempDict[key]=l[key][i]
    mdata[l['File'][i]] = tempDict
del tempDict,l

#Get image filenames from featureDict:
target_full = featureDict.keys() #Get full list of individuals

#Apply filtering to your individuals (or not):
target = target_full
#target = [i for i in target_full if mdata[i]['Country']=='Peru']

#Make data matrix from only the values in 'target':
data = np.array([featureDict[i] for i in target]) #Make data matrix
data = normalize(data,axis=1)#Normalize data coefficients

#Get species names for each individual in 'target':
labels = [mdata[i]['Species'] for i in target]

#%%


#PRINCIPAL COMPONENT ANALYSIS (PCA):
#-----------------------------------
title = 'PCA: title...'
pca = PCA(n_components=2,whiten=True) #Set up PCA
pcaT = pca.fit_transform(data) #Fit & simulteneously tranform you data with PCA
var = pca.explained_variance_ratio_ #Get % variances for PCs
loadings = pca.components_ #PC loading values (aka Eigen-vectors)
  # Rows are PCs (starting with PC1), and columns are the relative contribution
  # that each feature plays in that PC.

#%%


#PLOT PCA:
#---------
fig = plt.figure(figsize=(10.,10.)) #Set up plot
ax = fig.add_subplot(111,axisbg='w') #'k' for black background, 'w' for white
ax.plot(pcaT[:,0],pcaT[:,1],'o',alpha=0.3) #Plot PCA points
#Add text to each point (text can be 'labels' or 'target' depending on your needs):
for pt,name in zip(pcaT,labels):
    ax.annotate(name,xy=(0,0),xycoords='data',
                xytext=pt,textcoords='data',color='k',fontsize=8)

ax.axis('normal')
#Set axes labels:
ax.set_xlabel('PC 1 (%.2f %% of total variance)' % (var[0]*100))#,weight='bold',fontsize='xx-large');
ax.set_ylabel('PC 2 (%.2f %% of total variance)' % (var[1]*100))#,weight='bold',fontsize='xx-large');
fig.suptitle(title) #Title the figure
plt.show()

#%%


#PLOT PC LOADINGS - AS HORIZONTAL BAR CHARTS:
#-------------------------------------------
"""This plots the loadings for each coefficient for PC1 and PC2 as vertically-oriented bar charts"""
#Coefficient labels: channels (r,g,b), mean (m) or stdev (s), & grid number
coeffLabels = np.array([k+i+str(j) for k in ('rgb') for i in ('ms') for j in range(40)])

# Min and max values for axes bounds
mmin = loadings.min()-0.1*loadings.std()
mmax = loadings.max()+0.1*loadings.std()

# Set up plot and subplots
fig,axes = plt.subplots(nrows=1,ncols=2,facecolor='w',figsize=(4,12))

# Plot loadings for each PC as a separate bar plot:
for ax,l,pc,v in zip(axes,loadings,['PC1','PC2'],var):
    ax.barh(bottom=range(len(l)), # y-value for each bar
            width=l, # length of each bar
            height=0.6, # how wide to make each bar
            color='k',linewidth=0)

    # Set/adjust x-axis:
    ax.set_xlim([mmin,mmax]) # sets x-axis limits
    ax.set_xlabel('{} ({:.2f}%)'.format(pc,v*100)) # makes x-axislabels
    # rotate the x-axis labels by 45 deg so they don't overlap
    plt.setp(ax.xaxis.get_majorticklabels(),rotation=45)

    # Set/adjust y-axis:
    ax.set_ylim([-2,len(l)+2]) # adds a little space above 1st bar and below last bar
    # Label the y-ticks using the coeffLabels:
    ax.set_yticks(np.arange(len(l)))
    ax.set_yticklabels(coeffLabels)

    # Adjust both axes:
    ax.tick_params(axis='both',
                   length=0, # removes tick marks from plot
                   labelsize=6 # sets fontsize for x- and y-labels
                   )

plt.tight_layout()
plt.show()

#%%


#PLOT PC LOADINGS - AS 2D PLOT:
#-----------------------------
"""This plots loadings for PC1 and PC2 in 2D space and highlights (with circles and labels) the top `n_hc` coeffs that make the highest contribution to those PCs. Highlighting circles range from purple (highest contribution) to yellow (relatively lowest contribution)."""
#Coefficient labels: channels (r,g,b), mean (m) or stdev (s), & grid number
coeffLabels = np.array([k+i+str(j) for k in ('rgb') for i in ('ms') for j in range(40)])

n_hc = 10 # no. of highest-contributing coeffs to highlight (must be n_hc<=n_coeffs)
n_coeffs = loadings.shape[1]

# plot bound
mmax = np.abs(loadings).max()+0.2*loadings.std()

# Calculate highest-contributing loading values by their distance from origin
loading_dists=np.zeros(n_coeffs)
for i in range(n_coeffs):
    # calculate Euclidean distance from loadings to origin (0,0):
    loading_dists[i] = np.sqrt(loadings[0,i]**2+loadings[1,i]**2)
# get indices for `loading_dists` sorted highest to lowest
ind = np.argsort(loading_dists)[::-1]
hc = ind[:n_hc] # highest contributors
print '{} highest contributers: {}'.format(n_hc,coeffLabels[hc])

# set up figure
fig = plt.figure()
fig.suptitle('Loading plot')
ax = fig.add_subplot(111,axisbg='w')

# draw lines at origin, scale plot, label axes
ax.vlines(0.,-mmax,mmax,colors='k',alpha=0.3)
ax.hlines(0.,-mmax,mmax,colors='k',alpha=0.3)
ax.set_xlim(-mmax,mmax)
ax.set_ylim(-mmax,mmax)
ax.set_xlabel('Loadings on PC1 ({:.1f}% of total var)'.format(var[0]*100))
ax.set_ylabel('Loadings on PC2 ({:.1f}% of total var)'.format(var[1]*100))

# plot loading values
ax.plot(*loadings, marker = 'o', linestyle='None', alpha=0.3, label='Loadings')

# circle highest-contributing loading values
color = iter(plt.cm.plasma(np.linspace(0,1,len(hc))))
for i,c in zip(hc,color):
    ax.plot(*loadings[:,i], c=c, marker='o', markersize=10,
            linestyle='None', markeredgewidth=2, markerfacecolor='none',
            markeredgecolor=c, label='Highest contributers', alpha =0.6)

# annotate the highest-contributing loading values
for h in hc:
    ax.annotate(coeffLabels[h],# Annotate with coefficient label
            xy=(0,0),xycoords='data', #dummy coords
            xytext=loadings[:,h],textcoords='data')

plt.show()
