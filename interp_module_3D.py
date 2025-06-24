import numpy.random as nprand
import matplotlib.pyplot as plt
import skimage.io as io
import random
import sys
import copy
import numpy as np

# Initialize random interp artifact location
def location(hgt):
    initial_row = random.randint(hgt//6, (hgt*5)//6)
    width_row = random.randint(3, 45)
    return initial_row, width_row

def interpolationArtifactLocation(I, initial_row, width_row, wid):
    for i in range(initial_row, initial_row+width_row-1):
        for j in range(wid-1):
            I[i][j] = 0

# Linear Interpolation
def interpolationArtifact(I, initial_row, width_row, wid):
    for k in range(wid):
        for i in range(initial_row, initial_row+width_row):
            y0 = I[initial_row-1][k]
            y1 = I[initial_row+width_row][k]
            x0 = initial_row-1
            x1 = initial_row+width_row+1
            x = i
            y = ((y0*(x1-x))+(y1*(x-x0)))/(x1-x0)
            I[i][k] = y

def addMask(mask, initial_row, width_row, wid):
    for i in range(wid):
        for j in range(initial_row, initial_row+width_row):
            mask[j][i] = 4.0 # mask value was changed from 255.0 to 4.0

def addArtifact(image):
    # Load Image
    I = image
    wid = image.shape[1]
    hgt = image.shape[0]
    mask = np.zeros(shape=[hgt, wid], dtype=np.uint8)

    # Possibility of Two Artifacts:
    lst = [1, 2]
    nArtifacts = nprand.choice(lst, 1, p=[0.65, 0.35])

    if nArtifacts == 2:
        initial_row1, width_row1 = location(hgt)
        initial_row2, width_row2 = location(hgt)
        if initial_row2 not in range(initial_row1, initial_row1+width_row1) and initial_row2+width_row2 not in range(initial_row1, initial_row1+width_row1):
            interpolationArtifactLocation(I, initial_row1, width_row1, wid)
            interpolationArtifactLocation(I, initial_row2, width_row2 // 2, wid)
            interpolationArtifact(I, initial_row1, width_row1, wid)
            interpolationArtifact(I, initial_row2, width_row2 // 2, wid)
            addMask(mask, initial_row1, width_row1, wid)
            addMask(mask, initial_row2, width_row2 // 2, wid)
        else:
            interpolationArtifactLocation(I, initial_row1, width_row1, wid)
            interpolationArtifact(I, initial_row1, width_row1, wid)
            addMask(mask, initial_row1, width_row1, wid)
    else:
        initial_row, width_row = location(hgt)
        interpolationArtifactLocation(I, initial_row, width_row, wid)
        interpolationArtifact(I, initial_row, width_row, wid)
        addMask(mask, initial_row, width_row, wid)
    return I, mask

def addSingleArtifact(image, initial_row, width_row):
    # Load Image
    I = image
    wid = image.shape[1]
    hgt = image.shape[0]
    mask = np.zeros(shape=[hgt, wid], dtype=np.uint8)

    # Possibility of Two Artifacts:
    lst = [1, 2]
    nArtifacts = nprand.choice(lst, 1, p=[0.65, 0.35])

    if nArtifacts == 2:
        if initial_row not in range(initial_row, initial_row+width_row) and initial_row+width_row not in range(initial_row, initial_row+width_row) and (initial_row+50+(width_row//2)) < hgt:
            interpolationArtifactLocation(I, initial_row, width_row, wid)
            interpolationArtifactLocation(I, initial_row+50, width_row // 2, wid)
            interpolationArtifact(I, initial_row, width_row, wid)
            interpolationArtifact(I, initial_row+50, width_row // 2, wid)
            addMask(mask, initial_row, width_row, wid)
            addMask(mask, initial_row+50, width_row // 2, wid)
        else:
            interpolationArtifactLocation(I, initial_row, width_row, wid)
            interpolationArtifact(I, initial_row, width_row, wid)
            addMask(mask, initial_row, width_row, wid)
    else:
        interpolationArtifactLocation(I, initial_row, width_row, wid)
        interpolationArtifact(I, initial_row, width_row, wid)
        addMask(mask, initial_row, width_row, wid)
    return I, mask

def addArtifact3D(image, initial_row1, initial_row2, width_row1, width_row2, nArtifacts):
    # Load Image
    I = image
    wid = image.shape[1]
    hgt = image.shape[0]
    mask = np.zeros(shape=[hgt, wid], dtype=np.uint8)

    # Possibility of Two Artifacts:
    # lst = [1, 2]
    # nArtifacts = nprand.choice(lst, 1, p=[0.65, 0.35])

    if nArtifacts == 2:
        # initial_row1, width_row1 = location(hgt)
        # initial_row2, width_row2 = location(hgt)
        if initial_row2 not in range(initial_row1, initial_row1+width_row1) and initial_row2+width_row2 not in range(initial_row1, initial_row1+width_row1):
            interpolationArtifactLocation(I, initial_row1, width_row1, wid)
            interpolationArtifactLocation(I, initial_row2, width_row2 // 2, wid)
            interpolationArtifact(I, initial_row1, width_row1, wid)
            interpolationArtifact(I, initial_row2, width_row2 // 2, wid)
            addMask(mask, initial_row1, width_row1, wid)
            addMask(mask, initial_row2, width_row2 // 2, wid)
        else:
            interpolationArtifactLocation(I, initial_row1, width_row1, wid)
            interpolationArtifact(I, initial_row1, width_row1, wid)
            addMask(mask, initial_row1, width_row1, wid)
    else:
        # initial_row, width_row = location(hgt)
        interpolationArtifactLocation(I, initial_row1, width_row1, wid)
        interpolationArtifact(I, initial_row1, width_row1, wid)
        addMask(mask, initial_row1, width_row1, wid)
    return I, mask

def addInterp3D(img): # Updated to include binary mask
    img_copy = copy.deepcopy(img)
    mask = np.zeros(img_copy.shape)
    
    width = random.randint(3,60)
    x0 = random.randint(0, img_copy.shape[0]-width)
    x1 = x0+(width-1)
    mask[x0:x1,:,:] = 1.0

    for indx in range(x0,x1):
        y0 = img[x0,:,:]
        y1 = img[x1,:,:]

        img_copy[indx,:,:] = (y0*(x1-indx) + y1*(indx-x0))/(x1-x0)
    return img_copy, mask

def setInterp3D(img,width): # Updated to include input for specifying width of artifact
    img_copy = copy.deepcopy(img)
    mask = np.zeros(img_copy.shape)
    
    x0 = random.randint(1, img_copy.shape[0]-width)
    x1 = x0+(width-1)
    mask[x0:x1,:,:] = 1.0

    for indx in range(x0,x1):
        y0 = img[x0,:,:]
        y1 = img[x1,:,:]

        img_copy[indx,:,:] = (y0*(x1-indx) + y1*(indx-x0))/(x1-x0)
    return img_copy, mask

def addInterp3D_big_artifacts(img): # Updated to include binary mask
    img_copy = copy.deepcopy(img)
    mask = np.zeros(img_copy.shape)
    
    width = random.randint(15,50)
    x0 = random.randint(int(img_copy.shape[0]*0.25), int(img_copy.shape[0]*0.75)-width)
    x1 = x0+(width-1)
    mask[x0:x1,:,:] = 1.0

    for indx in range(x0,x1):
        y0 = img[x0,:,:]
        y1 = img[x1,:,:]

        img_copy[indx,:,:] = (y0*(x1-indx) + y1*(indx-x0))/(x1-x0)
    return img_copy, mask