import math

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
img = cv.imread('messi_face.jpg',0)
template = cv.imread('fullPicture.jpg',0)
#img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#template_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

location_score = {}

def R(i1,j1):
    k = 0
    l = 0
    sum_sqr = 0
    for k in range(0,template.shape[0]):
        for l in range(0, template.shape[1]):
            sum_sqr = sum_sqr + math.pow(tamplate[k,l]-img[k+i1,l+j1])

#for i,j in img:
while i < img.shape(0):
    while j < img.shape(1):
        location_score[i,j] = R(i,j)