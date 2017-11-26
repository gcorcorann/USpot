#!/usr/bin/env python3
"""
Histogram of Orientated Gradients.

@author: Gary Corcoran
@date_created: Nov. 23rd, 2017

USAGE: hog.py [<image_source>]

Keys:
    any key   -   exit
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
from math import cos, sin, radians

class HOG():
    """ Histogram of Orientated Gradients Model. """

    def __init__(self, winSize, blockSize, blockStride, cellSize, nbins):
        """
        Initialize parameters.
        
        @param  winSize:        HoG window size
        @param  blockSize:      HoG block size
        @param  blockStride:    HoG block stride
        @param  cellSize:       HoG cell size
        @param  nbins:          HoG number of bins
        """
        self.img = None
        self._winSize = winSize
        self._blockSize = blockSize
        self._cellSize = cellSize
        self._nbins = nbins
        self._hog = cv2.HOGDescriptor(winSize, blockSize, blockStride,
                cellSize, nbins)
        self.h = None  # HoG descriptor

    def _preprocess(self):
        """
        Perform preprocessing on image.

        @modifies self.img:    input image after preprocessing
        """
        # resize input image to same shape as hog window
        self.img = cv2.resize(self.img, self._winSize)

    def set_image(self, img):
        """
        Set input image path.

        @param  img: input image

        @modifies   self.img:  stores image
        """
        self.img = img

    def compute(self):
        """
        Compute HoG features.

        @modifies self.h:   stores HoG descriptor
        
        @return h:  array of HoG features [num_feats x 1]
        """
        if self.img is None:
            print('Error: Please set input image.')
            return False
        self._preprocess()
        h = self._hog.compute(self.img)
        self.h = h
        return h

    def _display_hist(self, img_disp, center, hist):
        """
        Display HoG histogram.

        @param  img_disp:   input display image
        @param  center:     location of HoG descriptor
        @param  hist:       histogram of HoG descriptor

        @modifies   img_disp:   draw HoG descriptor on image
        """
        angles = [x for x in range(0, 180, 180//self._nbins)]
        for i, b in enumerate(hist):
            if b <= 0.1:
                b = 0
            ang = angles[i]
            # multiple gradient magnitude by scalar for viewing
            b *= 50
            x = int(round(b*cos(radians(ang))))
            y = int(round(b*sin(radians(ang))))
            # draw line in direction of gradient
            cv2.line(img_disp, (center[0]-y,center[1]-x),
                    (center[0]+y,center[1]+x), (0, 0, 180))

    def visualize(self, skip_every=1):
        """
        Visualize HoG features.

        @param  skip_every: when display, skip every other HoG feature

        @return img_disp:   input image with HoG features overlayed
        """
        if self.h is None:
            print('Please compute HoG features before displaying.')
            return
        cellSize = self._cellSize[0]
        blockSize = self._blockSize[0]
        winSize = self._winSize
        img_disp = self.img.copy()
        rows, cols = img_disp.shape[:2]
        # format HoG descriptor
        h = np.reshape(self.h, (winSize[0]//cellSize-1, winSize[1]//cellSize-1,
                2*blockSize//cellSize, self._nbins))
        # draw HoG features
        for i, x in enumerate(range(cellSize, cols, skip_every*cellSize)):
            for j, y in enumerate(range(cellSize, rows, skip_every*cellSize)):
                _i = i * skip_every
                _j = j * skip_every
                hist = h[_i,_j,0] + h[_i,_j,1] + h[_i,_j,2] + h[_i,_j,3]
                den = sum(hist)
                if den != 0:
                    hist /= den
                self._display_hist(img_disp, (x,y), hist) 
        return img_disp

def test():
    """ Test Function. """
    import sys
    print(__doc__)
    if len(sys.argv) >= 2:
        # set image path to user's input
        img_path = sys.argv[1]
    else:
        # read default input image
        img_path = '../images/pic6.png'

    # read input image
    img = cv2.imread(img_path)
    # HOG parameters
    hog_params = {'winSize': (224, 320), 'blockSize': (32, 32),
            'blockStride': (16, 16), 'cellSize': (16, 16),
            'nbins': 9}

    # create HoG object
    hog = HOG(**hog_params)
    hog.set_image(img)
    # copmute and visualize HoG features
    h = hog.compute()
    print('HoG feature dimension:', h.shape)
    img_disp = hog.visualize(skip_every=1)
    cv2.imshow('HoG Features', img_disp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    test()
