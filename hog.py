"""
Histogram of Orientated Gradients.

@author: Gary Corcoran
@date_created: Nov. 23rd, 2017
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
            ang = angles[i]
            # multiple gradient magnitude by scalar for viewing
            b *= 10
            x = int(round(b*cos(radians(ang))))
            y = int(round(b*sin(radians(ang))))
            # draw line in direction of gradient
            cv2.line(img_disp, (center[0]-y,center[1]-x),
                    (center[0]+y,center[1]+x), (0, 0, 180))

    def visualize(self):
        """
        Visualize HoG features.

        @return img_disp:   input image with HoG features overlayed
        """
        if self.h is None:
            print('Please compute HoG features before displaying.')
            return
        cellSize = self._cellSize[0]
        blockSize = self._blockSize[0]
        img_disp = self.img.copy()
        rows, cols = img_disp.shape[:2]
        # format HoG descriptor
        h = np.reshape(self.h, (-1, 2*blockSize//cellSize, self._nbins))
        count = 0
        # draw HoG features
        for x in range(cellSize, cols, 2*cellSize):
            for y in range(cellSize, rows, 2*cellSize):
                hist = h[count,0] + h[count,1] + h[count,2] + h[count,3]
                self._display_hist(img_disp, (x,y), hist) 
                count += 2
        return img_disp

def test():
    """ Test Function. """
    # read input image
    img_path = '/home/gary/opencv/samples/data/pic6.png'
    img = cv2.imread(img_path)

    # HOG parameters
    hog_params = {'winSize': (224, 320), 'blockSize': (32, 32),
            'blockStride': (16, 16), 'cellSize': (16, 16),
            'nbins': 9}

    # create HoG object
    hog = HOG(**hog_params)
    hog.set_image(img)
    h = hog.compute()
    print('h:', h.shape)
    img_disp = hog.visualize()
    cv2.imshow('HoG Features', img_disp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    test()
