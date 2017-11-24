import cv2

class HOG():
    """ Histogram of Orientated Gradients Model. """

    def __init__(self, img_path, winSize, blockSize, blockStride, 
            cellSize, nbins):
        """
        Initialize parameters.
        
        @param  img_path:       path to input image
        @param  winSize:        HoG window size
        @param  blockSize:      HoG block size
        @param  blockStride:    HoG block stride
        @param  cellSize:       HoG cell size
        @param  nbins:          HoG number of bins
        """
        self.img_path = img_path
        self.__winSize = winSize
        self.hog = cv2.HOGDescriptor(winSize, blockSize, blockStride,
                cellSize, nbins)

    def __preprocess(self):
        """
        Perform preprocessing on image.

        @return img:    input image after preprocessing
        """
        img = cv2.imread(self.img_path)
        # resize input image to same shape as hog window
        img = cv2.resize(img, self.__winSize)
        return img

    def compute(self):
        """
        Compute HoG features.
        
        @return h:  array of HoG features [num_feats x 1]
        """
        img = self.__preprocess()
        h = self.hog.compute(img)
        return h

def main():
    """ Main Function. """
    img_path = 'lena.jpg'

    # HOG parameters
    hog_params = {'winSize': (208, 304), 'blockSize': (32, 32),
            'blockStride': (16, 16), 'cellSize': (16, 16),
            'nbins': 9}

    # create HoG object
    hog = HOG(img_path, **hog_params)
    h = hog.compute()
    print(h.shape)

if __name__ == '__main__':
    main()
