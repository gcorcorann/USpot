import cv2

def main():
    """ Main Function. """
    img_path = 'lena.jpg'
    img = cv2.imread(img_path)
    img = cv2.resize(img, (208, 304))
    
    # HOG parameters
    winSize = (208, 304)
    blockSize = (32, 32)
    blockStride = (16, 16)
    cellSize = (16, 16)
    nbins = 9

    # compute HOG descriptor
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride,
            cellSize, nbins)
    h = hog.compute(img)
    print(h.shape)

if __name__ == '__main__':
    main()
