import numpy as np
import cv2
import scipy.io as sio

def read_annotations(mat_path):
    """
    Read image name and car bounding box.

    @param  mat_path:   path to matrix of annotations

    @return X:  list of car annotations [car_name, x1, y1, x2, y2]
    """
    mat_contents = sio.loadmat(mat_path)
    car = mat_contents['annotations']
    X = []
    for i in range(car.shape[1]):
        l = []
        for j in range(5):
            if j == 0:
                l.append(car[0,i][j][0])
            else:
                l.append(car[0,i][j][0][0])
        X.append(l)
    return X

def read_dataset(mat_path, img_size):
    """
    Store cropped and resized car image.

    @param  mat_path:   path matrix of annotations
    @param  img_size:   resize all images to this size

    @return X:  array of flattened car images [num_examples, num_features]
    """
    print('Loading Dataset...')
    X_list = read_annotations(mat_path)
    num_images = len(X_list)
    X = np.zeros((num_images, img_size[1], img_size[0], 3), dtype=np.uint8)
    for i, img_info in enumerate(X_list):
        img_name = img_info[0]
        bb = img_info[1:]
        img = cv2.imread('dataset/' + img_name)
        # crop image
        img = img[bb[1]:bb[3], bb[0]:bb[2]]
        img = cv2.resize(img, img_size)
        X[i] = img
    return X
        

def main():
    """ Main Function. """
    # read all images in dataset
    X = read_dataset('dataset/cars_annos.mat', img_size=(224,128))
    print('X:', X.shape)


if __name__ == '__main__':
    main()
