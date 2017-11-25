import scipy.io as sio

def read_annotations(mat_name):
    """
    Read image name and car bounding box.

    @param  mat_name:   matrix of annotations

    @return X:  list of car annotations [car_name, x1, y1, x2, y2]
    """
    mat_contents = sio.loadmat(mat_name)
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

def main():
    """ Main Function. """
    X = read_annotations('cars_annos.mat')
    print(X)

if __name__ == '__main__':
    main()
