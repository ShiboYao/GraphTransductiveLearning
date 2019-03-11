import os
import re
import numpy as np



def read_pgm(pgmf):
    """Return a raster of integers from a PGM as a list of lists."""
    assert pgmf.readline() == b'P5\n'
    (width, height) = [int(i) for i in pgmf.readline().split()]
    depth = int(pgmf.readline())
    assert depth <= 255

    size = width*height
    img = []
    img = [ord(pgmf.read(1)) for i in range(size)]
    img = np.array(img)

    return img.reshape((height,width))



if __name__ == "__main__":
    images = []
    labels = []
    
    folders = [f for f in os.listdir(".") if os.path.isdir(f)]
    for fold in folders:
        files = [f for f in os.listdir(fold) if re.match("yaleB.._P00A.*.pgm", f)]
        for f in files:
            with open(fold+'/'+f, 'rb') as File:
                img = read_pgm(File)
                images.append(img)

        labels.extend([fold]*len(files))
    '''            
    print(len(labels))
    print(labels[0])
    print(len(images))
    print(images[0])
    print(images[0].shape)
    '''
    X = [img.flatten() for img in images]
    X = np.array(X)
    X = X.astype(str)
    y = [l[-2:] for l in labels]
    y = np.array(y)
    y = y.reshape(-1,1)
    print(X.shape, y.shape)
    data = np.hstack((X,y))
    data = [','.join(d) for d in data]
    data = '\n'.join(data)

    with open("../yale.txt", 'w') as f:
        f.write(data)
