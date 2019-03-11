import numpy as np
from sklearn import datasets

x,y = datasets.load_breast_cancer(return_X_y=True)
y = y.reshape(-1,1)
data = np.hstack((x,y))
data = data.astype(str)
data = [','.join(d) for d in data]
data = '\n'.join(data)

with open('breast_cancer.txt', 'w') as f:
    f.write(data)
