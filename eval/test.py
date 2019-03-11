import sys
import time
import pandas as pd
sys.path.append("../model")
from stl import *
from util import *
from sklearn.semi_supervised import label_propagation
from sklearn.semi_supervised import LabelPropagation
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import time
import os #ommit tensorflow debugging info
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


methods = {        
        SVC():{
            'kernel':['linear','poly','rbf'],
            'C':[0.01,0.05,0.1]
            },
        KNeighborsClassifier(weights='distance'):{
            'n_neighbors':[5,10,20]
            }
        }


def classify(data, cname, params, cv=3, scoring='accuracy'):
    x_train = data[0]
    y_train = data[1]
    x_test = data[2]
    y_test = data[3]
    clf = GridSearchCV(cname, params, cv=cv, scoring=scoring)
    start = time.time()
    clf.fit(x_train, y_train)
    print(clf.best_params_)
    y_hat = clf.predict(x_test)
    print("%s Test accuracy:%.2f" %(cname, sum(y_test==y_hat)/len(y_test)))
    print("%s time:%.3f" %(cname, time.time()-start))



if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Specify fname maxi, omega, delta and eta!")
        exit(0)

    fname = sys.argv[1]
    maxi = int(sys.argv[2])
    omega = float(sys.argv[3])
    delta = int(sys.argv[4])
    eta = float('1e-'+sys.argv[5])

    if fname=="20news.txt":
        data = get20news(30000)
        data = data.toarray()
    else:
        data = pd.read_csv("../data/"+fname, header=None)
        data = data.values
    
    X, y_p, y_q = makeData(data, omega=omega, maxi=maxi, 
            shuffle=True, balance=False)
    del data 
    #X = X/255. #normalize when using image
    
    #X_p = X[:len(y_p)]
    #X_q = X[len(y_p):]
    y_p_oneHot = oneHot(y_p, verbose=False)
    y_q_oneHot = oneHot(y_q)
    y_hat = TC(X, y_p, delta=delta, eta=eta) #one hot output
    
    print("%.5f" %(evaluate(y_hat, y_q_oneHot)))
    #print("TC accuracy: %.5f" %(evaluate(y_hat, y_q_oneHot)))

    '''
    y_train = np.hstack((y_p, -np.ones(y_q.shape)))
    #lp_model = label_propagation.LabelSpreading(kernel='knn',n_neighbors=20)
    lp_model = LabelPropagation(kernel='knn', n_neighbors=delta, tol=eta)
    start = time.time()
    lp_model.fit(X, y_train)
    print("LP(KNN) training time: %.3f" %(time.time()-start))
    y_hat_lp = lp_model.transduction_[len(y_p):]
    print("LP(KNN) accuracy: %.5f" %(sum(y_hat_lp==y_q)/len(y_q)))
    '''
    '''
    lp_model = LabelPropagation(kernel='rbf', tol=eta)
    start = time.time()
    lp_model.fit(X, y_train)
    print("LP(rbf) training time: %.3f" %(time.time()-start))
    y_hat_lp = lp_model.transduction_[len(y_p):]
    print("LP(rbf) accuracy: %.5f" %(sum(y_hat_lp==y_q)/len(y_q)))
    '''

    '''
    #benchmark methods with gridsearch
    for clf, params in methods.items():
        classify([X_p, y_p, X_q, y_q], clf, params)
    '''
