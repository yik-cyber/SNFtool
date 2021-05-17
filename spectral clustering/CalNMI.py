import numpy as np
from sklearn import metrics
eps = np.finfo(np.float64).eps


# can be H(X) - H(X|Y). When Y is absolutely independent of X, H(X) - H(X|Y) = 0
def MutualInformation(x, y):
    class_x = np.unique(x)
    class_y = np.unique(y)
    nx = len(x)
    ncx = len(class_x)
    ncy = len(class_y)
    
    # joint pdf
    probxy = np.empty((ncx, ncy))
    probxy.fill(0)
    for i in range(ncx):
        for j in range(ncy):
            # the number of elemts equaling class_x[i] in x, as well as for y
            idx = np.where(x == class_x[i])
            idy = np.where(y == class_y[j])
            probxy[i][j] = 1.0 * len(np.intersect1d(idx, idy)) / nx
    
    # if there exits a better way to construct probx ? 
    probx = np.empty((ncx, ncy))
    for i in range(ncy):
        probx[:, i] = np.sum(probxy, axis=1)
    proby = np.sum(probxy, axis=0) + np.zeros((ncx, ncy))
    
    # eps for log
    eps_m = np.empty((ncx, ncy))
    eps_m.fill(eps)
    
    temp = probxy * np.log2(probxy / (probx * proby) + eps_m)
    print(temp)
    result = np.sum(temp)
    return result


def Entropy(x):
    # calculate the entropy of vector x
    class_x = np.unique(x)
    nx = len(x)
    ncx = len(class_x)

    prob = np.zeros(ncx)
    for i in range(ncx):
        prob[i] = np.sum(x == class_x[i]) / nx
    eps_m = np.empty(ncx)
    eps_m.fill(eps)
        
    return -np.sum(prob * np.log2(prob + eps_m))


def CalNMI(x, y):
    x = np.array(x)
    y = np.array(y)

    return max(0, MutualInformation(x, y) / np.sqrt(Entropy(x) * Entropy(y)))


if __name__ == '__main__':
    A = np.array([1,1,1,1,1,1,2,2,2,2,2,2,3,3,3,3,3])
    B = np.array([1,2,1,1,1,1,1,2,2,2,2,3,1,1,3,3,3])
    print(CalNMI(A,B))
    # compare
    print(metrics.normalized_mutual_info_score(A,B))  