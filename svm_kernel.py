# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from qpth.qp import QPFunction

import skcuda.linalg as culinalg
import numpy as np
from numpy import linalg

import cvxopt
import pdb
import cvxopt.solvers
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

import cheb
import time
mod = SourceModule("""
        #include <stdio.h>
        __global__ void Chebyshev(float *x, float *y, float *p_cheb, int Sample){
          int i, j;
          float a,b, chebyshev_result, weight;
          int idx = blockIdx.x  blockDim.x + threadIdx.x *4;
          int jj = threadIdx.x;
          for ( int index=0; index < Sample ; index++){
                chebyshev_result=0;
                p_cheb[index*Sample+jj]=1;
                for( j=0; j<4; j++){    
                    for( i=0; i<4; i++){
                        a = cos(i * acos(x[j+index*idx]));
                        b = cos(i * acos(y[j+index*idx]));
                        chebyshev_result += (a * b);
                    }
                    weight = sqrt(1 - (x[index*idx+j] * y[j+index*idx]) + 0.0002);
                    p_cheb[index*Sample+jj] *= chebyshev_result / weight;
                }
            }
        }
""")


class SVM(object):
  

    def __init__(self, kernel=cheb.Chebyshev_Tn_gpu , C=None):
        self.kernel = kernel
        self.C = C

        if self.C is not None: self.C = float(self.C)

    def fit(self, X, y):
        t = time.process_time()
        n_samples, n_features = X.shape
        #print(X.shape)

        # Gram matrix
        K = np.zeros((n_samples, n_samples))
        
        p_cheb = np.random.randn(n_samples*n_samples,1)
        p_cheb = p_cheb.astype(np.float32)
        p_cheb_gpu = cuda.mem_alloc(p_cheb.nbytes)
        cuda.memcpy_htod(p_cheb_gpu, p_cheb)
        t2 = X.astype(np.float32)

        y_gpu = cuda.mem_alloc(t2.nbytes)
        cuda.memcpy_htod(y_gpu, t2)
        func = mod.get_function("Chebyshev")
        NSamples = np.int32(n_samples)
        func(y_gpu,y_gpu,p_cheb_gpu,NSamples, block=(n_samples,1,1))
        
        p_cheb = np.empty_like(p_cheb)
        cuda.memcpy_dtoh(p_cheb, p_cheb_gpu)


        for i in range(n_samples):
          for j in range(n_samples):
            K[i,j]=p_cheb[i*n_samples+j]

        y = y.astype(float) #convert to float
        
        elapsed_time = time.process_time() - t
        print("Time = " , elapsed_time)

        def check_symmetric(a, tol=1e-8):
            return np.all(np.abs(a - a.T) < tol)
        
        
        P =  cvxopt.matrix(np.outer(y, y)*K ) 
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        
        
        A = cvxopt.matrix(y, (1, n_samples))
        b = cvxopt.matrix(0.0)

        if self.C is None:
            G = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
            h = cvxopt.matrix(np.zeros(n_samples))
        else:
            tmp1 = np.diag(np.ones(n_samples) * -1)
            tmp2 = np.identity(n_samples)
            G = cvxopt.matrix(np.vstack((tmp1, tmp2)))
            tmp1 = np.zeros(n_samples)
            tmp2 = np.ones(n_samples) * self.C
            h = cvxopt.matrix(np.hstack((tmp1, tmp2)))
            
        # solve QP problem
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        #solution = QPFunction(verbose=False)(P, q, G, h, A, b)

        # Lagrange multipliers
        a = np.ravel(solution['x'])
        
        # Support vectors have non zero lagrange multipliers
        sv = a > 1e-7 # i.e. if 1e-5 returns true if each a[i] > 100000
        ind = np.arange(len(a))[sv]

        self.a = a[sv]
        self.sv = X[sv]
        self.sv_y = y[sv]
 
        # Intercept
        self.b = 0
        for n in range(len(self.a)):
            self.b += self.sv_y[n] 
            self.b -= np.sum(self.a * self.sv_y * K[ind[n], sv])
            
        self.b /= len(self.a)
        
        # Weight vector
        if self.kernel == 'linear':
            self.w = np.zeros(n_features)
            for n in range(len(self.a)):
                self.w += self.a[n] * self.sv_y[n] * self.sv[n]
        else:
            self.w = None

    def project(self, X):
        
        if self.w is not None:
            return np.dot(X, self.w) + self.b
        else:
            y_predict = np.zeros(len(X))
            for i in range(len(X)):
                s = 0
                for a, sv_y, sv in zip(self.a, self.sv_y, self.sv):
                    s += a * sv_y * self.kernel(X[i], sv)
                    
                y_predict[i] = s
            return y_predict + self.b

    def predict(self, X):
        #sign(array [, out]) function is used to indicate the sign of a number element-wise. For integer inputs, if array value is greater than 0 it returns 1, if array value is less than 0 it returns -1, and if array value 0 it returns 0.
        return np.sign(self.project(X))





if __name__ == "__main__":

    # LOAD iris dataset from sklearn
    def data_set():
        bcancer = datasets.load_breast_cancer()
        iris = datasets.load_iris()
        X = iris.data
        y = iris.target
        # to make it binary classification
        for i in range(0, len(y)):
            if y[i] == 1:
                y[i] = 1
            else :
                y[i] = -1
                
        return X, y


    # split last 50 samples as test set
    def split_train_test(X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, shuffle=False)

        return X_train, X_test, y_train, y_test


    def test_linear():
        X, y = data_set()
        #print(y)
        
        
        def normalize(X):
            X_new = np.zeros(shape=X.shape)
            for i in range(0, len(X[0])): #for features. len(X[0])=4, X[0] is an array whith 4 elements
                max_x = np.amax(X[:, i]) + 0.0002
                min_x = np.amin(X[:, i]) - 0.0002
                alpha=0.9
                for j in range(0, len(X)):#len(X):150 -> rows
                   
                   X_new[j,i] = 2* (((X[j, i] - min_x) ** alpha ) /(max_x - min_x)) -1

            return X_new
        X_new = normalize(X)
        
       	X_train, X_test, y_train, y_test = split_train_test(X_new, y)
        
        clf = SVM()
        t = time.process_time()
        clf.fit(X_train, y_train)
        elapsed_time = time.process_time() - t
        print("Fit Time = " , elapsed_time)
        y_predict = clf.predict(X_test)
        results = confusion_matrix(y_test, y_predict) 
        print('Confusion Matrix :')
        print(results)
        print('Accuracy Score :', accuracy_score(y_test, y_predict))
        print('Report : ')
        print(classification_report(y_test, y_predict))
        correct = np.sum(y_predict == y_test)
        print("%d out of %d predictions correct" % (correct, len(y_predict)))


    # plot_margin(X_train[y_train == 1] X_train[y_train == -1], clf)
    t = time.process_time()
    test_linear()
    elapsed_time = time.process_time() - t
    print("Total Time = " , elapsed_time) 