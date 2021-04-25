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

#edited
import torch
from qpth.qp import QPFunction
import time



def polynomial_kernel(x, y, p=3):

    return (1 + np.dot(x, y)) ** p # A.B=C, A: 2D, B: 2D, **:^


def gaussian_kernel(x, y, sigma=1.8):
    return np.exp(-linalg.norm(x - y) ** 2 / (2 * (sigma ** 2))) #||x-y||^2/(2* (sigma^2))




def Tn(x, n,f='r'):
    
    if f == 'e':
        """
            Explicit form
        """
        return np.cos(n * np.arccos(x))
        
    elif f == 'r':
        """
            Recursive form
        """
        if n == 0:
            return 1
        elif n == 1:
            return x
        elif n >= 2:
            return 2 * x * (Tn(x, n - 1)) - Tn(x, n - 2)
            
        



def Chebyshev_Tn(x,y,n=3,f='e'):
    m=len(x)
    chebyshev_result = 0
    p_cheb = 1
    
    for j in range(0,m):    
        for i in range(0,n+1):
            a = np.cos(i * np.arccos(x[j]))
            b = np.cos(i * np.arccos(y[j]))
            #print("a=cos (",i," * acos(x[",j,"]))=cos (",i," * acos(",x[j],"))= ",a)
            #print("b=cos (",i," * acos(y[",j,"]))=cos (",i," * acos(",y[j],"))= ",b)
            chebyshev_result += (a * b)
        weight = np.sqrt(1 - (x[j] * y[j]) + 0.0002)
        p_cheb *= chebyshev_result / weight 
        #print("p_cheb[jj]= ",p_cheb)
        #pdb.set_trace()
    #print (p_cheb)
    return  p_cheb









class SVM(object):

    def __init__(self, kernel=Chebyshev_Tn , C=None):
        self.kernel = kernel
        self.C = C

        if self.C is not None: self.C = float(self.C)

    def fit(self, X, y):
        t = time.process_time()
        n_samples, n_features = X.shape
        #print(X.shape)

        # Gram matrix
        K = np.zeros((n_samples, n_samples))
        X_ = X.astype(np.float32)
        for i in range(n_samples): #n_samples
            for j in range(n_samples): #n_samples
                #pdb.set_trace()
                K[i, j] = self.kernel(X_[i], X_[j]) #Chebyshev_Tn(X[i],X[j],n=3,f='r')
                #print('K[',i,', ',j,']= ',K[i, j])
        #print(K)
        y = y.astype(float) #convert to float
        elapsed_time = time.process_time() - t
        print("Time = " , elapsed_time)
        """
        to solve the inequality(the standard form of the QP):

            min 1/2 (x.T)Px + q.Tx
        subject to: 
            Gx <= h
            Ax = b
        """
        #pdb.set_trace()
        
        def check_symmetric(a, tol=1e-8):
            return np.all(np.abs(a - a.T) < tol)
        
        #print("np.outer(y,y):\n",np.outer(y, y))
        
        P = cvxopt.matrix(np.outer(y, y) * K)
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
        #print("status:",solution['status'])
        # Lagrange multipliers
        a = np.ravel(solution['x'])
        
        # Support vectors have non zero lagrange multipliers
        sv = a > 1e-7 # i.e. if 1e-5 returns true if each a[i] > 100000
        ind = np.arange(len(a))[sv]
        print("ind:",ind)
        self.a = a[sv]
        self.sv = X[sv]
        self.sv_y = y[sv]
        #print("a:\n",a)
        #print("%d support vectors out of %d points" % (len(self.a), n_samples))
        
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
                    """
                    SVM decision function:
                        f(x) = sgn(∑αi*yi*k(x,xi)+b)
                    
                    """
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
        #np.savetxt("iris_data.csv", X, delimiter=',')
        #np.savetxt("iris_target.csv", y, delimiter=',')
        #print("#################################");
        #print(len(X));
        #print(y)
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
                   #X_new[j, i] = ((2 * (X[j, i] - min_x)) / (max_x - min_x)) - 1

            np.savetxt("test.csv", X_new, delimiter=',')
            return X_new
        #pdb.set_trace()
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
#do some stuff

test_linear()
elapsed_time = time.process_time() - t
print("Time = " , elapsed_time)