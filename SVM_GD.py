import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter as clock
from math import log

class Kernel:
    def __init__(self):
        pass
#        self.sigma = sigma
#        self.dim = dim
#        self.family = [self.linear, self.poly, self.rbf]
    
    def linear(self):
        return lambda x: x
    
    def quadratic(self):
        def transform(x):
            temp_mat=np.mat(x)
            for i in range(x.shape[1]):
                for j in range(i,x.shape[1]):
                    temp_mat=np.hstack((temp_mat,np.multiply(x[:,i],x[:,j])))
            return temp_mat
        return transform

    def third_order(self):
        def transform(x):
            temp_mat=np.mat(x)
            for i in range(x.shape[1]):
                for j in range(i,x.shape[1]):
                    for k in range(j,x.shape[1]):
                        temp_mat=np.hstack((temp_mat,np.multiply(x[:,i],x[:,j],x[:,k])))
            return temp_mat
        return transform
    
    def rbf(self, x, y):
        return np.exp(-float(np.linalg.norm(x - y)) ** 2 / (2 * self.sigma ** 2))

#def line_search(f, gf, x, d, α):
#    p=0.5
#    β=0.0001
#    y= f(x),
#    g=gf(x)
    #num_of_eval=1
#    print(np.dot(g.T,d).shape)
#    print(y)
#    print(f(x + α*d))
#    print()
#    print(y[0]+β*α*np.dot(g.T,d)[0][0])

#    while f(x + α*d)[0] > (y[0] + β*α*np.dot(g.T,d)[0][0]):
#                α *= p
#                num_of_eval+=1
#    print("a=",α)
#    return α

def line_search(f, x, d):
    obj = lambda α: f(x + α*d)
    a, b, num_of_eval1 = bracket_minimum(obj)
    α, num_of_eval2 = golden_section_search(obj, a, b)
    return α, num_of_eval1+num_of_eval2

def bracket_minimum(f, x=0):
    s=1e-2
    k=2.0
    a, ya = x, f(x)
    b, yb = a + s, f(a + s)
    NumberOfEvals=2
    if yb > ya:
        a, b = b, a
        ya, yb = yb, ya
        s = -s
    
    while True:
        c, yc = b + s, f(b + s)
        NumberOfEvals+=1
        if yc > yb:
            return (a, c, NumberOfEvals) if a < c else (c, a, NumberOfEvals)
        a, ya, b, yb = b, yb, c, yc
        s *= k
        
def golden_section_search(f, a, b, n=0.0001):
    φ=(1+5**0.5)/2
    ρ = φ-1
    d = ρ * b + (1 - ρ)*a
    yd = f(d)
    dxk=1
    NumberOfEvals=1
    while dxk>n:
        c = ρ*a + (1 - ρ)*b
        yc = f(c)
        NumberOfEvals+=1
        if yc < yd:
            dxk=abs(b-d)
            b, d, yd = d, c, yc
        else:
            dxk=abs(c-a)
            a, b = b, c
    return ((a+b)/2,NumberOfEvals ) if a < b else ((b+a)/2, NumberOfEvals)


class SVM_SGD:
    def __init__(self,iterations):
        self.iterations=iterations

    def objective_function(self):
        def inner(w):
            temp=0
            for i in range(len(self.X)):
                x=self.X[i]
                y=self.Y[i]
                temp += max(0,1-y.T*(np.dot(x,w)))
            return (1/5)*2*w.T*w+temp
        return inner
#    def gradient(self):
#        return lambda w: (2)*w - self.X.T*self.Y  if self.Y.T*(np.dot(self.X,w))<1 else (2)*w
        
    def fit(self,X,Y,transform):
        self.X=np.mat(X, dtype='float64')
        self.transform=transform
        self.X=np.hstack((transform(self.X), np.ones((self.X.shape[0],1))))
        self.Y=np.mat(Y).T
        self.num_of_data,self.num_of_feature=self.X.shape
        self.w=np.mat(np.zeros((self.num_of_feature,1)))


        iter_num=0
        for iters in range(1,self.iterations+1):
            iter_num+=1

            error=0

            update=np.mat(np.zeros(self.w.shape))
            for i in range(len(self.X)):
                x=self.X[i]
                y=self.Y[i]
                if (y*(np.dot(x,self.w)))<1:
                    #self.w+=( (1/(iters**(1/3)))*(-2)*self.w + x.T*y )
                    #self.w+=(1/log(iters+1))*( (-2)*self.w+x.T*y )
                    #self.w+=( x.T*y )
                    update+=( x.T*y )
                    error+=1
                else:
                    pass
                    #self.w+=(1/(iters**(1/3)))*(-2)*self.w
                    #self.w+=(-2)*self.w

            update+=(-4/5)*self.w
#            if error==0:
#                break
#            if error==0:
#                self.w+=(1/iters)*(-2)*self.w
            a,x=line_search(self.objective_function(),self.w,update)    
            self.w+=a*update
        print("Number of iterations:",iter_num)

#        svm_iter_num=0
#        for iters in range(1,self.iterations/2+1):
##            svm_iter_num+=1
 #           sv_pos1=[]
#           sv_neg1=[]
#            min_distance=
#            for i in range(len(self.X)):
#                x=self.X[i]
#                y=self.Y[i]
#                d=abs(np.dot(x,self.w))
                
            
#        print(self.X)
#        print(self.w)
#        print(np.hstack(((self.X*self.w),self.Y)))
#        #print(self.error)
#        print(self.Y)
    def predict(self,X):
        X =np.mat(X)
        X=np.hstack((self.transform(X), np.ones((X.shape[0],1)) ))
        return np.sign(np.dot(X,self.w))
    
    def error(self,X,Y):
        prediction=self.predict(X)
        e=0
        for i in range(len(X)):
            if prediction[i] != Y[i]:
                e+=1
        return e/len(X)
    
    def plot(self):
        ax=plt.subplots(figsize= (10,6))[1]
        
        x_min, x_max = self.X[:,0].min() - 1, self.X[:,0].max() + 1
        y_min, y_max = self.X[:,1].min() - 1, self.X[:,1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05), np.arange(y_min, y_max, 0.05))
        Z=self.predict(np.c_[xx.ravel(), yy.ravel()])
        ax.contourf(xx,yy,np.asarray(Z).reshape(xx.shape),cmap=plt.cm.coolwarm,alpha=0.6 )

        ax.scatter(np.asarray(self.X)[:,0], np.asarray(self.X)[:,1], c=np.asarray(self.Y).flatten(), cmap=plt.cm.coolwarm, s = 50)


        plt.show()
        
'''
These codes attempts to optimize the primal problem of the SVM.
The primal problem is:

w = argmin(w)[(1/R)*w^2 + Sum( max(0, 1-y*(w*x+b)) )]

Optimizing this problem helps us to find a decision boundary, (a hyperplane), which is represented by its weights on all the features.
and said decision boundary shall make such correct decision as far to the data as possible.

Gradient descent optimization method is applied to the primal problem
in each iteration, the gradient with respect to w is used as the decesnt direction.
the decsent direction, or the negative gradient is :

d=-Grad(primal problem)=-R*w+x*y

Then, according to this descent direction, a step size is chosen using bracketing and golden section search method.

a=argmin(a)[primal problem(w+a*d)]


w is then updated by: w+a*d.
finally, after a sufficient number of update iterations, w is outputed to be the final answer of the primal problem.
'''

d= np.genfromtxt('nonlinear_data.txt')#Name of the data file
X,Y=d[:,:2],d[:,-1]

svm_sgd=SVM_SGD(1000)#number of iterations to perform
svm_sgd.fit(X,Y,Kernel().quadratic())#apply a fit, under a kernel
print("Error:",svm_sgd.error(X,Y))#print the error
svm_sgd.plot()#plot the data and decision boundary
