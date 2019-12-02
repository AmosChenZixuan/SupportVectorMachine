import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter as clock


class SVM_SGD:
    def __init__(self,iterations):
        self.iterations=iterations
        
    def fit(self,X,Y):
        self.X=np.mat(X, dtype='float64')
        self.X=np.hstack((self.X, np.ones((100,1))))
        self.Y=np.mat(Y).T
        self.num_of_data,self.num_of_feature=self.X.shape
        self.w=np.mat(np.zeros((self.num_of_feature,1)))
        self.error=[]
        for iters in range(1,self.iterations+1):
            error=0
            for i in range(len(self.X)):
                x=self.X[i]
                y=self.Y[i]
                if (y*(np.dot(x,self.w)))<1:
                    self.w+=( (1/iters)*(-2)*self.w + x.T*y )
                    error+=1
                else:
                    self.w+=(1/iters)*(-2)*self.w
            self.error.append(error)

#        print(self.X)
#        print(self.w)
#        print(np.hstack(((self.X*self.w),self.Y)))
#        #print(self.error)
#        print(self.Y)
    def predict(self,X):
        X =np.mat(X)
        X=np.hstack((X, np.ones((X.shape[0],1)) ))
        return np.sign(np.dot(X,self.w))
    
    def error(self,X,Y):
        pass
    def plot(self):
        ax=plt.subplots(figsize= (10,6))[1]
        
        x_min, x_max = self.X[:,0].min() - 1, self.X[:,0].max() + 1
        y_min, y_max = self.X[:,1].min() - 1, self.X[:,1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05), np.arange(y_min, y_max, 0.05))
        Z=self.predict(np.c_[xx.ravel(), yy.ravel()])
        ax.contourf(xx,yy,np.asarray(Z).reshape(xx.shape),cmap=plt.cm.coolwarm,alpha=0.6 )

        ax.scatter(np.asarray(self.X)[:,0], np.asarray(self.X)[:,1], c=np.asarray(self.Y).flatten(), cmap=plt.cm.coolwarm, s = 50)


        plt.show()
        

d= np.genfromtxt('linear_data.txt')
X,Y=d[:,:2],d[:,-1]



svm_sgd=SVM_SGD(8000)
svm_sgd.fit(X,Y)
svm_sgd.plot()
