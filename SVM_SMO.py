import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter as clock

class Kernel:
    def __init__(self, sigma = 0.5, dim = 2):
        self.sigma = sigma
        self.dim = dim
        self.family = [self.linear, self.poly, self.rbf]
    
    def linear(self, x, y):
        return float(x.dot(y.T))
    
    def poly(self, x, y):
        return self.linear(x,y) ** self.dim
    
    def rbf(self, x, y):
        return np.exp(-float(np.linalg.norm(x - y)) ** 2 / (2 * self.sigma ** 2))
    
class SVM:
    '''
    C-Support Vector Classifier
    References: 
    J. C. Platt. Fast training of support vector machines using sequential minimal optimization. 1998. MIT Press
    P.-H. Chen, R.-E. Fan, and C.-J. Lin. A study on SMO-type decomposition methods for support vector machines.July 2006 https://www.csie.ntu.edu.tw/~cjlin/papers/generalSMO.pdf
    '''
    def __init__(self, kernel, C = 1, k_max = 100, tol = 1e-3):
        '''
        kernel: Kernel Transformation function
        C: Soft Margin Regularizer 
        k_max: Maximum update iteration
        tol: KKT tolerance
        '''
        self.kernel = kernel
        self.C = C
        self.k_max = k_max
        self.tol = tol
    
    # Optimization
    
    def fit(self, X, Y):
        '''
        X: Feature values # m*n
        Y: Target values {-1, 1} # m*1
        _alphas: Lagrangian Multipliers # m*1
        b = Bias
        _Q: Matrix of kernel values  # m*m
        _E: Error Cache # m*1
        
        sv: Support Vector informations
        report: Training report
        '''
        self.X = np.mat(X, dtype = 'float64') 
        if set(Y) != {-1, 1}:
            raise ValueError('Traget values have to be -1 or 1')
        self.Y = np.mat(Y).T      
        self.m, self.n = self.X.shape
        self._alphas = np.mat(np.zeros((self.m, 1))) 
        self.b = 0
        self._Q = self._compute_Q(self.X) 
        self._E = np.mat(-self.Y) 
        
        self.sv = dict()
        self.report = dict()
        self.report['iteration'] = 0
        self.report['time'] = clock()
        return self._sequential_minimal_optimization()
    
    def _sequential_minimal_optimization(self):
        '''
        Chen, Fan, and Lin. pg.2-4
        (1) if all alphas satisfy KKT, done
        (2) Choose a violating pair i and j, optimize the sub-problem S(i, j)
        (3) update alpha[i,j] and b. Repeat from (1)
        '''
        all_traversed, pair_updated = False, 0
        while self.report['iteration'] < self.k_max:
            self.report['iteration'] += 1
            if all_traversed and pair_updated == 0:
                break
            pair_updated = 0
            if not all_traversed:
                for i in range(self.m):
                    pair_updated += self._inner_optimize(i)
                all_traversed = True
            else:
                for i, alpha in enumerate(self._alphas):
                    if 0 < alpha and alpha < self.C:
                        pair_updated += self._inner_optimize(i)
                all_traversed = pair_updated != 0
                
        self._record_support_vectors()
        self.report['time'] = clock() - self.report['time']
        return self.report
    
    def _inner_optimize(self, i):
        '''
        J. C. Platt. Section 12.2.1 ~ 4
        Solving the sub-problem:
        (1) Maximal violating pair heuristic to find i, j (Chen, Fan, and Lin, pg.3)
        (2) Lagrange multipliers analysis to compute l, h, eta, and use them to update alphai, alphaj (Platt, Section 12.2.1)
        (3) Update b (Platt, Section 12.2.3)
        (4) Update E (Platt, Section 12.2.4)
        '''
        alphai, Yi, Ei = self._alphas[i, 0], self.Y[i, 0], self._E[i, 0]
        if self._KKT_check(alphai, Yi, Ei):
            # KKT satisfied, no optimization needed
            return 0
        # select j to have maximun |Ei - Ej|
        j = self._best_j(i)
        alphaj, Yj, Ej = self._alphas[j, 0], self.Y[j, 0], self._E[j, 0]
        # compute boundaries
        l, h = self._compute_bound(alphai, Yi, alphaj, Yj)
        eta = self._Q[i, i] + self._Q[j, j] - 2.0 * self._Q[i, j]
        if l == h or eta == 0:
            return 0
        
        # update alpha j and i
        j_prime = alphaj + Yj * (Ei - Ej) / eta
        self._alphas[j, 0] = h if j_prime > h else l if j_prime < l else j_prime
        j_delta = Yj * (self._alphas[j, 0] - alphaj)
        self._alphas[i, 0] -= Yi * j_delta
        i_delta = Yi * (self._alphas[i, 0] - alphai)
        
        # update b
        prev_b = self.b
        b1 = prev_b - Ei - i_delta * self._Q[i, i] - j_delta * self._Q[i, j]
        b2 = prev_b - Ej - i_delta * self._Q[i, j] - j_delta * self._Q[j, j]
        if 0 < self._alphas[i, 0] < self.C:
            self.b = b1
        elif 0 < self._alphas[j, 0] < self.C:
            self.b = b2
        else:
            self.b = (b1 + b2) / 2
            
        # update E
        b_delta = self.b - prev_b
        self._E += ([i_delta, j_delta] * self._Q[[i, j]]).T + b_delta
        
        return 1
            
        
    def _KKT_check(self, alpha, y, e):
        '''
        J. C. Platt. Pg.42, (12.2)
        '''
        return not((alpha < self.C and y * e < -self.tol) or\
                  (alpha > 0 and y * e > self.tol))
    
    def _compute_bound(self, alphai, Yi, alphaj, Yj):
        '''
        J. C. Platt. Section 12.2.1
        '''
        if Yi == Yj:
            l = max(0, alphai + alphaj - self.C)
            h = min(self.C, alphai + alphaj)
        else:
            l = max(0, alphaj - alphai)
            h = min(self.C, self.C + alphaj - alphai)
        return l, h
    
    def _best_j(self, i):
        '''
        Chen, Fan, and Lin. pg.3
        maximal violating pair:
        Find the best j such that |Ei - Ej| is the largest, that is:
        if Ei > 0, find the smallest Ej
        else, find the largest Ej
        '''
        indexes, values = [], []
        for k, alpha in enumerate(self._alphas):
            if k != i and 0 < alpha and alpha < self.C:
                indexes.append(k); values.append(self._E[k])
               
        if len(values) == 0: # no best j available, pick any j that is not i
            j = i
            while j == i: j = (self.m - i) // 2; i += 1
            return j
        # maximize |Ei - Ej|
        find = np.argmin if self._E[i] > 0 else np.argmax
        return indexes[find(values)]
        
    def _compute_Q(self, X):
        Q = np.mat(np.zeros((self.m, self.m)))
        for i, x_i in enumerate(X):
            for j, x_j in enumerate(X[:i + 1]):
                Q[i, j] = Q[j, i] = self.kernel(x_i, x_j)
        return Q
    
    def _record_support_vectors(self):
        self.sv['i'] = []
        for i, alpha in enumerate(self._alphas):
            if alpha > 0:
                self.sv['i'].append(i)
        svi = self.sv['i']
        self.sv['X'] = self.X[svi]
        self.sv['Y'] = self.Y[svi]
        self.sv['alphas'] = self._alphas[svi]
        self.report['sv'] = len(svi)
    
    # Methods
    
    def predict(self, X):
        '''
        w = alpha*y
        Decision Function: sign(w * K(x) + b)
        only support vectors make the prediction.
        '''
        if len(self.sv) == 0:
            raise RuntimeError('Untrained SVM. Please train the model first.')
        X = np.mat(X)
        K = np.mat([[self.kernel(x, xi) for xi in self.sv['X']] for x in X])
        y = K * np.multiply(self.sv['alphas'], self.sv['Y']) + self.b
        return np.sign(y)
    
    def error(self, X, Y):
        Y_hat = self.predict(X)
        err = sum([Y_hat[i] != Y[i] for i in range(len(X))]) / len(X)
        self.report['error'] = np.asscalar(err)
        return self.report['error']
    
    def plot(self, title = 'SVM', predict = True, data = True, sv = True):
        _, ax = plt.subplots(figsize = (10, 6))
        if predict:
            self._plot_contours(ax)
        if data:
            self._plot_data(ax)
        if sv:
            self._plot_sv(ax)
        ax.set_ylabel('feature 2')
        ax.set_xlabel('feature 1')
        ax.set_title(title, fontsize = 20)
        plt.show()
    
    def _plot_data(self, ax):
        X = np.asarray(self.X)
        Y = np.asarray(self.Y).flatten()
        ax.scatter(X[:,0], X[:,1], c = Y, cmap=plt.cm.coolwarm, s = 50)
        
    def _plot_contours(self, ax):
        xx, yy = self._meshgrid(self.X[:,0], self.X[:,1])
        Z = self.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = np.asarray(Z).reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha = 0.6)
    
    def _plot_sv(self, ax):
        for i, alpha in enumerate(self._alphas):
            if alpha > 0:
                ax.scatter(self.X[i, 0], self.X[i, 1], color='', edgecolors='k', marker='o', s=150)
    
    def _meshgrid(self, x, y, space = 0.5, h=.05):
        x_min, x_max = x.min() - space, x.max() + space
        y_min, y_max = y.min() - space, y.max() + space
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        return xx, yy