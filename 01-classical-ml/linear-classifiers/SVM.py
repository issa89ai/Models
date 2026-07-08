import numpy as np

class SVM():
    def __init__(self,alpha_in,reg_const_in):
        """
        Initialises Softmax classifier with initializing 
        weights, alpha(learning rate), number of epochs
        and regularization constant.
        """
        self.W = None
        #self.alpha = 1e-7
        self.alpha=alpha_in
        self.sigma = 0.01
        self.epochs = 1500
        self.batchSize=200
        self.reg_const = reg_const_in
        
    def calc_gradient(self, X_train, y_train):
        
        grad_w = np.zeros_like(self.W)
        
        s = X_train.dot(self.W)
        s_yi = s[np.arange(X_train.shape[0]),y_train]
        
        delta = s- s_yi[:,np.newaxis]+1
        
        loss_i = np.maximum(0,delta)
        loss_i[np.arange(X_train.shape[0]),y_train]=0
        loss = np.sum(loss_i)/X_train.shape[0]
        
        loss += self.reg_const*np.sum(self.W*self.W)
       
        ds = np.zeros_like(delta)
        ds[delta > 0] = 1
        ds[np.arange(X_train.shape[0]),y_train] = 0
        ds[np.arange(X_train.shape[0]),y_train] = -np.sum(ds, axis=1)

        grad_w = (1/X_train.shape[0]) * (X_train.T).dot(ds)
        grad_w = grad_w + (2* self.reg_const* self.W)

        return grad_w
    

    def train(self, X_train, y_train):

        sigma = self.sigma
        numClasses = np.max(y_train) + 1
        self.W = sigma * np.random.randn(3072,numClasses)
        
        for i in range(self.epochs):
            xBatch = None
            yBatch = None
            num_train = np.random.choice(X_train.shape[0], self.batchSize)
            xBatch = X_train[num_train]
            yBatch = y_train[num_train]
            grad_w = self.calc_gradient(xBatch,yBatch)
            self.W= self.W - self.alpha * grad_w

    def predict(self, X_test):
        pred = np.zeros(X_test.shape[0])
        s = X_test.dot(self.W)
        pred = np.argmax(s, axis=1)
        return pred