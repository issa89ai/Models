import numpy as np

class LogisticRegression():
    def __init__(self):
        """
        Initialises Softmax classifier with initializing 
        weights, alpha(learning rate), number of epochs
        and regularization constant.
        """
        self.w = None
        self.b = None
        self.alpha = 0.5
        self.epochs = 100
        self.reg_const = 0.05
    
    
    def sigmoid(self,z):
    
        s = 1/(1+np.exp(-1*z))
        return s
    
    
    # initialize and zero w and b
    def initialize_with_zeros(dim):
    
        w = np.zeros((dim,1))
        b = 0
    
        assert(w.shape == (dim, 1))
        assert(isinstance(b, float) or isinstance(b, int))
    
        return w, b
        
    def propagate(self,w, b, X, Y):
    
        m = X.shape[1]
    
        A = self.sigmoid(np.dot(w.T,X)+b)
        cost = np.sum(Y*np.log(A)+((1-Y)*np.log(1-A)))/-m
    
        dw = np.dot(X,(A-Y).T)/m
        db = (np.sum(A-Y))/m
    
        assert(dw.shape == w.shape)
        assert(db.dtype == float)
        cost = np.squeeze(cost)
        assert(cost.shape == ())
    
        grads = {"dw": dw,
                 "db": db}
    
        return grads, cost
        
    #iteratea and optize weights and bias
    def optimize(self,w, b, X, Y, num_iterations, learning_rate):
    
        costs = []
        print_cost = False
        for i in range(num_iterations):
    
            grads, cost = self.propagate(w, b, X, Y)
    
            dw = grads["dw"]
            db = grads["db"]
    
            w -= learning_rate*dw
            b -= learning_rate*db
    
            if i % 100 == 0:
                costs.append(cost)
    
            if print_cost and i % 100 == 0:
                print ("Cost after iteration %i: %f" %(i, cost))
    
        params = {"w": w,
                  "b": b}
    
        grads = {"dw": dw,
                 "db": db}
    
        return params, grads, costs
        
    def train(self,X_train, Y_train, num_iterations = 1000, learning_rate = 0.05):
        #self.w, self.b = self.initialize_with_zeros(3072)
        dim=3072
        self.w = np.zeros((dim,1))
        self.b = 0
    
        assert(self.w.shape == (dim, 1))
        assert(isinstance(self.b, float) or isinstance(self.b, int))

        parameters, grads, costs = self.optimize(self.w,self.b,X_train,Y_train,num_iterations,learning_rate)
    
        self.w = parameters["w"]
        self.b = parameters["b"]
    
    def predict(self,X):
    
        w=self.w
        b=self.b
        
        m = X.shape[1]
        Y_prediction = np.zeros((1,m))
        w = w.reshape(X.shape[0], 1)
        A = self.sigmoid(np.dot(w.T,X)+b)
    
        for i in range(A.shape[1]):
    
    
            if A[0,i] < 0.5:
                Y_prediction[0,i]=0
            elif A[0,i]>0.5:
                Y_prediction[0,i]=1
            pass
    
        assert(Y_prediction.shape == (1, m))
          
        return Y_prediction