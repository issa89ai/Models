import numpy as np
import torch
import torch.nn as nn
import torch
import torch.nn as nn
import argparse
import torch
import torch.optim as optim

class LinearSVM(nn.Module):
    """Support Vector Machine"""

    def __init__(self):
        super(LinearSVM, self).__init__()
        self.fc = nn.Linear(3072, 1)

    def forward(self, x):
        h = self.fc(x)
        return h
        
class SVM():
    def __init__(self):
        
        self.W = None
        self.alpha = 0.0001 #learning rate of the SGD
        self.epochs = 10  # number of batches
        self.reg_const = 0.005
        
    def train(self, X_train, y_train):
        c=self.reg_const
        lr=self.alpha
        batchsize=0.1
        epochs=self.epochs
     
        X, Y = X_train,y_train
        
        X = (X - X.mean()) / X.std()
        Y[np.where(Y == 0)] = -1

        model = LinearSVM()
        if torch.cuda.is_available():
            model.cuda()
        
        X = torch.FloatTensor(X)
        Y = torch.FloatTensor(Y)
        N = len(Y)

        optimizer = optim.SGD(model.parameters(), lr)

        model.train()
        for epoch in range(epochs):
            perm = torch.randperm(N)
            sum_loss = 0
    
            for i in range(0, N, batchsize):
                x = X[perm[i : i + batchsize]]
                y = Y[perm[i : i + batchsize]]
    
                if torch.cuda.is_available():
                    x = x.cuda()
                    y = y.cuda()
    
                optimizer.zero_grad()
                output = model(x)
    
                loss = torch.mean(torch.clamp(1 - output.t() * y, min=0))  # hinge loss
                loss += c * torch.mean(model.fc.weight ** 2)  # l2 penalty
                loss.backward()
                optimizer.step()
                loss_tran=loss.data.cpu().numpy()
                sum_loss += loss_tran
                
            W=model.fc.weight
            print("Shape of Weight",W.shape)
    
            print("Epoch:{:4d}\tloss:{}".format(epoch, sum_loss / N))
        self.W = W

    def predict(self, X_test):
        pred = np.zeros(X_test.shape[0])
        W_Tran=self.W
        W_Tr=W_Tran.data.cpu().numpy()
        W_T = np.transpose(W_Tr)
        s = X_test.dot(W_T)
        pred = np.argmax(s, axis=1)
        return pred