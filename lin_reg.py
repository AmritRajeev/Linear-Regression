import numpy as np
import pandas as pd

def NN(x,y):
    global y1
    
    inp = x
    
    w1 = np.random.rand(2,1)
    b = np.random.rand()
    
    y1 = np.dot(inp,w1) + b 
    #print(y1.shape)
    
    lr = 0.05
    epochs = 10
    
    for i in range(epochs):
        y1 = np.dot(inp,w1)+b
        D_w1 = -2 * (np.dot(np.transpose(inp),(y-y1)))  # Derivative wrt w1
        D_b = -2 * (y-y1)  # Derivative wrt b
        w1 = w1 - lr*D_w1  # Update w1
        b = b - lr*D_b  # Update b

    y1 = np.dot(inp,w1) + b
    loss(y,y1)


def loss(y,y1):
    print(y)
    print(y1)
    e=(y-y1)**2
    #print(e)

    
if __name__ == "__main__" :
    
    x = np.random.randint(20,size=(3,2))
    y = np.random.randint(20,size =(3,1))
    NN(x,y)
    
    
    
    
