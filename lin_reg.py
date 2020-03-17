import numpy as np

def NN(x,y):
    global y1
    
    inp = x
    w1 = np.random.rand(1,3)
    w1 = np.transpose(w1)
    b = np.random.randint(0,5)
    y1 = np.dot(inp,w1)+b
    y1 = np.transpose(y1)
   
    lr = 0.05
    epochs = 1
    
    for i in range(epochs):
        y1 = np.dot(inp,w1)+b
        D_w1 = -2 * (np.dot(np.transpose(inp),(y-y1)))  # Derivative wrt w1
        D_b = -2 * (y-y1)  # Derivative wrt b
        w1 = w1 - lr*D_w1  # Update w1
        b = b - lr*D_b  # Update b

    y1 = np.dot(inp,w1)+b
    y1 = np.transpose(y1)
    loss(y,y1)
    

def loss(y,y1):
    #print(y)
    #print(y1)
    e=(y-y1)**2
    print(e)

    
if __name__ == "__main__" :
    
    x = np.random.randint(20,size=(5, 3))
    y = np.random.randint(20,size =(1,5))
    NN(x,y)
    
    
    
    
