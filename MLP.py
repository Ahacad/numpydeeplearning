import numpy as np
#import matpotlib as plt


class nn:

    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def backward(self, loss):
        for layer in self.layers[::-1]: 
            loss = layer.backward(loss) 

    def train(self, sample, label, lossFunc, lr=0.1):
        predict = self.forward(sample.reshape(sample.size(), 1))
        loss = lossFunc(predict, label.reshape(label.size(),1))
        self.backward(loss * -lr)
        
    def setLayers(self, layers):
        self.layers = layers

     

class Dense:

    def __init__(self, inSize, outSize):
        self.inSize = inSize
        self.outSize = outSize
        self.w = np.random.normal(0.0, pow(self.inSize, -0.5), (self.outSize, self.inSize))      
        self.b = np.random.normal(0.0, 1.0, (self.outSize, 1))


    def __call__(self, x):
        self.x = x.reshape(x.shape[0], 1)
        return np.dot(self.w, x) + self.b

    def backward(self, loss):
        res = np.dot(self.w.T, loss)
        self.w += np.dot(loss, self.x.T)
        self.b += loss
        return res

class lossFunction:
    
    def __init__(self, function="squaredLoss"):

        if function == "squaredLoss":
            self.function = self.sl
        
    def __call__(self, x, y):
        return self.function(x, y)    

    def sl(self, x, y):
        return (x-y)**2 
    
def main():
    mlp = nn([Dense(4, 10), Dense(10, 1)])
    testSet = np.array([[1,3,4,5],[2,3,3,1],[13,4,2,3],[1,2,3,4]])
    label = np.array([1,2,3,4])
    print(mlp.forward(testSet[0]))


if __name__ == "__main__":
    main()


