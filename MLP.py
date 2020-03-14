import numpy as np
import os
import struct
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

    def train(self, sample, label, lossFunc, lr=0.0000001):
        predict = self.forward(sample)
        loss = lossFunc.derivate(predict, label)
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
        self.x = x
        return np.dot(self.w, x) + self.b

    def backward(self, loss):
        res = np.dot(self.w.T, loss)
        self.w += np.dot(loss, self.x.T)
        print('selfw     ::',self.w)
        self.b += loss
        print('selfb     ::',self.b)
        return res

class lossFunction:
    
    def __init__(self, function="squaredLoss"):

        if function == "squaredLoss":
            self.function = self.sl
        
    def __call__(self, x, y):
        return self.function(x, y)    

    def sl(self, x, y):
        return (x-y)**2 
   
    def derivate(self, x, y):
        return 2*(x-y)


def loadMnist(path, kind="train"):
    labelsPath = os.path.join(path, '%s-labels-idx1-ubyte' % kind)
    imagesPath = os.path.join(path, '%s-images-idx3-ubyte' % kind)

    with open(labelsPath, 'rb') as labelPath:
        magic, n = struct.unpack('>II', labelPath.read(8))
        labels = np.fromfile(labelPath, dtype = np.uint8)

    with open(imagesPath, 'rb') as imagePath:
        magic, num, rows, cols = struct.unpack('>IIII', imagePath.read(16))
        images = np.fromfile(imagePath, dtype=np.uint8).reshape(len(labels), 784)

    return images, labels



def mnistTest():
    trainSet, trainLabels = loadMnist('./datasets/mnist', 'train')
    testSet , testLabels  = loadMnist('./datasets/mnist', 'test' )
    network  = nn([Dense(784, 1000), Dense(1000, 10)])
    for i in range(5000):
        network.train(trainSet[i].reshape(-1,1), trainLabels[i].reshape(-1,1), lossFunction())

    count = 0
    for i in range(100):
        answer = np.argmax(network.forward(testSet[i].reshape(-1,1)))
        if answer ==testLabels[i]:
            count += 1
        print(answer, testLabels[i])
    print(count)

def main():

    ''' 
    mlp = nn([Dense(4, 10), Dense(10, 1)])
    testSet = np.array([[1,3,4,5],[2,3,3,1],[13,4,2,3],[1,2,3,4]])
    label = np.array([1,2,3,4])
    print(mlp.forward(testSet[0].reshape(testSet[0].shape[0],1)))
    for i in range(1000):
        print('#######################################################')
        mlp.train(testSet[0].reshape(testSet[0].shape[0],1), label[0].reshape(1,1), lossFunction(), lr=0.0001)
    
    print(mlp.forward(testSet[0].reshape(testSet[0].shape[0],1)))
    '''
    mnistTest()


if __name__ == "__main__":
    main()
