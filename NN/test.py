import numpy as np
import os
import struct
# import matplotlib.pyplot as plt
from nn import *
from layers import *
from functions import *


#
def loadMnist(path, kind='train'):
    '''
    import the MNIST dataset from path, which is the path of the folder
    kind should be either 'train', which is the training set or 't10k', meaning test-10k pictures
    '''
    imagePath = os.path.join(path, '%s-images.idx3-ubyte' % kind)
    labelPath = os.path.join(path, '%s-labels.idx1-ubyte' % kind)

    with open(labelPath, 'rb') as lbp:
        magic, n = struct.unpack('>II', lbp.read(8))
        label = np.fromfile(lbp, dtype=np.uint8)

    with open(imagePath, 'rb') as imp:
        magic, num, rows, columns = struct.unpack('>IIII', imp.read(16))
        image = np.fromfile(imp, dtype=np.uint8)
        image = image.reshape(len(label),784)

    return image, label

path = '../neuralNetwork/MNIST'
train, trainLabels = loadMnist(path)
test, testLabels = loadMnist(path, 't10k')

def computeAccuracy()



neural = nn([Dense(784, 200), FunctionLayer(relu), Dense(200,10), FunctionLayer(sigmoid)], 784, 10)
batchSize = 100

for i in range(100):
    batchData = train[i*batchSize:(i+1)*batchSize]
    batchLabel = trainLabels[i*batchSize:(i+1)*batchSize]
    if i % 10 == 0:
        count = 0 
        for j in range(batchSize):
            if int(batchLabel[j]) == int(neural.forward(batchData[j]).argmax()):
                count += 1
        print('accuracy = %f' % (count / batchSize))
    else:
        for j in range(batchSize):
            neural.train(batchData[j], batchLabel[j], sse, 0.1)
