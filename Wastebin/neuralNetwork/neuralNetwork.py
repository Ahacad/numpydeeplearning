import numpy as np
import os 
import struct

class neuralNetwork:

    def __init__(self, inNodes, hiddenNodes, outNodes, lr, activation='sigmoid'):
        self.inNodes = inNodes
        self.hiddenNodes = hiddenNodes
        self.outNodes = outNodes
        self.lr = lr
        self.wih = np.random.normal(0.0, pow(self.inNodes, -0.5), (self.hiddenNodes, self.inNodes))
        self.who = np.random.normal(0.0, pow(self.hiddenNodes, -0.5), (self.outNodes, self.hiddenNodes))
        def SIGMOID(x):
            return 1/(1+np.exp(-x))

        if activation == 'sigmoid':
            self.activation = SIGMOID
        else:
            self.activation = lambda x:x
        
    def train(self, inputVector,labels):
        inputs = np.array(inputVector, ndmin=2).T
        labels = np.array(labels, ndmin=2).T

        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation(hidden_inputs)
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation(final_inputs)
        output_errors = labels - final_outputs 
        hidden_errors = np.dot(self.who.T, output_errors)

        self.who += self.lr * np.dot    (output_errors * final_outputs* (1.0 - final_outputs)        , hidden_outputs.T)
        self.wih += self.lr *  np.dot   (hidden_errors*hidden_outputs*(1.0 - hidden_outputs)                 ,  inputs.T)
    #    print(self.who)
     #   print(self.wih)
   
    def forward(self, inputVector):
        inputs = np.array(inputVector, ndmin=2).T
        hiddenInputs = np.dot(self.wih, inputs)
        hiddenOutputs = self.activation(hiddenInputs)
        finalInputs = np.dot(self.who, hiddenOutputs)
        finalOutputs = self.activation(finalInputs)
        
        return finalOutputs




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

    return image, label, 


trainSet, trainLabels = loadMnist('MNIST', 'train')
test, testLabels = loadMnist('MNIST', 't10k')

trainSet = trainSet /255 *0.99 +0.01
test = test /255 *0.99 +0.01
nn = neuralNetwork(784, 200, 10, 0.1)


for i in range(2000):
    temp = np.zeros(10)
    temp[trainLabels[i]] = 1
    nn.train(trainSet[i], temp)

count = 0
for i in range(100):
    out = nn.forward(test[i])
#     print(out)
#     print(np.argmax(out))
#     print(testLabels[i])
    if int(np.argmax(out)) == int(testLabels[i]):
        count += 1
print(count/100)  



'''
def traintest(trainSet, trainLabels, nn):
    batchsize = 100
    for batch in range(100):
        data = trainSet[batch*batchsize:(batch+1)*batchsize]
        labels = trainLabels[batch*batchsize:(batch+1)*batchsize]
        if batch % 10 == 0:
            count = 0 
            for i in range(batchsize):
                predictions = nn.forward(data[i])
                if int(np.argmax(predictions)) == int(labels[i]):
                    count += 1
            print('accuracy = %f' % (count/batchsize))
        else:
            for i in range(batchsize):
                temp = np.zeros((10,1))
                temp[labels[i]][0] = 1
                nn.train(data[i], temp) 

traintest(trainSet,trainLabels, nn)
'''

'''
for i in range(40000):
    temp = np.zeros(10)
    temp[trainLabels[i]] = 1

    nn.train(trainSet[i], temp)

def testAccuracy(n):
    count = 0
    for i in range(n):
        temp = np.zeros(10)
        temp[testLabels[i]] = 1
        output = nn.forward(test[i])
        output = np.argmax(output)
        if int(output) == int(testLabels[i]):
            count += 1
    print('accuracy = %f' % (count/n)) 

testAccuracy(10000)
'''