import numpy 
import os 
import struct

# neural network class definition
class neuralNetwork:
    
    
    # initialise the neural network
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # set number of nodes in each input, hidden, output layer
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        
        # link weight matrices, wih and who
        # weights inside the arrays are w_i_j, where link is from node i to node j in the next layer
        # w11 w21
        # w12 w22 etc 
        self.wih = numpy.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))

        # learning rate
        self.lr = learningrate
        def SIGMOID(x):
            return 1/(1+numpy.exp(-x))
        # activation function is the sigmoid function
        self.activation_function = SIGMOID
        
        pass

    
    # train the neural network
    def train(self, inputs_list, targets_list):
        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
        
        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)
        
        # output layer error is the (target - actual)
        output_errors = targets - final_outputs
        # hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        hidden_errors = numpy.dot(self.who.T, output_errors) 
        
        # update the weights for the links between the hidden and output layers
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))
        
        # update the weights for the links between the input and hidden layers
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))
        
        pass

    
    # query the neural network
    def query(self, inputs_list):
        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        
        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs



def loadMnist(path, kind='train'):
    '''
    import the MNIST dataset from path, which is the path of the folder
    kind should be either 'train', which is the training set or 't10k', meaning test-10k pictures
    '''
    imagePath = os.path.join(path, '%s-images.idx3-ubyte' % kind)
    labelPath = os.path.join(path, '%s-labels.idx1-ubyte' % kind)

    with open(labelPath, 'rb') as lbp:
        magic, n = struct.unpack('>II', lbp.read(8))
        label = numpy.fromfile(lbp, dtype=numpy.uint8)

    with open(imagePath, 'rb') as imp:
        magic, num, rows, columns = struct.unpack('>IIII', imp.read(16))
        image = numpy.fromfile(imp, dtype=numpy.uint8)
        image = image.reshape(len(label),784)

    return image, label, 


trainSet, trainLabels = loadMnist('MNIST', 'train')
test, testLabels = loadMnist('MNIST', 't10k')

trainSet = trainSet /255 *0.99 +0.01
test = test /255 *0.99 +0.01
nn = neuralNetwork(784, 200, 10, 0.1)


for i in range(2000):
    temp = numpy.zeros(10)
    temp[trainLabels[i]] = 1
    nn.train(trainSet[i], temp)

count = 0
for i in range(100):
    out = nn.query(test[i])
#     print(out)
#     print(numpy.argmax(out))
#     print(testLabels[i])
    if int(numpy.argmax(out)) == int(testLabels[i]):
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
                if int(numpy.argmax(predictions)) == int(labels[i]):
                    count += 1
            print('accuracy = %f' % (count/batchsize))
        else:
            for i in range(batchsize):
                temp = numpy.zeros((10,1))
                temp[labels[i]][0] = 1
                nn.train(data[i], temp) 

traintest(trainSet,trainLabels, nn)
'''

'''
for i in range(40000):
    temp = numpy.zeros(10)
    temp[trainLabels[i]] = 1

    nn.train(trainSet[i], temp)

def testAccuracy(n):
    count = 0
    for i in range(n):
        temp = numpy.zeros(10)
        temp[testLabels[i]] = 1
        output = nn.forward(test[i])
        output = numpy.argmax(output)
        if int(output) == int(testLabels[i]):
            count += 1
    print('accuracy = %f' % (count/n)) 

testAccuracy(10000)
'''