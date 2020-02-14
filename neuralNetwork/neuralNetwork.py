import numpy as np

class neuralNetwork:

    def __init__(self, inNodes, hiddenNodes, outNodes, lr, activation='sigmodi'
        self.inNodes = inNodes
        self.hiddenNodes = hiddenNodes
        self.outNodes = outNodes
        self.lr = lr
        self.wih = np.random.normal(0.0, pow(self.hiddenNodes, -0.5), (self.hiddenNodes, self.inNodes))
        self.who = np.random.normal(0.0, pow(self.outNodes, -0.5), (self.outNodes, self.hiddenNodes))
        if activation == 'sigmoid':
            self.activation = lambda x: x[x<=0] = 0
        else:
            self.activation = lambda x: 1/(1+np.exp(-x))
        
    def train(self, inputVector,labels):
        inputs = inputVector.reshape(-1,1)
        labels = labels.reshape(-1,1)
        hiddenInputs = np.dot(self.wih, inputs)
        hiddenOutputs = self.activation(hiddenInputs)
        finalInputs = np.dot(self.who, hiddenOutputs)
        finalOutputs = self.activation(finalInputs)
        outputError = labels - finalOutputs
        hiddenError = np.dot(self.who.T, outputError)
        self.who += self.lr*np.dot(outputError*finalOutputs*(1-finalOutputs), hiddenOutputs.T)
        self.wih += self.lr*np.dot(hiddenError*hiddenInputs*(1-hiddenInputs), inputs.T)
   
    def forward(self, inputVector):
        inputs = inputVector.reshape(-1,1)
        hiddenInputs = np.dot(self.wih, inputs)
        hiddenOutputs = self.activation(hiddenInputs)
        finalInputs = np.dot(self.who, hiddenOutputs)
        finalOutputs = self.activation(finalInputs)
        