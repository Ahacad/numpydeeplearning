import numpy as np
class Function:

    def __init__(self, function, functionDerivate):
        self.function = function
        self.functionDerivate = functionDerivate

    def __call__(self, x):
        return self.function(x)
    
    def derivate(self, x):
        return self.functionDerivate(x)




def sigmoidFunction(x):
    return 1.0/(1.0 + np.exp(x))

def sigmoidFunctionDerivate(x):
    return sigmoid(x)*sigmoid(1-x)

sigmoid = Function(sigmoidFunction, sigmoidFunctionDerivate)

def reluFunction(x):
    x[x<=0] = 0
    return x

def reluFunctionDerivate(x):
    x[x<=0] = 0 
    x[x>0] = 1
    return x

relu = Function(reluFunction, reluFunctionDerivate)

def softmax(x):
    return (np.exp(x-max(x))) / sum(np.exp(x-max(x)))

def softmaxDeviate(x):
    pass

softmax = Function(softmax, softmaxDeviate)


class lossFunction:
    
    def __init__(self, function, functionDerivate):
        self.function = function
        self.functionDerivate = functionDerivate

    def __call__(self, prediction, labels):
        return self.function(prediction, labels)

    def derivate(self, prediction, labels):
        return self.functionDerivate(prediction, labels)


def l2error(prediction, labels):
    retunr (labels-prediction)**2

def l2errorDerivate(prediction, labels):
    return 2*(prediction-labels)

sse = lossFunction(l2error, l2errorDerivate)