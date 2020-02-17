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
    return max(0,x)

def reluFunctionDerivate(x):
    return x if x>0 else 0 

relu = Function(reluFunction, reluFunctionDerivate)

def softmax(x):
    return (np.exp(x-max(x))) / sum(np.exp(x-max(x)))

def softmaxDeviate(x):
    

softmax = Function(softmax, softmaxDeviate)