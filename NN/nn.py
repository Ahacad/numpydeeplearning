object nn:
    
    def __init__(self, layers, inputShape, outputShape):
        self.layers = layers
        self.layerNumber = len(layers)
        self.inputShape = inputShape
        self.outputShape = outputShape

    def forward(self, x:np.ndarray)->np.ndarray:
        a = x.reshape(self.inputShape, 1)
        for layer in self.layers:
            a = layer(a)
        return a
     
    def backward(self, de_da):
        d = de_da.reshape(self.output_shape)
        for layer in self.layers[::-1]:
            d = layer.backward(d)
        return d

    def train(self, inputVector, labels, lossFunction, lr):
        y = self.forward(inputVector)
        loss = lossFunction.derivate(y, labels)
        self.backward(loss * -lr)
        
    def setLayers(self, layers):
        self.layers = layers

    def appendLayers(self, layers):
        self.layers.append(layers)
    
    