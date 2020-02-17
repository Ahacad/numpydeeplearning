object nn:
    
    def __init__(self, layers):
        self.layers = layers
        self.layer_number = len(layers)
        

    def forward(self, x:np.ndarray)->np.ndarray:
        a = x.reshape(self.input_shape)
        for layer in self.layers:
            a = layer(a)
        return a
     
    def backward(self, de_da):
        d = de_da.reshape(self.output_shape)
        for layer in self.layers[::-1]:
            d = layer.backward(d)
        return d

    def train(self, input_vector, labels, loss_function, lr):
        y = self.forward(input_vector)
        loss = loss_function(y, labels)
        self.backward(loss * -lr)
        
    def setLayers(self, layers):
        self.layers = layers

    def appendLayers(self, layers):
        self.layers.append(layers)
    
    