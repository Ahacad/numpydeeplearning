class Dense:
    
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.w = np.random.normal(0.0, pow(self.input_size ,-0.5), (self.output_size, self.input_size))
        self.b = np.random.normal(0.0, 1.0, (self.output_size, 1))
        self.x = None

    def __call__(self, x):
        x = x.reshape(-1,1)
        self.x = x
        self.z = np.dot(self.w, x) + self.b
        return self.z

    def backward(self, de_dz):
        de_dx = np.dot(self.w.T, de_dz)
        self.w += np.dot(de_dz, self.x.T)
        self.b += de_dz
        return de_dx

        