import numpy as np

class NetStruct():
    def __init__(self, input_dim, layers_dim, layers_type, activation_func):
        # size of input samples
        self.input_dim = input_dim
        
        # size of network's output
        self.output_dim = layers_dim[-1]
        
        # layer num
        self.layer_num = len(layers_dim)

        # size of hidden layers
        self.layers_dim = layers_dim

        # layer type (0: fully connected layer, 1: BGC layer)
        self.layers_type = layers_type

        # activation function for fully connected layers and BGC layers, respectively
        self.activation_func_fc = activation_func[0]
        self.activation_func_bgc = activation_func[1]
