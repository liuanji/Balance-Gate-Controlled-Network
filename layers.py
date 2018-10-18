import tensorflow as tf
import numpy as np
import math

class BGClayer():
    def __init__(self, output_dim, h_out_dim, activation_func, layername, alpha = 0.2):
        '''
            Args:
            output_dim: output dimension of data vector x
            h_out_dim: output dimension of inter-layer state vector h
            activation_func: activation function
            layername: name of the current layer
            alpha: parameter that controls the linear update module
        '''
        self.output_dim = output_dim
        self.h_out_dim = h_out_dim
        self.activation_func = activation_func
        self.alpha = alpha
        self.layername = layername
    
    def __call__(self, x, h):
        '''
            Args:
            x: input data vector x, should have shape [None, input_dim]
            h: input inter-layer state vector h, should have shape [None, input_dim]
        '''
        self.input_dim = x.get_shape().as_list()[1]
    
        self.W_s = self.create_weight([self.input_dim, self.input_dim], self.layername + "_W_s", type = "xavier")
        self.b_s = self.create_bias([self.input_dim], self.layername + "_b_s")
    
        if self.h_out_dim != 0:
            if h is not None:
                self.W_Hh = self.create_weight([self.input_dim, self.h_out_dim], self.layername + "_W_Hh", type = "xavier")
            self.W_Hs = self.create_weight([self.input_dim, self.h_out_dim], self.layername + "_W_Hs", type = "xavier")
            self.b_h = self.create_bias([self.h_out_dim], self.layername + "_b_h")
                
        self.W = self.create_weight([self.input_dim, self.output_dim], self.layername + "_W", stddev = 0.01 / self.input_dim)
            
        if h is not None:
            s = tf.nn.tanh(tf.matmul(x + h, self.W_s) + self.b_s)
            if self.h_out_dim != 0:
                h = tf.nn.tanh(tf.matmul(h, self.W_Hh) + tf.matmul(s, self.W_Hs) + self.b_h)
        else:
            s = tf.nn.tanh(tf.matmul(x, self.W_s) + self.b_s)
            if self.h_out_dim != 0:
                h = tf.nn.tanh(tf.matmul(s, self.W_Hs) + self.b_h)
                    
        if self.activation_func == 'oneplus':
            y = tf.log(1.0 + tf.exp(tf.matmul(tf.mul(x,s), self.W))) + tf.matmul(tf.mul(x,self.alpha - s), self.W)
        elif self.activation_func == 'relu':
            y = tf.nn.relu(tf.matmul(tf.mul(x,s), self.W)) + tf.matmul(tf.mul(x,self.alpha - s), self.W)
        elif self.activation_func == 'sigmoid':
            y = tf.nn.sigmoid(tf.matmul(tf.mul(x,s), self.W)) + tf.matmul(tf.mul(x,self.alpha - s), self.W)
                
        if self.h_out_dim != 0:
            return y, h
        else:
            return y, None
    
    def create_weight(self, w_shape, name, type = "gauss", stddev = 0.1):
        '''
            Create weight variables
        '''
        if type == "gauss":
            initial = tf.truncated_normal(w_shape, stddev = min(stddev, 0.1), dtype = tf.float32)
        elif type == "uniform":
            initial = tf.random_uniform(w_shape, minval = -stddev, maxval = stddev, dtype = tf.float32)
        elif type == "xavier":
            stddev = 2 * math.sqrt(6.0 / (w_shape[0] + w_shape[1]))
            initial = tf.random_uniform(w_shape, minval = -stddev, maxval = stddev, dtype = tf.float32)
        else:
            print "Unexpected weight type."
            exit(1)
        var = tf.Variable(initial, name = self.layername + "_" + name)
        return var
    
    def create_bias(self, b_shape, name, type = "const", avrg = 0.0):
        '''
            Create bias variables
        '''
        if type == "const":
            initial = tf.constant(np.zeros(b_shape) + avrg, dtype = tf.float32)
        elif type == "gauss":
            initial = tf.truncated_normal(b_shape, stddev = min(avrg, 0.05), dtype = tf.float32)
        else:
            print "Unexpected bias type."
            exit(1)
        var = tf.Variable(initial, name = self.layername + "_" + name)
        return var

class FClayer():
    def __init__(self, output_dim, activation_func, layername):
        '''
            Args:
            output_dim: output dimension of data vector x
            activation_func: activation function
            layername: name of the current layer
        '''
        self.output_dim = output_dim
        self.activation_func = activation_func
        self.layername = layername
    
    def __call__(self, x):
        '''
            Args:
            x: input data vector x, should have shape [None, input_dim]
        '''
        self.input_dim = x.get_shape().as_list()[1]
        
        self.W = self.create_weight([self.input_dim, self.output_dim], self.layername + "_W", type = "xavier")
        self.b = self.create_bias([self.output_dim], self.layername + "_b")
        
        if self.activation_func == 'relu':
            y = tf.nn.relu(tf.matmul(x, self.W) + self.b)
        elif self.activation_func == 'oneplus':
            y = tf.log(1 + tf.exp(tf.matmul(x, self.W) + self.b))
        elif self.activation_func == 'sigmoid':
            y = tf.nn.sigmoid(tf.matmul(x, self.W) + self.b)
        elif self.activation_func == 'tanh':
            y = tf.nn.tanh(tf.matmul(x, self.W) + self.b)
        elif self.activation_func == 'softmax':
            y = tf.nn.softmax(tf.matmul(x, self.W))
        elif self.activation_func == 'none':
            y = tf.matmul(x, self.W) + self.b
        else:
            print 'Unexpected activation function'
            exit(1)
        
        return y

    def create_weight(self, w_shape, name, type = "gauss", stddev = 0.1):
        '''
            Create weight variables
        '''
        if type == "gauss":
            initial = tf.truncated_normal(w_shape, stddev = min(stddev, 0.1), dtype = self.dtype)
        elif type == "uniform":
            initial = tf.random_uniform(w_shape, minval = -stddev, maxval = stddev, dtype = self.dtype)
        elif type == "xavier":
            stddev = 2 * math.sqrt(6.0 / (w_shape[0] + w_shape[1]))
            initial = tf.random_uniform(w_shape, minval = -stddev, maxval = stddev, dtype = tf.float32)
        else:
            print "Unexpected weight type."
            exit(1)
        var = tf.Variable(initial, name = self.layername + "_" + name)
        return var
    
    def create_bias(self, b_shape, name, type = "const", avrg = 0.0):
        '''
            Create bias variables
        '''
        if type == "const":
            initial = tf.constant(np.zeros(b_shape) + avrg, dtype = tf.float32)
        elif type == "gauss":
            initial = tf.truncated_normal(b_shape, stddev = min(avrg, 0.05), dtype = tf.float32)
        else:
            print "Unexpected bias type."
            exit(1)
        var = tf.Variable(initial, name = self.layername + "_" + name)
        return var
