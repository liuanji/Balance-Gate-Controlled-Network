import tensorflow as tf
import numpy as np
import os
from layers import BGClayer, FClayer

class BGCnet():
    def __init__(self, sess, network_struct, dataset, train_batch_size = 32, test_batch_size = 256):
        '''
            Args:
            sess: Tensorflow session
            network_struct: a NetStruct object, see network_struct.py for details
            dataset: a Datasets object, see datasets.py for details
            train_batch_size: size of a training batch
            test_batch_size: size of a test batch
        '''
        self.sess = sess
        
        self.network_struct = network_struct
        self.dataset = dataset
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        
        self.dtype = np.float32
        
        self.network_vars = []
        self.netname = "BGCnet"
        
        self.create_network()
        self.create_training_method()

        self.sess.run(tf.global_variables_initializer())
    
        self.saver = tf.train.Saver(tf.trainable_variables())

    def create_network(self):
        '''
            Create a BGC net based on the structure described in network_struct.
        '''
        self.input_x = tf.placeholder(self.dtype, [None, self.network_struct.input_dim])
        self.input_y = tf.placeholder(self.dtype, [None, self.network_struct.output_dim])
    
        x = self.input_x
        h = None
    
        for l in xrange(self.network_struct.layer_num):
            if self.network_struct.layers_type[l] == 1:
                next_h_size = 0
                for i in xrange(l + 1, self.network_struct.layer_num):
                    if self.network_struct.layers_type[i] == 1:
                        next_h_size = self.network_struct.layers_dim[i-1]
                        break

                layer = BGClayer(self.network_struct.layers_dim[l], next_h_size, self.network_struct.activation_func_bgc, "layer" + str(l))
                x, h = layer(x, h)
            else:
                if l == self.network_struct.layer_num - 1:
                    layer = FClayer(self.network_struct.layers_dim[l], "none", "layer" + str(l))
                else:
                    layer = FClayer(self.network_struct.layers_dim[l], self.network_struct.activation_func_fc, "layer" + str(l))
                x = layer(x)

        self.y = x

    def create_training_method(self):
        '''
            Create loss, optimizer, etc.
        '''
        error = tf.reduce_mean(tf.square(self.y - self.input_y))
    
        self.optimizer = tf.train.AdamOptimizer(1e-4).minimize(error)
        self.loss = tf.sqrt(error)

    def train_step(self):
        '''
            Train model.
        '''
        x, y = self.dataset.fetch_samples(type = "train", num = self.train_batch_size)
        
        self.sess.run(self.optimizer, feed_dict = {self.input_x: x,
                      self.input_y: y
                      })
            
    def test_step(self):
        '''
            Test model.
        '''
        x, y = self.dataset.fetch_samples(type = "test", num = self.test_batch_size)
        
        loss = self.sess.run(self.loss, feed_dict = {self.input_x: x,
                             self.input_y: y
                             })
    
        return loss
    
    def get_output(self, x):
        '''
            Get the model's output.
        '''
        y = self.sess.run(self.y, feed_dict = {self.input_x: x})
    
        return y
    
    def save_network(self):
        '''
            Save network.
        '''
        if not os.path.exists("save/" + self.netname):
            os.mkdir("save/" + self.netname)
        self.saver.save(self.sess, "save/" + self.netname + "/" + self.netname + ".ckpt")
        print "network saved"
    
    def load_network(self):
        '''
            Load network.
        '''
        if os.path.exists("save/" + self.netname + "/checkpoint"):
            self.saver.restore(self.sess, "save/" + self.netname + "/" + self.netname + ".ckpt")
            print "network loaded"








