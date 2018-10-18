import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from datasets import Datasets
from network_struct import NetStruct
from model import BGCnet

def main():
    with tf.Session() as sess:
        # dataset
        dataset = Datasets(dataset = "MacheyGlass")
    
        # network structure
        # layers_dim contains the output size of each layer
        # layers_type determines the type of each layer (1: BGC layer 2: FC layer)
        # activation_func: activation function used for FC layer and BGC layer, respectively
        layers_dim = [12,12,8,6,6,4,4,dataset.output_dim]
        layers_type = [1,1,0,1,1,0,1,0]
        activation_func = ("relu", "oneplus")
        
        netstruct = NetStruct(dataset.input_dim, layers_dim, layers_type, activation_func)

        # construct the network using parameters in netstruct
        net = BGCnet(sess, netstruct, dataset)

        for iter in xrange(100000):
            net.train_step()

            if iter % 100 == 0 and iter != 0:
                loss = net.test_step()
                print "iter: ", iter, " loss: ", loss

        # visualization
        netoutput = net.get_output(dataset.test_x)
        plt.figure()
        plt.plot(netoutput, 'r--')
        plt.plot(dataset.test_y, 'b')
        plt.legend(["network output", "ground truth"])
        plt.show()

if __name__ == "__main__":
    main()
