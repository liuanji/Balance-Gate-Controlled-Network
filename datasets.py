import numpy as np
import scipy.io as sio

class Datasets():
    def __init__(self, dataset = "MacheyGlass"):
        '''
            Args:
            dataset: name of the dataset. Data files are stored in folder
                     "Datasets" with MATLAB format.
        '''
        filename = "Datasets/" + dataset + ".mat"
        data = sio.loadmat(filename)
    
        self.train_x = data["train_x"]
        self.train_y = data["train_y"]
        self.test_x = data["test_x"]
        self.test_y = data["test_y"]
    
        self.train_sample_num = np.shape(self.train_x)[0]
        self.test_sample_num = np.shape(self.test_x)[0]
    
        self.input_dim = np.shape(self.train_x)[1]
        self.output_dim = np.shape(self.train_y)[1]

    def fetch_samples(self, type = "train", num = 1):
        '''
            Args:
            type: fetch samples from (train/test) batch
            num: number of returned samples
        '''
        if type == "train":
            if num > self.train_sample_num:
                num = self.train_sample_num
    
            m = np.random.choice(self.train_sample_num, num)
    
            return self.train_x[m,:], self.train_y[m,:]
        elif type == "test":
            if num > self.test_sample_num:
                num = self.test_sample_num
        
            m = np.random.choice(self.test_sample_num, num)
            
            return self.test_x[m,:], self.test_y[m,:]
        else:
            print "Unexpected type"
            exit(1)
