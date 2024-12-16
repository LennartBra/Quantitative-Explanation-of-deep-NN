# -*- coding: utf-8 -*-
"""
Master-Studienarbeit
Autor: Lennart Brakelmann
Thema: Quantitative Erklärbarkeit tiefer neuronaler Netze
Skript: DataGenerator
"""
import numpy as np
import random
from tensorflow.keras.utils import Sequence
## @brief Datagenerator for 6 Inputs of 3 different sources. 
## @details This Datagenerator can be used for big data which would overload the RAM.

class DataGenerator(Sequence):
    def __init__(self, path_main, list_id, batch_size, typ, n_sample=1000, n_classes=2, shuffle=False):        
        ##
        # @brief This constructor initalizes the DataGenerator object.
        # @param path_main      The main path which includes the preprocessed data.
        # @param list_id        The list of the subject IDs.                 
        # @param batch_size     The size of each data batch.
        # @param n_sample       The number of samples in each data instance. Default is 1000.
        # @param n_classes      The number of output classes. Default is 2.
        # @param shuffle        Whether to shuffle the data after each epoch. Default is True.
        ##
        
        #Initialization
        self.path_main = path_main
        self.list_id = list_id
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.n_sample = n_sample
        self.typ = typ
        
        #Define indexes for the signals 
        self.PPG0 = 0 #Index of PPG signal
        self.PPG1 = 1 #Index of the first derivative of PPG signal
        self.PPG2 = 2 #Index of the second derivative of PPG signal
        self.ABP0 = 3 #Index of the arterial blood pressure signal
        self.ABP1 = 4 #Index of the first derivative of arterial blood pressure
        self.ABP2 = 5 #Index of the second derivative of arterial blood pressure
        self.temp0 = 6 #Index of the template signal of PPG
        self.temp1 = 7 #Index of the template signal for first derivative of PPG
        self.temp2 = 8 #Index of the template signal for second derivative of PPG
        self.ground_truth = 9 #Index of ground truth values
        
        self.on_epoch_end()


    # Method --> Denotes the number of batches per epoch
    def __len__(self):
        ##
        # @brief This method count the total number of batches.
        # @return Total number of batches.
        nr_batches = int(np.floor(len(self.list_id)/self.batch_size))
        
        return nr_batches
    
    
    # Method --> Generate one batch of data
    def __getitem__(self, index):
        ##
        # @brief This method returns a batch of data.
        # @return A batch of data
        ##
        #Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        #Find list of IDs
        list_IDs_temp = [self.list_id[k] for k in indexes]
        #Generate data
        x, y = self.__data_generation(list_IDs_temp)
        
            
        if self.typ == 'PPG':
            X = [x[0], x[1], x[2], x[3], x[4], x[5]]
        elif self.typ == 'ABP_multi':
            X = [x[0], x[1], x[2]]
        elif self.typ == 'ABP_single':
            X = [x[0]]

        return X, y
    
    
    def on_epoch_end(self):
        ## 
        # @brief This method declares the order of the data --> linear or random
        ##
        self.indexes = np.arange(len(self.list_id))
        if self.shuffle==True:
            random.shuffle(self.indexes)
            
    # Method --> Generates data containing batch_size samples
    def __data_generation(self, list_IDs_temp):
        ##
        # @brief This method generate one batch.
        # @return A batch of data
        ##
        #Initialization and generation of data
        if self.typ == 'PPG':
            #Initialization
            x1, x2, x3, x4, x5, x6 = [np.zeros((self.batch_size, self.n_sample)) for i in range(6)]
            y = np.zeros((self.batch_size, self.n_classes))
            
            #Generate data
            for i, ID in enumerate(list_IDs_temp):
                Segment = np.load(self.path_main+ID)
                x1[i] = Segment[self.PPG0]
                x2[i] = Segment[self.PPG1]
                x3[i] = Segment[self.PPG2]
                x4[i] = Segment[self.temp0]
                x5[i] = Segment[self.temp1]
                x6[i] = Segment[self.temp2]
                
                y[i] = Segment[self.ground_truth][0:2]
                
            x = np.asarray([x1, x2, x3, x4, x5, x6])
            y = np.asarray(y)
            
        elif self.typ == 'ABP_multi':
            #Intialization
            x1, x2, x3 = [np.zeros((self.batch_size, self.n_sample)) for i in range(3)]
            y = np.zeros((self.batch_size, self.n_classes))
            
            #Generate data
            for i, ID in enumerate(list_IDs_temp):
                Segment = np.load(self.path_main+'/'+ID)
                x1[i] = Segment[self.ABP0]
                x2[i] = Segment[self.ABP1]
                x3[i] = Segment[self.ABP2]
                
                y[i] = Segment[self.ground_truth][0:2]
                
            x = np.asarray([x1, x2, x3])
            y = np.asarray(y)
        elif self.typ == 'ABP_single':
            #Initialization
            x1 = np.zeros((self.batch_size, self.n_sample))
            y = np.zeros((self.batch_size, self.n_classes))
            
            #Generate data
            for i, ID in enumerate(list_IDs_temp):
                Segment = np.load(self.path_main+'/'+ID)
                x1[i] = Segment[self.ABP0]
                
                y[i] = Segment[self.ground_truth][0:2]
                
            x = np.asarray([x1])
            y = np.asarray(y)
            
        return x, y
