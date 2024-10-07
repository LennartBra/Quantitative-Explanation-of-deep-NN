import numpy as np
import random
from tensorflow.keras.utils import Sequence
## @brief Datagenerator for 6 Inputs of 3 different sources. 
## @details This Datagenerator can be used for big data which would overload the RAM.  

class DataGenerator(Sequence):
    def __init__(self, path_main, list_id, batch_size, typ="big", n_sample=1000, n_classes=2, shuffle=True):        
        ##
        # @brief This constructor initalizes the DataGenerator object.
        # @param path_main      The main path which includes the preprocessed data.
        # @param list_id        The list of the subject IDs.                 
        # @param batch_size     The size of each data batch.
        # @param n_sample       The number of samples in each data instance. Default is 624.
        # @param n_classes      The number of output classes. Default is 2.
        # @param shuffle        Whether to shuffle the data after each epoch. Default is True.
        ##
        
        self.path_main = path_main
        self.list_id = list_id
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.n_sample = n_sample
        self.typ = typ
        
        ## @brief Index of the actual ID from the parameter list_id. 
        ##
        self._id_idx = 0     
        ## @brief Total number of batches. 
        ##        
        self._nr_batches = 0 
        ## @brief Last index of periods of actual subject.
        ##
        self.inPatientIdx = 0
        ## @brief Input data of actual subject.
        ##
        self._dev0 = np.load(self.path_main+"dev0/"+self.list_id[0], allow_pickle=True)
        ## @brief First derivation of input data of actual subject.
        ##
        self._dev1 = np.load(self.path_main+"dev1/"+self.list_id[0], allow_pickle=True)
        ## @brief Second derivation of input data of actual subject.
        ##
        self._dev2 = np.load(self.path_main+"dev2/"+self.list_id[0], allow_pickle=True)
        ## @brief Templates of input data of the actual subject.
        ##
        self._temp0 = np.load(self.path_main+"template/dev0/"+self.list_id[0], allow_pickle=True)
        ## @brief Templates of the first derivation of input data of actual subject.
        ##
        self._temp1 = np.load(self.path_main+"template/dev1/"+self.list_id[0], allow_pickle=True)
        ## @brief Templates of the first derivation of input data of actual subject.
        ##
        self._temp2 = np.load(self.path_main+"template/dev2/"+self.list_id[0], allow_pickle=True)
        ## @brief Target data of actual subject.
        ##
        self._target = np.load(self.path_main+"ground_truth/"+self.list_id[0])


    
    def __count_batches(self):
        ##
        # @brief    This method count the total number of batches.
        # @return   Total number of batches.
        ##
        if self._nr_batches == 0:
            #print("Counting Subjects")
            rest = 0
            for nr, sub in enumerate(self.list_id):
                if nr!=len(self.list_id)-1:
                    n_seg = len(np.load(self.path_main+"dev0/"+sub))
                    #print(n_seg)
                    temp_b1, temp_r = divmod(n_seg, self.batch_size)
                    temp_b2, rest = divmod(temp_r+rest, self.batch_size)       
                    self._nr_batches += temp_b1+temp_b2
                else:
                    n_seg = len(np.load(self.path_main+"dev0/"+sub))
                    #print(n_seg)
                    temp_b1, temp_r = divmod(n_seg, self.batch_size)
                    temp_b2, rest = divmod(temp_r+rest, self.batch_size)  
                    if rest>0:
                        self._nr_batches += temp_b1+temp_b2+1
                    else:
                        self._nr_batches += temp_b1+temp_b2
                
        return self._nr_batches
        

    def __len__(self):
        ##
        # @brief This method count the total number of batches.
        # @return Total number of batches.
        ##
        return self.__count_batches()
    
    def __getitem__(self, idx):
        ##
        # @brief This method returns a batch of data.
        # @return A batch of data
        ##
        x, y = self.__data_generation()
        
        if self.typ=='small':
            return [x[0], x[3]], y
        elif self.typ=='big':
            return [x[0], x[1], x[2], x[3], x[4], x[5]], y

    def on_epoch_end(self):
        ## 
        # @brief This method updates the indexes after each epoch.
        ##
        if self.shuffle==True:
            random.shuffle(self.list_id)
        self.inPatientIdx = 0
        self._id_idx = 0
        self.__load_data()
            

    def __data_generation(self):
        ##
        # @brief This method generate one batch.
        # @return A batch of data
        ##
        x1, x2, x3, x4, x5, x6 = [np.zeros((self.batch_size, self.n_sample)) for i in range(6)]
        y = np.zeros((self.batch_size, self.n_classes))
         
        for i in range(self.batch_size):
            
            if self.inPatientIdx==len(self._dev0):
                if self._id_idx==len(self.list_id)-1:
                    break
                else:
                    self._id_idx += 1
                    self.inPatientIdx = 0
                    self.__load_data()
                    
            x1[i] = self._dev0[self.inPatientIdx]
            x2[i] = self._dev1[self.inPatientIdx]
            x3[i] = self._dev2[self.inPatientIdx]
            x4[i] = self._temp0[self.inPatientIdx]
            x5[i] = self._temp1[self.inPatientIdx]
            x6[i] = self._temp2[self.inPatientIdx]
            y[i] = self._target[self.inPatientIdx, :self.n_classes]
            self.inPatientIdx += 1
            
        x = np.asarray([x1, x2, x3, x4, x5, x6])
        #x = np.reshape(x, (6,1,self.batch_size,self.n_sample))
        y = np.asarray(y)
        return x, y


    def __load_data(self):
        ##
        # @brief This method loads data of the next subject.
        ##   
        self._dev0 = np.load(self.path_main+"dev0/"+self.list_id[self._id_idx], allow_pickle=True)
        self._dev1 = np.load(self.path_main+"dev1/"+self.list_id[self._id_idx], allow_pickle=True)
        self._dev2 = np.load(self.path_main+"dev2/"+self.list_id[self._id_idx], allow_pickle=True)
        self._temp0 = np.load(self.path_main+"template/dev0/"+self.list_id[self._id_idx], allow_pickle=True)
        self._temp1 = np.load(self.path_main+"template/dev1/"+self.list_id[self._id_idx], allow_pickle=True)
        self._temp2 = np.load(self.path_main+"template/dev2/"+self.list_id[self._id_idx], allow_pickle=True)
        self._target = np.load(self.path_main+"ground_truth/"+self.list_id[self._id_idx])

            


        
        
        
        
        
        
        
        
        