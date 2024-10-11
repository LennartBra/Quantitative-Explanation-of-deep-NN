"""
Master Studienarbeit im Master Studiengang: Biomedizinische Informationstechnik
Titel: Quantitative Erklärbarkeit tiefer neuronaler Netze in der Analyse von Biosignalen
Autor: Lennart Brakelmann
FH Dortmund
"""
import os
import numpy as np
from scipy.signal import sosfiltfilt, medfilt, butter, resample
import mat73
import scipy.io
from scipy.stats import kurtosis, skew
import neurokit2 as nk
## @brief Class for Preprocessing PulseDB  
## @details This class is used to do a pipeline of preprocessing steps for the PulseDB to equal the data to an iPPG database.


class PreprocessingPulseDB_Pressure():
    def __init__(self, data_path, target_path, fs=100, new_nr_sample=1000, db='m', replace=False):
        ##
        # @brief This constructor initalizes the DataGenerator object.
        # @param data_path      The main path which includes the data of the mimic3 or the vitaldb with all subjects.
        # @param target_path    The target path, where the preprocessed data need to be saved.
        # @param fs             The target sampling rate.                 
        # @param old_fs         The sampling rate of the raw database.
        # @param new_nr_sample  The new number of sample per time epoch
        # @param db             This parameter set the database, which should be preprocessed. 'm'(Default) for Mimic3 and 'v' for VitalDB.      
        ##
        self.data_path = data_path
        self.target_path = target_path
        self.fs = fs
        self.new_nr_sample = new_nr_sample
        self.db = db
        
        self.ids = os.listdir(self.data_path)
        if replace==False:          
            id_ready = os.listdir(self.target_path+"ground_truth/")
            self.ids = [x for x in self.ids if x[:-4]+db+".npy" not in id_ready]
            
        self.keys = None
        self._abp_segments = None
        self._sbp = None
        self._dbp = None
        self._sos = None  
        self._mean = None
        self._var = None
        self.data_error = False
        self._no_error = True
        self._ibi = None

        self._design_filt()


    def _design_filt(self, lowcut=0.5, highcut=8, order=4):
        ##
        # @brief    This method set the sos filter parameter.
        ##
        nyq = 0.5 * self.fs
        low = lowcut / nyq
        high = highcut / nyq
        self._sos = butter(order, [low, high], btype='band', output='sos')
   
    
    def _error_handling(self, _id):
         if self._no_error==True:
             with open(self.target_path+self.db+'_error.txt', 'w') as file:
                 file.write('Following ID(s) have caused an error:\n')
                 file.write(_id[:-4]+self.db)
                 self._no_error==False
         else:
             with open(self.target_path+self.db+'_error.txt', 'a') as file:
                 file.write(_id[:-4]+self.db)
         print(_id, " creates an error!")
         self.data_error = True
   

    def _load_data(self, _id):
        ##
        # @brief This method loads data of the next subject.
        ##   
        # Loading MIMIC3
        if self.db == 'm':
            try:
                try:
                    data_dict = mat73.loadmat(self.data_path+_id)
                    data = data_dict['Subj_Wins']
                    if self.keys==None:
                        keys = data.keys()
                        self.keys = [x for x in keys]
                    self._sbp = np.float32(data['SegSBP'])
                    self._dbp = np.float32(data['SegSBP'])
                    if type(self._sbp)==list:
                        self._abp_segments = np.squeeze(np.float32(data['ABP_Raw']), axis=1)    
                    else:
                        self._dbp = np.expand_dims(self._dbp, axis=0)
                        self._sbp = np.expand_dims(self._sbp, axis=0)
                        self._abp_segments = np.squeeze(np.swapaxes(np.float32(data['ABP_Raw']), 0, 1), axis=0)
                        
                except:
                    data_dict = scipy.io.loadmat(self.data_path+_id)
                    data = data_dict['Subj_Wins'][0][0]
                    self._abp_segments = np.float32(np.swapaxes(data[9], 0, 1))
                    self._dbp = np.float32(np.squeeze(data[-1]), axis=0)
                    self._sbp = np.float32(np.squeeze(data[-2]), axis=0)
                        
            except:
                self._error_handling(_id)
        
        # Loading VitalDB
        elif self.db == 'v':
            try:
                try:
                    data_dict = mat73.loadmat(self.data_path+_id)
                    data = data_dict['Subj_Wins']
                    if self.keys==None:
                        keys = data.keys()
                        self.keys = [x for x in keys]
                    self._sbp = np.float32(data['SegSBP'])
                    self._dbp = np.float32(data['SegDBP'])
                    self._abp_segments = np.squeeze(np.float32(data['ABP_Raw']), axis=1) 
                except:
                    data_dict = mat73.loadmat(self.data_path+_id)
                    data = data_dict['Subj_Wins']
                    if self.keys==None:
                        keys = data.keys()
                        self.keys = [x for x in keys]
                    self._sbp = np.float32(np.expand_dims(data['SegSBP'], axis=0))
                    self._dbp = np.float32(np.expand_dims(data['SegDBP'], axis=0))
                    #self._ppg_segments = np.swapaxes(np.float32(data['PPG_Raw']), 0, 1)
                    self._abp_segments = np.float32(np.expand_dims(data['ABP_Raw'], axis=0))
            except:
                self._error_handling(_id)
                
        
    def _standardize(self):
        ##
        # @brief This method standardizes the data of one subject.
        ##   
        mean = np.mean(np.mean(self._abp_segments, axis=1))
        var = np.mean(np.var(self._abp_segments, axis=1))
        self._abp_segments = (self._abp_segments-mean)/var
        
    
    def _standardize_epochwise(self):
        ##
        # @brief This method standardizes the data per epoch.
        ##   
        def standard_scale(data):
            mean = np.mean(data, axis=1)
            mean_mat = np.tile(mean[:, np.newaxis], len(data[0]))
            std = np.std(data, axis=1)
            std_mat = np.tile(std[:, np.newaxis], len(data[0]))
            data = np.divide(np.subtract(data, mean_mat), std_mat)
        
        standard_scale(self._abp_segments)
        standard_scale(self._dev1_abp)
        standard_scale(self._dev2_abp)
        
        
    def _resampling(self):  
        ##
        # @brief This method resamples the data.
        ##   
        self._abp_segments = np.array([resample(epoch, self.new_nr_sample) for epoch in self._abp_segments])

        
    def _filt(self):
        ##
        # @brief This method does frequency filter the data.
        ##   
        for i in range(len(self._abp_segments)):
            #self._ppg_segments[i] = medfilt(sosfiltfilt(self._sos, self._ppg_segments[i]))
            self._abp_segments[i] = medfilt(sosfiltfilt(self._sos, self._abp_segments[i]))

    
            
        
    def _derivation(self):
        ##
        # @brief This method extracts the first and second derivation of the data.
        ##   
        self._dev1_abp = np.diff(np.pad(self._abp_segments, ((0, 0),(0, 1)), mode='edge'), axis=-1)
        self._dev2_abp = np.diff(np.pad(self._dev1_abp, ((0, 0),(0, 1)), mode='edge'), axis=-1)
        
        
    def process(self):
        ##
        # @brief This method summarize all preprocessing steps in form of a Pipeline
        ##   
        for nr_sub, sub_id in enumerate(self.ids):
            print ("Load Subject ", str(nr_sub+1), " of ", str(len(self.ids)))
            # Load Data
            self._load_data(sub_id)
            if self.data_error==False:
                # Resampling
                self._resampling()
                # Frequency Filter
                self._filt()
                # Derivations
                self._derivation()
                # Standardize
                self._standardize_epochwise()
                # Save Data
                np.save(self.target_path+'dev0_abp/'+str(sub_id[:-4])+self.db, self._abp_segments)
                np.save(self.target_path+'dev1_abp/'+str(sub_id[:-4])+self.db, self._dev1_abp)
                np.save(self.target_path+'dev2_abp/'+str(sub_id[:-4])+self.db, self._dev2_abp)

                '''
                target = np.concatenate((self._sbp, self._dbp), axis=-1)
                if len(target.shape)==3:
                    target = np.squeeze(target, axis=0)
                np.save(self.target_path+'ground_truth/'+str(sub_id[:-4])+self.db, target)
                '''
            else:
                self.data_error=False
      
        


        
        
        
        
        
        
        
        
