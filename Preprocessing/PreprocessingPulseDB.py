import os
import numpy as np
from scipy.signal import sosfiltfilt, medfilt, butter, resample
import mat73
import scipy.io
from scipy.stats import kurtosis, skew
import neurokit2 as nk
## @brief Class for Preprocessing PulseDB  
## @details This class is used to do a pipeline of preprocessing steps for the PulseDB to equal the data to an iPPG database.


class PreprocessingPulseDB():
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
        self._ppg_segments = None
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
                    self._dbp = np.float32(data['SegDBP'])
                    self._sbp = np.float32(data['SegSBP'])
                    if type(self._sbp)==list:
                        self._ppg_segments = np.squeeze(np.float32(data['PPG_Raw']), axis=1)    
                    else:
                        self._dbp = np.expand_dims(self._dbp, axis=0)
                        self._sbp = np.expand_dims(self._sbp, axis=0)
                        self._ppg_segments = np.squeeze(np.swapaxes(np.float32(data['PPG_Raw']), 0, 1), axis=0)
                        
                except:
                    data_dict = scipy.io.loadmat(self.data_path+_id)
                    data = data_dict['Subj_Wins'][0][0]
                    self._ppg_segments = np.float32(np.swapaxes(data[9], 0, 1))
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
                    self._dbp = np.float32(data['SegDBP'])
                    self._sbp = np.float32(data['SegSBP'])
                    self._ppg_segments = np.squeeze(np.float32(data['PPG_Raw']), axis=1) 
                except:
                    data_dict = mat73.loadmat(self.data_path+_id)
                    data = data_dict['Subj_Wins']
                    if self.keys==None:
                        keys = data.keys()
                        self.keys = [x for x in keys]
                    self._dbp = np.float32(np.expand_dims(data['SegDBP'], axis=0))
                    self._sbp = np.float32(np.expand_dims(data['SegSBP'], axis=0))
                    #self._ppg_segments = np.swapaxes(np.float32(data['PPG_Raw']), 0, 1)
                    self._ppg_segments = np.float32(np.expand_dims(data['PPG_Raw'], axis=0))
            except:
                self._error_handling(_id)
                
        
    def _standardize(self):
        ##
        # @brief This method standardizes the data of one subject.
        ##   
        mean = np.mean(np.mean(self._ppg_segments, axis=1))
        var = np.mean(np.var(self._ppg_segments, axis=1))
        self._ppg_segments = (self._ppg_segments-mean)/var
        
    
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
        
        standard_scale(self._ppg_segments)
        standard_scale(self._dev1)
        standard_scale(self._dev2)
        
        
    def _resampling(self):  
        ##
        # @brief This method resamples the data.
        ##   
        self._ppg_segments = np.array([resample(epoch, self.new_nr_sample) for epoch in self._ppg_segments])

        
    def _filt(self):
        ##
        # @brief This method does frequency filter the data.
        ##   
        for i in range(len(self._ppg_segments)):
            #self._ppg_segments[i] = medfilt(sosfiltfilt(self._sos, self._ppg_segments[i]))
            self._ppg_segments[i] = medfilt(sosfiltfilt(self._sos, self._ppg_segments[i]))

    def _make_template(self, epochs, indices, ibi):
        template0, template1, template2 = 0, 0, 0
        n_error = 0
        for idx in indices:
            if idx[0]<0:
                continue
            
            hb0 = epochs[0, idx[0]:idx[1]]
            hb1 = epochs[1, idx[0]:idx[1]]
            hb2 = epochs[2, idx[0]:idx[1]]
            
            # print(len(hb0))
            # print(ibi)
            if len(hb0)!=ibi:
                hb0 = resample(hb0, ibi)
                hb1 = resample(hb1, ibi)
                hb2 = resample(hb2, ibi)
            
            if type(template0)==int:
                template0 = hb0
                template1 = hb1
                template2 = hb2
                
            elif np.any(np.isnan(hb0)):
                n_error+=1
                continue   
            else:
                template0 += hb0
                template1 += hb1
                template2 += hb2

        template0 = resample(template0 / (len(indices)-n_error), 1000)
        template1 = resample(template1 / (len(indices)-n_error), 1000)
        template2 = resample(template2 / (len(indices)-n_error), 1000)
        return template0, template1, template2
    
        
    
    def create_templates(self):
        self.feat = np.zeros((len(self._ppg_segments), 5))
        self.template0, self.template1, self.template2 = [np.zeros((len(self._ppg_segments), self.new_nr_sample)) for i in range(3)]   
        epoch_correction = []
        for n_epoch in range(len(self._ppg_segments)):
            # try:
            epoch_info = nk.ppg_segment(self._ppg_segments[n_epoch], sampling_rate=100, return_idx=True)
            epoch_indices = []
            for key, df in epoch_info.items():
                if int(key)==len(epoch_info.items()):
                    continue
                epoch_indices.append([df["Index"].iloc[0], df["Index"].iloc[-1]]) 
            #print(np.array(epoch_indices).shape)        
            merged = np.array([self._ppg_segments[n_epoch], self._dev1[n_epoch], self._dev2[n_epoch]])
            
            ibi = int(np.median([x1-x0 for x0, x1 in epoch_indices]))
            self.template0[n_epoch], self.template1[n_epoch], self.template2[n_epoch] = self._make_template(merged, epoch_indices, ibi)
            
            std = np.std(self.template0[n_epoch])
            kurt = kurtosis(self.template0[n_epoch])
            skewness = skew(self.template0[n_epoch])
            dev1max = np.max(self.template1[n_epoch])
            
            self.feat[n_epoch] = np.array([ibi, std, kurt, skewness, dev1max])
            
            
            # except:
            #     epoch_correction.append(n_epoch)
                
            
        if len(epoch_correction) != 0:
            self._ppg_segments = np.delete(self._ppg_segments, epoch_correction, axis=0)
            self._dev1 = np.delete(self._dev1, epoch_correction, axis=0)
            self._dev2 = np.delete(self._dev2, epoch_correction, axis=0)
            self._sbp = np.delete(self._sbp, epoch_correction, axis=0)
            self._dbp = np.delete(self._dbp, epoch_correction, axis=0)
            self.template0 = np.delete(self.template0, epoch_correction, axis=0)
            self.template1 = np.delete(self.template1, epoch_correction, axis=0)
            self.template2 = np.delete(self.template2, epoch_correction, axis=0)
            self.feat = np.delete(self.feat, epoch_correction, axis=0)
            
        
    def _derivation(self):
        ##
        # @brief This method extracts the first and second derivation of the data.
        ##   
        self._dev1 = np.diff(np.pad(self._ppg_segments, ((0, 0),(0, 1)), mode='edge'), axis=-1)
        self._dev2 = np.diff(np.pad(self._dev1, ((0, 0),(0, 1)), mode='edge'), axis=-1)
        
        
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
                # Make Template
                self.create_templates()
                # Save Data
                np.save(self.target_path+'dev0/'+str(sub_id[:-4])+self.db, self._ppg_segments)
                np.save(self.target_path+'dev1/'+str(sub_id[:-4])+self.db, self._dev1)
                np.save(self.target_path+'dev2/'+str(sub_id[:-4])+self.db, self._dev2)
                np.save(self.target_path+'template/dev0/'+str(sub_id[:-4])+self.db, self.template0)
                np.save(self.target_path+'template/dev1/'+str(sub_id[:-4])+self.db, self.template1)
                np.save(self.target_path+'template/dev2/'+str(sub_id[:-4])+self.db, self.template2)
                np.save(self.target_path+'feature/'+str(sub_id[:-4])+self.db, self.feat)
                
                target = np.concatenate((self._sbp, self._dbp), axis=-1)
                if len(target.shape)==3:
                    target = np.squeeze(target, axis=0)
                np.save(self.target_path+'ground_truth/'+str(sub_id[:-4])+self.db, target)
            else:
                self.data_error=False
      
        


        
        
        
        
        
        
        
        