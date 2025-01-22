# -*- coding: utf-8 -*-
"""
Master-Studienarbeit
Autor: Lennart Brakelmann
Thema: Quantitative Erklärbarkeit tiefer neuronaler Netze
Skript: Processing of npy data for DataGenerator
"""
import numpy as np
import random
import os
from scipy.signal import resample
import XAI_Method as XAI

path_main = "C:/Biomedizinische Informationstechnik/2. Semester/Projektarbeit/Code/Data/"
target_path = "C:/Biomedizinische Informationstechnik/3. Semester/Master-Studienarbeit/Data/"
files = os.listdir(path_main+"dev0_abp/")
db = 'v'

#Define Function for processing of data
def process_data(data_path, list_id, target_path, db):
    total_nr = 0
    nr_signals = 10
    nr_samples = 1000
    for i in range(len(list_id)):
        print(f'Liste ID: {i} von 2938')
        ## @brief PPG input data of actual subject.
        ##
        dev0 = np.load(path_main+"dev0/"+list_id[i], allow_pickle=True)
        ## @brief First derivation of input data of actual subject.
        ##
        dev1 = np.load(path_main+"dev1/"+list_id[i], allow_pickle=True)
        ## @brief Second derivation of input data of actual subject.
        ##
        dev2 = np.load(path_main+"dev2/"+list_id[i], allow_pickle=True)
        ## @brief Templates of input data of the actual subject.
        ##
        temp0 = np.load(path_main+"template/dev0/"+list_id[i], allow_pickle=True)
        ## @brief Templates of the first derivation of input data of actual subject.
        ##
        temp1 = np.load(path_main+"template/dev1/"+list_id[i], allow_pickle=True)
        ## @brief Templates of the first derivation of input data of actual subject.
        ##
        temp2 = np.load(path_main+"template/dev2/"+list_id[i], allow_pickle=True)
        ## @brief PPG input data of actual subject --> ABP
        ##
        dev0_abp = np.load(path_main+"dev0_abp/"+list_id[i], allow_pickle=True)
        ## @brief First derivation of input data of actual subject. --> ABP
        ##
        dev1_abp = np.load(path_main+"dev1_abp/"+list_id[i], allow_pickle=True)
        ## @brief Second derivation of input data of actual subject. --> ABP
        ##
        dev2_abp = np.load(path_main+"dev2_abp/"+list_id[i], allow_pickle=True)
        ## @brief Target data of actual subject.
        ##
        target = np.load(path_main+"ground_truth/"+list_id[i])
        
        nr_segments = len(dev0)
        print(f'nr_segments: {nr_segments}')
        for j in range(nr_segments):
            X = np.ones((nr_signals,nr_samples))
            
            X[0][:] = dev0[j] #PPG Signal
            X[1][:] = dev1[j] #Erste Ableitung des PPG Signals
            X[2][:] = dev2[j] #Zeite Ableitung des PPG Signals
            X[3][:] = dev0_abp[j] #Arterielles Blutdrucksignal
            X[4][:] = dev1_abp[j] #Erste Ableitung des arteriellen Blutdrucksignals
            X[5][:] = dev2_abp[j] #Zweite Ableitung des arteriellen Blutdrucksignals
            X[6][:] = temp0[j] #Template Signal PPG
            X[7][:] = temp1[j] #Template Signal Erste Ableitung des PPG
            X[8][:] = temp2[j] #Template Signal Zweite Ableitung des PPG
            if nr_segments == 1:
                X[9][0:2] = target #Ground Truth Werte
            else:
                X[9][0:2] = target[j] #Ground Truth Werte
            
            np.save(target_path+str(total_nr)+db, X)
            
            total_nr += 1
            
#%% Process Data        
#process_data(path_main, files, target_path, db='v')    

#%% Make quantitative Data from quant_id
quant_id = np.load('ids_fold_pressure/quant_id.npy')
for i in range(0,len(quant_id)):
    print(f'Segment:{i}/{len(quant_id)}')
    Segment = np.load('C:/Biomedizinische Informationstechnik/3. Semester/Master-Studienarbeit/Data/'+quant_id[i])
    np.save('D:/Quantitative Data/'+quant_id[i], Segment) 
    

#%% Process the derivatives new
main_path = "C:/Biomedizinische Informationstechnik/3. Semester/Master-Studienarbeit/Data/"
target_path = 'C:/Biomedizinische Informationstechnik/3. Semester/Master-Studienarbeit/Data_new/'
all_segments = os.listdir(main_path)
db = 'v'

def process_derivatives(data_path, target_path, files, db):
    for i in range(0,len(files)):
        print(i)
        Segment = np.load(data_path+str(i)+db+'.npy')
        all_signals = Segment.copy()
        ABP_Signal = all_signals[3]
        first_deriv = np.diff(ABP_Signal)
        second_deriv = np.diff(first_deriv)
        first_deriv_resampled = resample(first_deriv, num=1000)
        second_deriv_resampled = resample(second_deriv, num=1000)
        
        Segment[4] = first_deriv_resampled
        Segment[5] = second_deriv_resampled
        
        #np.save(target_path+str(i)+db+'.npy', Segment)
        np.save(target_path+str(i)+db+'.npy', Segment)




#process_derivatives(main_path, target_path, all_segments, db='v')
#TestSegment1 = np.load(main_path+'0v.npy')
#TestSegment2 = np.load(target_path+'0v.npy')

#%% Compare Data
Data_old = np.load('C:/Biomedizinische Informationstechnik/3. Semester/Master-Studienarbeit/Data/836594v.npy')
Data_new = np.load('C:/Biomedizinische Informationstechnik/3. Semester/Master-Studienarbeit/Data/836594v.npy')

#%% Load quant Data
Quant_data_old = np.load('D:/Master-Studienarbeit/Quantitative Data Old/2292v.npy')
Quant_data_new = np.load('D:/Master-Studienarbeit/Quantitative Data New/2292v.npy')
Quant_data_test = np.load('C:/Biomedizinische Informationstechnik/3. Semester/Master-Studienarbeit/Data/2292v.npy')
#%% Make quantitative Data from quant_id - NEW DATA --> derivatives
quant_id = np.load('ids_fold_pressure/quant_id.npy')
for i in range(0,len(quant_id)):
    print(f'Segment:{i}/{len(quant_id)}')
    Segment = np.load('C:/Biomedizinische Informationstechnik/3. Semester/Master-Studienarbeit/Data_new/'+quant_id[i])
    np.save('H:/Quantitative Data New/'+quant_id[i], Segment) 
    
