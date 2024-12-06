# -*- coding: utf-8 -*-
"""
Master-Studienarbeit
Autor: Lennart Brakelmann
Thema: Quantitative Erklärbarkeit tiefer neuronaler Netze
Skript: Processing of npy data for DataGenerator
"""
import numpy as np
import random
from tensorflow.keras.utils import Sequence
import os

path_main = "C:/Biomedizinische Informationstechnik/2. Semester/Projektarbeit/Code/Data/"
target_path = "C:/Biomedizinische Informationstechnik/3. Semester/Master-Studienarbeit/Data/"
files = os.listdir(path_main+"dev0_abp/")
db = 'v'


def process_data_alt(data_path, list_id, target_path, db):
    total_nr = 0
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
            dev0_segment = dev0[j]
            np.save(target_path+'dev0/'+str(total_nr)+db, dev0_segment)
            dev1_segment = dev1[j]
            np.save(target_path+'dev1/'+str(total_nr)+db, dev1_segment)
            dev2_segment = dev2[j]
            np.save(target_path+'dev2/'+str(total_nr)+db, dev2_segment)
            dev0_abp_segment = dev0_abp[j]
            np.save(target_path+'dev0_abp/'+str(total_nr)+db, dev0_abp_segment)
            dev1_abp_segment = dev1_abp[j]
            np.save(target_path+'dev1_abp/'+str(total_nr)+db, dev1_abp_segment)
            dev2_abp_segment = dev2_abp[j]
            np.save(target_path+'dev2_abp/'+str(total_nr)+db, dev2_abp_segment)
            temp0_segment = temp0[j]
            np.save(target_path+'temp0/'+str(total_nr)+db, temp0_segment)
            temp1_segment = temp1[j]
            np.save(target_path+'temp1/'+str(total_nr)+db, temp1_segment)
            temp2_segment = temp2[j]
            np.save(target_path+'temp2/'+str(total_nr)+db, temp2_segment)
            if nr_segments == 1:
                target_segment = target.reshape((1,2))
            else:
                target_segment = target[j]
                target_segment = target_segment.reshape((1,2))
            np.save(target_path+'ground_truth/'+str(total_nr)+db, target_segment)
            
            total_nr += 1
            

#process_data(path_main, files, target_path, db='v')


#signals = np.load(path_main+"dev0/"+files[36])
#z = np.load(path_main+"ground_truth/"+files[36])
#z_example = z[0]
#signal2 = signals[0]
#z_reshaped = z_example.reshape((1,2))
#Examples = np.load(target_path+'/dev0/18898v.npy')
#target_segment = target[0]
#target_segment = target.reshape((1,2))



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
            
            
#process_data(path_main, files, target_path, db='v')    


            
