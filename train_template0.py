"""
Master Studienarbeit im Master Studiengang: Biomedizinische Informationstechnik
Titel: Quantitative Erklärbarkeit tiefer neuronaler Netze in der Analyse von Biosignalen
Autor: Lennart Brakelmann
FH Dortmund
"""
#---------------------------------------------------------------------------------------------------------------------
# Imports
#---------------------------------------------------------------------------------------------------------------------
import numpy as np
import os
#os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

from DataGenerator_Implementation import DataGenerator
from make_model_template import make_model_PPG 

import tensorflow as tf
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split, LeaveOneOut, KFold
from keras import optimizers
from keras.models import clone_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow import keras
import matplotlib.pyplot as plt
import metrics as metrics
import visualization as vs
import XAI_Method as XAI

keras.backend.clear_session()

if tf.test.gpu_device_name():
    print("Default GPU Device: {}".format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")

#Make Matlab like plots
#%matplotlib qt

#Load train_id, test_id, val_id and quant_id
train_id = np.load("C:/Biomedizinische Informationstechnik/3. Semester/Master-Studienarbeit/IG/train_id.npy")
val_id = np.load("C:/Biomedizinische Informationstechnik/3. Semester/Master-Studienarbeit/IG/val_id.npy")
test_id = np.load("C:/Biomedizinische Informationstechnik/3. Semester/Master-Studienarbeit/IG/test_id.npy")
quant_id = np.load("C:/Biomedizinische Informationstechnik/3. Semester/Master-Studienarbeit/IG/quant_id.npy")

# Main path of final preprocessed data
path_main = "C:/Biomedizinische Informationstechnik/3. Semester/Master-Studienarbeit/Data/"
# IDs
files = os.listdir(path_main)
#Define batch size
batch_size = 64

#Make model
#model = make_model_PPG()
#Load model with weights
model_PPG = keras.models.load_model('C:/Biomedizinische Informationstechnik/2. Semester/Projektarbeit/Code/best_model_template0.h5', compile=False)
#%% Train Neural Network
print("Train TemplateNet with PulseDB")
# Load Data Generators
print("Loading Datagenerator")
generator_train = DataGenerator(path_main, train_id, batch_size=batch_size, typ='PPG')
generator_val = DataGenerator(path_main, val_id, batch_size=batch_size, typ='PPG')
generator_test = DataGenerator(path_main, test_id, batch_size=batch_size, typ='PPG')

# Make training
optimizer = optimizers.Adam(learning_rate=0.0001)

es = EarlyStopping(monitor="mae", patience=10)
mcp = ModelCheckpoint('best_model_template.h5', monitor='val_mae', save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-8)
model_PPG.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

model_PPG.fit(generator_train,
                validation_data=generator_val,
                epochs=1,
                verbose=1,
                callbacks=[es, mcp, reduce_lr])


#%% Make prediction on quantitative dataset and get MAE
model_PPG = keras.models.load_model('C:/Biomedizinische Informationstechnik/2. Semester/Projektarbeit/Code/best_model_template0.h5', compile=False)

all_preds_SBP = []
all_preds_DBP = []
all_ytrue_SBP = []
all_ytrue_DBP = []

print('Make Prediction on quantitative dataset')
#for i in range(0,len(quant_id)):
for i in range(0,500):    
    segment_tensor = XAI.make_tf_tensor_from_quant_id(path_main, quant_id[i], typ='PPG')
    segment = XAI.make_segment_from_quant_id(path_main, quant_id[i])
    print(f'Segment: {i}/{len(quant_id)}')
    pred_segment = model_PPG.predict(segment_tensor, verbose=0)
    
    all_preds_SBP.append(pred_segment[0,0])
    all_preds_DBP.append(pred_segment[0,1])

    all_ytrue_SBP.append(segment[9,0])
    all_ytrue_DBP.append(segment[9,1])

MAE_SBP = mean_absolute_error(all_ytrue_SBP,all_preds_SBP)
MAE_DBP = mean_absolute_error(all_ytrue_DBP,all_preds_DBP)

print("Mean of SBP: ", MAE_SBP)
print("Mean of DBP: ", MAE_DBP)


#%% Calculate Integrated Gradients for quant_ids - multivariate time series
quant_id = np.load('ids_fold_pressure/quant_id.npy')
model_PPG = keras.models.load_model('C:/Biomedizinische Informationstechnik/2. Semester/Projektarbeit/Code/best_model_template0.h5', compile=False)
path_main = 'C:/Biomedizinische Informationstechnik/3. Semester/Master-Studienarbeit/Data/'

for i in range(len(quant_id)):
    print(f'Example No:{i}')
    segment_tensor = XAI.make_tf_tensor_from_quant_id(path_main, quant_id[i], typ='PPG')
    
    IG_PPG_SBP, IG_PPG_DBP = XAI.get_integrated_gradients(segment_tensor, model=model_PPG, baseline=None, num_steps=50)
    #np.save('C:/Biomedizinische Informationstechnik/3. Semester/Master-Studienarbeit/IG/TemplateNetPPG/SBP/quant_id_'+str(i), IG_PPG_SBP)
    #np.save('C:/Biomedizinische Informationstechnik/3. Semester/Master-Studienarbeit/IG/TemplateNetPPG/DBP/quant_id_'+str(i), IG_PPG_DBP)

#%% Calculate AOPC for all quant_ids
#Load variables
quant_id = np.load('ids_fold_pressure/quant_id.npy')
model_PPG = keras.models.load_model('C:/Biomedizinische Informationstechnik/2. Semester/Projektarbeit/Code/best_model_template0.h5', compile=False)

#Define Hyperparameters for AOPC
k = 15
pattern = 'morf'
window_length = 10
replacement_strategy = 'global_mean'

#Make list for all sums
all_sums_SBP_PPG = []
all_sums_DBP_PPG = []
#Loop over all instances of quantitative dataset
#for i in range(0,len(quant_id)):
for i in range(0,20):
    print(f'ID: {i}/{len(quant_id)}')
    #Load Integrated Gradients
    IG_SBP = np.load('C:/Biomedizinische Informationstechnik/3. Semester/Master-Studienarbeit/IG/TemplateNetPPG/SBP/quant_id_'+str(i)+'.npy')
    IG_DBP = np.load('C:/Biomedizinische Informationstechnik/3. Semester/Master-Studienarbeit/IG/TemplateNetPPG/DBP/quant_id_'+str(i)+'.npy')
    #Load Segment data
    segment_tensor = XAI.make_tf_tensor_from_quant_id(path_main, quant_id[i], typ='PPG')
    #Calculate AOPC sum for segment with Integrated Gradients
    summe_SBP = metrics.calculate_AOPC_sum(segment_tensor, IG_SBP, k, pattern, window_length, replacement_strategy='global_mean', model=model_PPG)
    summe_DBP = metrics.calculate_AOPC_sum(segment_tensor, IG_DBP, k, pattern, window_length, replacement_strategy='global_mean', model=model_PPG)
    
    all_sums_SBP_PPG.append(summe_SBP)
    all_sums_DBP_PPG.append(summe_DBP)
 
#Make np arrays    
all_sums_SBP_PPG = np.squeeze(np.array(all_sums_SBP_PPG))
all_sums_DBP_PPG = np.squeeze(np.array(all_sums_DBP_PPG))
#Calculate mean over whole dataset
mean_SBP_PPG = np.mean(all_sums_SBP_PPG[:,0])
mean_DBP_PPG = np.mean(all_sums_DBP_PPG[:,1])
#Calculate final AOPC values
AOPC_SBP_PPG = (1/(k+1)) * mean_SBP_PPG
AOPC_DBP_PPG = (1/(k+1)) * mean_DBP_PPG


#%% Calculate APT for all quant_ids
#Load variables
quant_id = np.load('ids_fold_pressure/quant_id.npy')
model_PPG = keras.models.load_model('C:/Biomedizinische Informationstechnik/2. Semester/Projektarbeit/Code/best_model_template0.h5', compile=False)

#Define Hyperparameters for APT
alpha = 0.05
pattern = 'morf'
window_length = 10
replacement_strategy = 'global_mean'

#Make list for all APT scores
all_APT_SBP_PPG = []
all_APT_DBP_PPG = []
#Loop over all instances of quantitative dataset
#for i in range(0,len(quant_id)):
for i in range(30,60):
    print(f'ID: {i}/{len(quant_id)}')
    #Load Integrated Gradients
    IG_SBP = np.load('C:/Biomedizinische Informationstechnik/3. Semester/Master-Studienarbeit/IG/TemplateNetPPG/SBP/quant_id_'+str(i)+'.npy')
    IG_DBP = np.load('C:/Biomedizinische Informationstechnik/3. Semester/Master-Studienarbeit/IG/TemplateNetPPG/DBP/quant_id_'+str(i)+'.npy')
    #Load Segment data
    segment_tensor = XAI.make_tf_tensor_from_quant_id(path_main, quant_id[i], typ='PPG')
    #Calculate APT for segment with Integrated Gradients
    APT_SBP, k_SBP = metrics.calculate_APT(segment_tensor, IG_SBP, alpha , pattern, window_length, replacement_strategy, model=model_PPG, mode='SBP')
    APT_DBP, k_DBP = metrics.calculate_APT(segment_tensor, IG_DBP, alpha , pattern, window_length, replacement_strategy, model=model_PPG, mode='DBP')
    
    all_APT_SBP_PPG.append(APT_SBP)
    all_APT_DBP_PPG.append(APT_DBP)
    
APT_SBP_PPG = np.mean(all_APT_SBP_PPG)
APT_DBP_PPG = np.mean(all_APT_DBP_PPG)
    

#%% Plot one example for paper
#New Visualization
subject_nr = 35 #35
IG_SBP_PPG_example = np.load('C:/Biomedizinische Informationstechnik/3. Semester/Master-Studienarbeit/IG/TemplateNetPPG/SBP/quant_id_'+str(subject_nr)+'.npy')
segment = XAI.make_segment_from_quant_id(path_main, quant_id[subject_nr])
vs.subplot_3_signals_bwr_heatmap(IG_SBP_PPG_example, segment, colorbar='single', mode='PPG')
vs.plot_PPG_heatmap_scatter_subplot(segment[0], segment[1], segment[2], IG_SBP_PPG_example)

 
