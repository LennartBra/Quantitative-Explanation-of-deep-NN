"""
Master Studienarbeit im Master Studiengang: Biomedizinische Informationstechnik
Titel: Quantitative Erklärbarkeit tiefer neuronaler Netze in der Analyse von Biosignalen
Autor: Lennart Brakelmann
FH Dortmund
"""
# ---------------------------------------------------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------------------------------------------------
import numpy as np
import os
import math
#os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

import DataGenerator_template as DGT
from DataGenerator_Implementation import DataGenerator
from make_model_template import make_model_pressure_multi, make_model_pressure_uni, make_model_pressure_multi_900

from scipy.signal import resample
import tensorflow as tf
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split, LeaveOneOut, KFold
from keras import optimizers
from keras.models import clone_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
import metrics as metrics
import visualization as vs
import XAI_Method as XAI
# Make Matlab like plots
# %matplotlib qt

keras.backend.clear_session()

if tf.test.gpu_device_name():
    print("Default GPU Device: {}".format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")

# ---------------------------------------------------------------------------------------------------------------------
# Initialize paths
# ---------------------------------------------------------------------------------------------------------------------

# Main path of final preprocessed data
path_main = "C:/Biomedizinische Informationstechnik/3. Semester/Master-Studienarbeit/Data/"
# All IDs
files = os.listdir(path_main)

#Define batch size
batch_size = 64

#Split data into training, validation and test data
all_indexes = []
for c in range(0, len(files)):
    all_indexes.append(c)
all_indexes = np.array(all_indexes)

# Separate training, validation and test IDs
train_index, test_index = train_test_split(all_indexes, test_size=0.1, random_state=42)
train_index, val_index = train_test_split(train_index, test_size=0.11111, random_state=42)
train_id = [files[x] for x in train_index]
val_id = [files[x] for x in val_index]
test_id = [files[x] for x in test_index]

train_id = np.load('ids_fold_pressure/train_id.npy')
val_id = np.load('ids_fold_pressure/val_id.npy')
test_id = np.load('ids_fold_pressure/test_id.npy')
quant_id = np.load('ids_fold_pressure/quant_id.npy')

#%% Make IDs for quantitative dataset
#Calculate distribution of test data
generator_test = DataGenerator(path_main, test_id, batch_size=batch_size, typ='ABP_single', shuffle=False)
nr_data = generator_test.__len__()  

calc_distribution = False
if calc_distribution == True:
    #for batch_index in range(0, nr_data-1):
    for batch_index in range(0, nr_data):
        print(batch_index)
        batch_data, temp_true = generator_test.__getitem__(batch_index)
    
        if batch_index == 0:
            data_true = temp_true
        else:
            data_true = np.concatenate((data_true, temp_true), axis=0)

#Round the Blood pressure values to generate distribution
data_true_rounded = np.round(data_true, decimals=0)

#Plot distribution of test data
n_bins_true = len(np.unique(data_true_rounded))
plt.figure()
plt.hist(data_true_rounded, n_bins_true*2)
plt.xlim([20, 200])
plt.ylim([0, 5000])
plt.title('Histogram of the Ground Truth data - whole test dataset')
plt.xlabel('Blood Pressure')
plt.ylabel('Counts')
plt.grid()

#Seperate SBP and DBP
data_true_SBP = data_true_rounded[:, 0]
data_true_DBP = data_true_rounded[:, 1]

#%% Generate quantitative distribution of data with StratifiedKFold
#Initialize StratifiedKFold
skf = StratifiedKFold(n_splits=14, shuffle=True, random_state=42)

#Get test_ids for StratifiedKFold
test_samples = test_id[0:len(data_true_SBP)]

#Split test_ids into 10 Folds and take one test set for quantitative evaluation 
for i, (train_index, test_index) in enumerate(skf.split(test_samples, data_true_SBP)):
    print(f"Fold {i}:")
    print(f"  Train: index={train_index}")
    print(f"  Test:  index={test_index}")
    #Take test set from Fold 3 as quantitative dataset
    if i == 3:
        quant_indexes = test_index

data_true_rounded_SBP = data_true_rounded[quant_indexes, 0]
data_true_rounded_DBP = data_true_rounded[quant_indexes, 1]


n_bins_true = len(np.unique(data_true_rounded_SBP))

plt.figure()
plt.hist(data_true_SBP, n_bins_true*3)
plt.hist(data_true_DBP, n_bins_true*3)
plt.xlim([20, 200])
plt.ylim([0, 5500])
plt.title('Histogram of the Ground Truth data - Test Dataset - Sample Fold')
plt.xlabel('Blood Pressure')
plt.ylabel('Counts')
plt.grid()

quant_id = []
for i in range(0,len(quant_indexes)):
     quant_id.append(test_id[quant_indexes[i]])

#quant_id = np.load('ids_fold_pressure/quant_id.npy')

y_true_quant_id_SBP = []
y_true_quant_id_DBP = []
for ID in quant_id:
    Instance = np.load(path_main+ID)
    y_true_quant_id_SBP.append(Instance[9][0])
    y_true_quant_id_DBP.append(Instance[9][1])
    
y_true_quant_id_SBP = np.round(y_true_quant_id_SBP, decimals=0)
y_true_quant_id_DBP = np.round(y_true_quant_id_DBP, decimals=0)
    
plt.figure()
plt.hist(y_true_quant_id_SBP, n_bins_true*2)
plt.hist(y_true_quant_id_DBP, n_bins_true*2)
plt.xlim([20, 200])
plt.ylim([0, 400])
plt.title('Histogram of the Ground Truth data - Quant ID')
plt.xlabel('Blood Pressure')
plt.ylabel('Counts')
plt.grid()

#Make figure of distributions for MasterStudienarbeit
fig, axs = plt.subplots(1,2)
fig.supxlabel('Blood Pressure in mmHg')
axs[0].hist(data_true_SBP, n_bins_true*3)
axs[0].hist(data_true_DBP, n_bins_true*3)
axs[0].set_xlim([20, 200])
axs[0].set_ylim([0, 5000])
axs[0].title.set_text('Test Dataset')
axs[0].set_ylabel('Count')
axs[1].hist(y_true_quant_id_SBP, n_bins_true*2)
axs[1].hist(y_true_quant_id_DBP, n_bins_true*2)
axs[1].set_xlim([20, 200])
axs[1].set_ylim([0, 400])
axs[1].title.set_text('Quantitative Dataset')


#np.save('C:/Biomedizinische Informationstechnik/3. Semester/Master-Studienarbeit/Code/Quantitative Explanation of deep NN/ids_fold_pressure/quant_id.npy', quant_id)
#train_id2 = np.load('C:/Biomedizinische Informationstechnik/3. Semester/Master-Studienarbeit/Code/Quantitative Explanation of deep NN/ids_fold_pressure/train_id.npy')
# %%Train neural network - multivariate time series
print("Train TemplateNet with PulseDB - Multivariate Time Series")
all_mae_sbp, all_mae_dbp, subject_result, all_pred, all_r_sbp, all_r_dbp = [], [], [], [], [], []

# Generators
print("Loading Datagenerator")
generator_train = DataGenerator(path_main, train_id, batch_size=batch_size, typ='ABP_multi', shuffle=False)
generator_val = DataGenerator(path_main, val_id, batch_size=batch_size, typ='ABP_multi', shuffle=False)
generator_test = DataGenerator(path_main, test_id, batch_size=batch_size, typ='ABP_multi', shuffle=False)

# %% Start Training
#model_abp_multi = make_model_pressure_multi()
# Load model with weights
model_abp_multi = keras.models.load_model('models/best_model_pressure_multivariate_datanew_epoch2.h5', compile=False)


# Make training
optimizer = optimizers.Adam(learning_rate=0.0001)

es = EarlyStopping(monitor="mae", patience=10)
mcp = ModelCheckpoint('best_model_pressure_multivariate_datanew_epoch3'+'.h5',
                      monitor='val_mae', save_best_only=True)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', factor=0.1, patience=5, min_lr=1e-8)
model_abp_multi.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

model_abp_multi.fit(generator_train,
              validation_data=generator_val,
              epochs=1,
              verbose=1,
              callbacks=[es, mcp, reduce_lr])


#%% Make prediction on quantitative dataset and get MAE
model_abp_multi = keras.models.load_model('models/best_model_pressure_multivariate_datanew_epoch2.h5', compile=False)

all_preds_SBP = []
all_preds_DBP = []
all_ytrue_SBP = []
all_ytrue_DBP = []

print('Make Prediction on quantitative dataset')
#for i in range(0,len(quant_id)):
for i in range(0,500):    
    segment_tensor = XAI.make_tf_tensor_from_quant_id(path_main, quant_id[i], typ='ABP_multi')
    segment = XAI.make_segment_from_quant_id(path_main, quant_id[i])
    print(f'Segment: {i}/{len(quant_id)}')
    pred_segment = model_abp_multi.predict(segment_tensor, verbose=0)
    
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
path_main = 'C:/Biomedizinische Informationstechnik/3. Semester/Master-Studienarbeit/Data_new/'
#model_abp_multi = keras.models.load_model('best_model_template_pressure_multivariate-TS-Session2.h5')
model_abp_multi = keras.models.load_model('models/best_model_pressure_multivariate_datanew_epoch2.h5', compile=False)

#for i in range(len(quant_id)):
for i in range(55,56):
    print(f'Example No:{i}')
    segment_tensor = XAI.make_tf_tensor_from_quant_id(path_main, quant_id[i], typ='ABP_multi')
    
    IG_SBP_ABP, IG_DBP_ABP = XAI.get_integrated_gradients(segment_tensor, model=model_abp_multi, baseline=None, num_steps=50)
    #np.save('C:/Biomedizinische Informationstechnik/3. Semester/Master-Studienarbeit/IG/TemplateNetPressure_Multi/SBP/quant_id_'+str(i), IG_SBP_ABP)
    #np.save('C:/Biomedizinische Informationstechnik/3. Semester/Master-Studienarbeit/IG/TemplateNetPressure_Multi/SBP/quant_id_'+str(i), IG_DBP_ABP)
    
#%% Calculate AOPC for all quant_ids
#Load variables
quant_id = np.load('ids_fold_pressure/quant_id.npy')
model_ABP = keras.models.load_model('models/best_model_pressure_multivariate_datanew_epoch2.h5', compile=False)

#Define Hyperparameters for AOPC
k = 15
pattern = 'morf'
window_length = 10
replacement_strategy = 'global_mean'

#Make list for all sums
all_sums_SBP_ABP = []
all_sums_DBP_ABP = []
#Loop over all instances of quantitative dataset
#for i in range(0,len(quant_id)):
for i in range(0,20):
    print(f'ID: {i}/{len(quant_id)}')
    #Load Integrated Gradients
    IG_SBP = np.load('C:/Biomedizinische Informationstechnik/3. Semester/Master-Studienarbeit/IG/TemplateNetABP/SBP/quant_id_'+str(i)+'.npy')
    IG_DBP = np.load('C:/Biomedizinische Informationstechnik/3. Semester/Master-Studienarbeit/IG/TemplateNetABP/DBP/quant_id_'+str(i)+'.npy')
    #Load Segment data
    segment_tensor = XAI.make_tf_tensor_from_quant_id(path_main, quant_id[i], typ='ABP_multi')
    #Calculate AOPC sum for segment with Integrated Gradients
    summe_SBP = metrics.calculate_AOPC_sum(segment_tensor, IG_SBP, k, pattern, window_length, replacement_strategy='global_mean', model=model_ABP)
    summe_DBP = metrics.calculate_AOPC_sum(segment_tensor, IG_DBP, k, pattern, window_length, replacement_strategy='global_mean', model=model_ABP)
    
    all_sums_SBP_ABP.append(summe_SBP)
    all_sums_DBP_ABP.append(summe_DBP)
 
#Make np arrays    
all_sums_SBP_ABP = np.squeeze(np.array(all_sums_SBP_ABP))
all_sums_DBP_ABP = np.squeeze(np.array(all_sums_DBP_ABP))
#Calculate mean over whole dataset
mean_SBP_ABP = np.mean(all_sums_SBP_ABP[:,0])
mean_DBP_ABP = np.mean(all_sums_DBP_ABP[:,1])
#Calculate final AOPC values
AOPC_SBP_ABP = (1/(k+1)) * mean_SBP_ABP
AOPC_DBP_ABP = (1/(k+1)) * mean_DBP_ABP


#%% Calculate APT for all quant_ids
#Load variables
quant_id = np.load('ids_fold_pressure/quant_id.npy')
model_ABP = keras.models.load_model('models/best_model_pressure_multivariate_datanew_epoch2.h5', compile=False)

#Define Hyperparameters for APT
alpha = 0.05
pattern = 'morf'
window_length = 10
replacement_strategy = 'global_mean'

#Make list for all APT scores
#all_APT_SBP_ABP = []
#all_APT_DBP_ABP = []
#Loop over all instances of quantitative dataset
#for i in range(0,len(quant_id)):
for i in range(16,17):
    print(f'ID: {i}/{len(quant_id)}')
    #Load Integrated Gradients
    IG_SBP = np.load('C:/Biomedizinische Informationstechnik/3. Semester/Master-Studienarbeit/IG/TemplateNetABP/SBP/quant_id_'+str(i)+'.npy')
    IG_DBP = np.load('C:/Biomedizinische Informationstechnik/3. Semester/Master-Studienarbeit/IG/TemplateNetABP/DBP/quant_id_'+str(i)+'.npy')
    #Load Segment data
    segment_tensor = XAI.make_tf_tensor_from_quant_id(path_main, quant_id[i], typ='ABP_multi')
    #Calculate APT for segment with Integrated Gradients
    APT_SBP, k_SBP = metrics.calculate_APT(segment_tensor, IG_SBP, alpha , pattern, window_length, replacement_strategy, model=model_ABP, mode='SBP')
    APT_DBP, k_DBP = metrics.calculate_APT(segment_tensor, IG_DBP, alpha ,pattern, window_length, replacement_strategy, model=model_ABP, mode='DBP')
    
    #all_APT_SBP_ABP.append(APT_SBP)
    #all_APT_DBP_ABP.append(APT_DBP)
    
#APT_SBP_ABP = np.mean(all_APT_SBP_ABP)
#APT_DBP_ABP = np.mean(all_APT_DBP_ABP)
    
    
    
    
    
    
    
    
###############################################################################
##############################       TEST      ################################
###############################################################################
#%% Plot quant_id[16] --> APT of 1
segment = XAI.make_segment_from_quant_id(path_main, quant_id[16])
vs.subplot_input_signals(segment, mode='ABP_multi')

IG_DBP_16 = np.load('C:/Biomedizinische Informationstechnik/3. Semester/Master-Studienarbeit/IG/TemplateNetABP/DBP/quant_id_'+str(16)+'.npy')
vs.subplot_3_signals_bwr_heatmap(IG_DBP_16, segment, colorbar='single', mode='ABP_multi')

#%% Plot some DBP examples
for i in range(20,50):
    segment_tensor = XAI.make_tf_tensor_from_quant_id(path_main, quant_id[i], typ='ABP_multi')
    segment = XAI.make_segment_from_quant_id(path_main, quant_id[i])
    pred = model_ABP.predict(segment_tensor)
    true = segment[9,1]
    print(f'Segment:{i}, y_pred: {pred[0,1]}, y_true: {true}')
    IG_DBP = np.load('C:/Biomedizinische Informationstechnik/3. Semester/Master-Studienarbeit/IG/TemplateNetABP/DBP/quant_id_'+str(i)+'.npy')
    #vs.subplot_3_signals_bwr_heatmap(IG_DBP, segment, colorbar='single', mode='ABP_multi')


