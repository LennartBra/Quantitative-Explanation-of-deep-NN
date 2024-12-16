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
from make_model_template import make_model_pressure_multi, make_model_pressure_uni

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

#Define GPU settings for training of neural network 
'''
#Tensorflow settings - use CPU or GPU

#GPU settings 1
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)
#os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

#CPU settings
#tf.device('/cpu:0')


#GPU settings 2
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
'''

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
plt.hist(data_true_rounded_SBP, n_bins_true*2)
plt.hist(data_true_rounded_DBP, n_bins_true*2)
plt.xlim([20, 200])
plt.ylim([0, 400])
plt.title('Histogram of the Ground Truth data - Test Dataset - Sample Fold')
plt.xlabel('Blood Pressure')
plt.ylabel('Counts')
plt.grid()

quant_id = []
for i in range(0,len(quant_indexes)):
     quant_id.append(test_id[quant_indexes[i]])
     
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

#np.save('C:/Biomedizinische Informationstechnik/3. Semester/Master-Studienarbeit/Code/Quantitative Explanation of deep NN/ids_fold_pressure/quant_id.npy', quant_id)
#train_id2 = np.load('C:/Biomedizinische Informationstechnik/3. Semester/Master-Studienarbeit/Code/Quantitative Explanation of deep NN/ids_fold_pressure/train_id.npy')
# %%Train neural network - multivariate time series
###############################################################################
#######################   MULTIVARIATE TIME SERIES    #########################
###############################################################################
print("Train TemplateNet with PulseDB - Multivariate Time Series")
all_mae_sbp, all_mae_dbp, subject_result, all_pred, all_r_sbp, all_r_dbp = [], [], [], [], [], []

# Generators
print("Loading Datagenerator")
generator_train = DataGenerator(path_main, train_id, batch_size=batch_size, typ='ABP_multi', shuffle=False)
generator_val = DataGenerator(path_main, val_id, batch_size=batch_size, typ='ABP_multi', shuffle=False)
generator_test = DataGenerator(path_main, test_id, batch_size=batch_size, typ='ABP_multi', shuffle=False)

# %% Start Training
model_abp_multi = make_model_pressure_multi()
# Load model with weights
#model_abp_multi = keras.models.load_model('C:/Biomedizinische Informationstechnik/3. Semester/Master-Studienarbeit/Code/Quantitative Explanation of deep NN/best_model_template_pressure_multivariate-TS-Session1.h5', compile=False)


# Make training
optimizer = optimizers.Adam(learning_rate=0.0001)

es = EarlyStopping(monitor="mae", patience=10)
mcp = ModelCheckpoint('best_model_template_pressure_multivariate-TS-Session1'+'.h5',
                      monitor='val_mae', save_best_only=True)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', factor=0.1, patience=5, min_lr=1e-8)
model_abp_multi.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

model_abp_multi.fit(generator_train,
              validation_data=generator_val,
              epochs=1,
              verbose=1,
              callbacks=[es, mcp, reduce_lr])


#%% Make prediction
model_abp_multi = keras.models.load_model('C:/Biomedizinische Informationstechnik/3. Semester/Master-Studienarbeit/Code/Quantitative Explanation of deep NN/best_model_template_pressure_multivariate-TS-Session1.h5', compile=False)
nr_data = generator_test.__len__()
all_mae = np.zeros((nr_data, 2))

print('Prediction')
#for batch_index in range(0, nr_data):
for batch_index in range(0,10):
    batch_data, temp_true = generator_test.__getitem__(batch_index)
    if batch_index == 1:
        batch_data1 = batch_data
        temp_true1 = temp_true
    if batch_index == 2:
        batch_data2 = batch_data
        temp_true2 = temp_true

    print(f'batch_index: {batch_index}')

    temp_pred = model_abp_multi.predict(batch_data, verbose=0, batch_size=batch_size)
    
    if batch_index == 0:
        data_true = temp_true
        data_pred = temp_pred
    else:
        data_true = np.concatenate((data_true, temp_true), axis=0)
        data_pred = np.concatenate((data_pred, temp_pred), axis=0)

    mae_sbp_batch = mean_absolute_error(temp_pred[..., 0], temp_true[:, 0])
    mae_dbp_batch = mean_absolute_error(temp_pred[..., 1], temp_true[:, 1])

    all_mae[batch_index] = np.array([mae_sbp_batch, mae_dbp_batch])

r_sbp, _ = pearsonr(data_pred[:, 0], data_true[:, 0])
r_dbp, _ = pearsonr(data_pred[:, 1], data_true[:, 1])

mae_sbp = np.mean(all_mae[:, 0])
mae_dbp = np.mean(all_mae[:, 1])
all_pred.append(np.array(data_pred))

print(mae_sbp)
print(mae_dbp)
print(r_sbp, r_dbp)

all_mae_sbp.append(mae_sbp)
all_mae_dbp.append(mae_dbp)
all_r_sbp.append(r_sbp)
all_r_dbp.append(r_dbp)

mae_sbp_mean = np.mean(all_mae_sbp)
mae_dbp_mean = np.mean(all_mae_dbp)
r_mean_sbp = np.mean(all_r_sbp)
r_mean_dbp = np.mean(all_r_dbp)

print("Mean of SBP: ", mae_sbp_mean)
print("Mean of DBP: ", mae_dbp_mean)
print("Mean of r: ", r_mean_sbp, r_mean_dbp)

#%% Calculate Integrated Gradients for quant_ids - multivariate time series
quant_id = np.load('C:/Biomedizinische Informationstechnik/3. Semester/Master-Studienarbeit/Code/Quantitative Explanation of deep NN/ids_fold_pressure/quant_id.npy')

all_IG_pressure_SBP_multi = []
all_IG_pressure_DBP_multi = []
#for i in range(len(quant_id)):
for i in range(0,15):
    print(f'Example No:{i+1}')
    segment_tensor = XAI.make_tf_tensor_from_quant_id(path_main, quant_id[i], typ='ABP_multi')
    
    IG_pressure_SBP_multi, IG_pressure_DBP_multi = XAI.get_integrated_gradients(segment_tensor, model=model_abp_multi, baseline=None, num_steps=50)
    all_IG_pressure_SBP_multi.append(IG_pressure_SBP_multi)
    all_IG_pressure_DBP_multi.append(IG_pressure_DBP_multi)
    #np.save('C:/Biomedizinische Informationstechnik/3. Semester/Master-Studienarbeit/IG/TemplateNetPressure_Multi/SBP/'+str(i), IG_pressure_SBP_multi)
    #np.save('C:/Biomedizinische Informationstechnik/3. Semester/Master-Studienarbeit/IG/TemplateNetPressure_Multi/SBP/'+str(i), IG_pressure_DBP_multi)
    
#%% Calculate Metrics for all quant_ids
all_AOPC_SBP_multi = []
all_AOPC_DBP_multi = []
all_APT_SBP_multi = []
all_APT_DBP_multi = []

#for i in range(0,len(quant_id)):
for i in range(0,15):
    #IG_pressure_SBP = np.load('C:/Biomedizinische Informationstechnik/3. Semester/Master-Studienarbeit/IG/TemplateNetPressure_Multi/SBP/'+str(i))
    #IG_pressure_SBP = np.load('C:/Biomedizinische Informationstechnik/3. Semester/Master-Studienarbeit/IG/TemplateNetPressure_Multi/SBP/'+str(i))
    segment_tensor = XAI.make_tf_tensor_from_quant_id(path_main, quant_id[i], typ='ABP_multi')
    #AOPC_SBP_multi = metrics.calculate_AOPC(segment_tensor ,IG_pressure_SBP , k=10, pattern='morf', window_length=5, replacement_strategy='global_mean', model=model_abp_multi)
    AOPC_SBP_multi, y_pred_SBP_x = metrics.calculate_AOPC(segment_tensor, all_IG_pressure_SBP_multi[i], k=15, pattern='morf', window_length=5, replacement_strategy='global_mean', model=model_abp_multi)
    AOPC_DBP_multi, y_pred_DBP_x = metrics.calculate_AOPC(segment_tensor, all_IG_pressure_DBP_multi[i], k=15, pattern='morf', window_length=5, replacement_strategy='global_mean', model=model_abp_multi)
    all_AOPC_SBP_multi.append(AOPC_SBP_multi)
    all_AOPC_DBP_multi.append(AOPC_DBP_multi)
    APT_SBP_multi, k_SBP = metrics.calculate_APT(segment_tensor, all_IG_pressure_SBP_multi[i], alpha=0.05 , pattern='morf', window_length=5, replacement_strategy='global_mean', model=model_abp_multi, mode='SBP')
    APT_DBP_multi, k_DBP = metrics.calculate_APT(segment_tensor, all_IG_pressure_DBP_multi[i], alpha=0.05 ,pattern='morf', window_length=5, replacement_strategy='global_mean', model=model_abp_multi, mode='SBP')
    all_APT_SBP_multi.append(APT_SBP_multi)
    all_APT_DBP_multi.append(APT_DBP_multi)
    
#%% Test Importance of 995th sample
all_AOPC_SBP_multi = []
all_AOPC_DBP_multi = []
all_diffs_SBP = []
all_diffs_DBP = []

for i in range(0,15):
    #IG_pressure_SBP = np.load('C:/Biomedizinische Informationstechnik/3. Semester/Master-Studienarbeit/IG/TemplateNetPressure_Multi/SBP/'+str(i))
    #IG_pressure_SBP = np.load('C:/Biomedizinische Informationstechnik/3. Semester/Master-Studienarbeit/IG/TemplateNetPressure_Multi/SBP/'+str(i))
    segment_tensor = XAI.make_tf_tensor_from_quant_id(path_main, quant_id[i], typ='ABP_multi')
    #AOPC_SBP_multi = metrics.calculate_AOPC(segment_tensor ,IG_pressure_SBP , k=10, pattern='morf', window_length=5, replacement_strategy='global_mean', model=model_abp_multi)
    AOPC_SBP_multi, diff_SBP = metrics.calculate_AOPC(segment_tensor, all_IG_pressure_SBP_multi[i], k=100, pattern='morf', window_length=1, replacement_strategy='global_mean', model=model_abp_multi)
    AOPC_DBP_multi, diff_DBP = metrics.calculate_AOPC(segment_tensor, all_IG_pressure_DBP_multi[i], k=100, pattern='morf', window_length=1, replacement_strategy='global_mean', model=model_abp_multi)
    all_AOPC_SBP_multi.append(AOPC_SBP_multi)
    all_AOPC_DBP_multi.append(AOPC_DBP_multi)
    all_diffs_SBP.append(diff_SBP)
    all_diffs_DBP.append(diff_DBP)
    
#%% Mirroring of signal
test_segment = XAI.make_segment_from_quant_id(path_main, quant_id[0], typ='ABP_multi')
normal_segment = XAI.make_segment_from_quant_id(path_main, quant_id[0], typ='ABP_multi')
normal_segment_tensor = XAI.make_tf_tensor_from_quant_id(path_main, quant_id[0], typ='ABP_multi')

#Continue ABP Signal with ABP signal
ABP_signal = test_segment[3].copy()
ABP_signal_flipped = np.flip(ABP_signal)
ABP_one = np.concatenate((ABP_signal, ABP_signal_flipped), axis=0)
ABP_new = resample(ABP_one[0:1200], num=1000)

test_segment[3] = ABP_new
test_segment_tensor = XAI.make_tf_tensor_from_Segment(test_segment, typ='ABP_multi')

#IG_SBP_test, IG_DBP_test = XAI.get_integrated_gradients(test_segment_tensor, model=model_abp_multi)
#IG_SBP_normal, IG_DBP_normal = XAI.get_integrated_gradients(normal_segment_tensor, model=model_abp_multi)

vs.subplot_3_signals_bwr_heatmap(IG_DBP_test, test_segment, colorbar='single', mode='ABP_multi')
vs.subplot_3_signals_bwr_heatmap(IG_DBP_normal, normal_segment, colorbar='single', mode='ABP_multi')

# %%Train neural network - univariate time series
###############################################################################
########################   UNIVARIATE TIME SERIES    ##########################
###############################################################################
#Define batch size
batch_size = 64

#Define variables for evaluation on test data
print("Train TemplateNet with PulseDB")
all_mae_sbp, all_mae_dbp, subject_result, all_pred, all_r_sbp, all_r_dbp = [], [], [], [], [], []

# Generators
print("Loading Datagenerator")
generator_train = DataGenerator(path_main, train_id, batch_size=batch_size, typ='ABP_single', shuffle=False)
generator_val = DataGenerator(path_main, val_id, batch_size=batch_size, typ='ABP_single', shuffle=False)
generator_test = DataGenerator(path_main, test_id, batch_size=batch_size, typ='ABP_single', shuffle=False)

# %% Start Training
#model_abp_uni = make_model_pressure_uni()
# Load model with weights
#model_abp_uni = keras.models.load_model('C:/Biomedizinische Informationstechnik/3. Semester/Master-Studienarbeit/Code/Quantitative Explanation of deep NN/best_model_template_pressure_univariate-TS.h5', compile=False)
model_abp_uni = keras.models.load_model('C:/Biomedizinische Informationstechnik/3. Semester/Master-Studienarbeit/Code/Quantitative Explanation of deep NN/best_model_template_pressure_univariate-TS-Session2.h5', compile=False)
'''
# Make training
optimizer = optimizers.Adam(learning_rate=0.0001)

es = EarlyStopping(monitor="mae", patience=10)
mcp = ModelCheckpoint('best_model_template_pressure_univariate-TS_session2' +
                      '.h5', monitor='val_mae', save_best_only=True)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', factor=0.1, patience=5, min_lr=1e-8)
model_abp.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

model_abp.fit(generator_train,
              validation_data=generator_val,
              epochs=4,
              verbose=1,
              callbacks=[es, mcp, reduce_lr])
'''
# Make prediction
nr_data = generator_test.__len__() 
all_mae = np.zeros((nr_data, 2))

print('Prediction')
for batch_index in range(0, nr_data):
    # for batch_index in range(0,1):
    batch_data, temp_true = generator_test.__getitem__(batch_index)
    #print(f'batch_index: {batch_index}')

    if batch_index == 0:
        data_true = temp_true
        data_pred = temp_pred
    else:
        data_true = np.concatenate((data_true, temp_true), axis=0)
        data_pred = np.concatenate((data_pred, temp_pred), axis=0)

    mae_sbp_batch = mean_absolute_error(temp_pred[..., 0], temp_true[:, 0])
    mae_dbp_batch = mean_absolute_error(temp_pred[..., 1], temp_true[:, 1])

    all_mae[batch_index] = np.array([mae_sbp_batch, mae_dbp_batch])

r_sbp, _ = pearsonr(data_pred[:, 0], data_true[:, 0])
r_dbp, _ = pearsonr(data_pred[:, 1], data_true[:, 1])

mae_sbp = np.mean(all_mae[:, 0])
mae_dbp = np.mean(all_mae[:, 1])
all_pred.append(np.array(data_pred))

print(mae_sbp)
print(mae_dbp)
print(r_sbp, r_dbp)

all_mae_sbp.append(mae_sbp)
all_mae_dbp.append(mae_dbp)
all_r_sbp.append(r_sbp)
all_r_dbp.append(r_dbp)

mae_sbp_mean = np.mean(all_mae_sbp)
mae_dbp_mean = np.mean(all_mae_dbp)
r_mean_sbp = np.mean(all_r_sbp)
r_mean_dbp = np.mean(all_r_dbp)

print("Mean of SBP: ", mae_sbp_mean)
print("Mean of DBP: ", mae_dbp_mean)
print("Mean of r: ", r_mean_sbp, r_mean_dbp)


#%% Caclulate IG for quant_ids with univariate time series
model_abp_uni = keras.models.load_model('C:/Biomedizinische Informationstechnik/3. Semester/Master-Studienarbeit/Code/Quantitative Explanation of deep NN/best_model_template_pressure_univariate-TS-Session2.h5', compile=False)

quant_id = np.load('C:/Biomedizinische Informationstechnik/3. Semester/Master-Studienarbeit/Code/Quantitative Explanation of deep NN/ids_fold_pressure/quant_id.npy')

all_IG_pressure_SBP_uni = []
all_IG_pressure_DBP_uni = []
#for i in range(len(quant_id)):
for i in range(0,15):
    print(f'Example No:{i+1}')
    segment_tensor = XAI.make_tf_tensor_from_quant_id(path_main, quant_id[i], typ='ABP_single')
    
    IG_pressure_SBP_uni, IG_pressure_DBP_uni = XAI.get_integrated_gradients(segment_tensor, model=model_abp_uni, baseline=None, num_steps=50)
    all_IG_pressure_SBP_uni.append(IG_pressure_SBP_uni)
    all_IG_pressure_DBP_uni.append(IG_pressure_DBP_uni)
    #np.save('C:/Biomedizinische Informationstechnik/3. Semester/Master-Studienarbeit/IG/TemplateNetPressure_Multi/SBP/'+str(i), IG_pressure_SBP_multi)
    #np.save('C:/Biomedizinische Informationstechnik/3. Semester/Master-Studienarbeit/IG/TemplateNetPressure_Multi/SBP/'+str(i), IG_pressure_DBP_multi)
    
#%% Visualize examples
subject_nr = 13
Segment = np.load(path_main+quant_id[subject_nr])
vs.plot_onesignal_bwr_heatmap(all_IG_pressure_SBP_uni[subject_nr], Segment[3], signal_nr=0)
#vs.subplot_3_signals_bwr_heatmap(all_IG_pressure_SBP_multi[subject_nr], Segment, colorbar='single', mode='ABP_multi')

# %% Visualize Results
subject_nr = 2
Segment0 = np.load(path_main+quant_id[0])
Segment1 = np.load(path_main+quant_id[1])
Segment2 = np.load(path_main+quant_id[2])
#IG_normalized = normalize_IG(all_IG_pressure_SBP[subject_nr], method='zero_mean')

#vs.subplot_3_signals_bwr_heatmap(all_IG_pressure_SBP[subject_nr], all_input_signals, subject_nr, colorbar='single', mode='PPG')
#vs.subplot_3_signals_bwr_heatmap(IG_normalized, all_input_signals, subject_nr, colorbar='single', mode='PPG')

#vs.plot_onesignal_bwr_heatmap(all_IG_pressure_SBP[subject_nr], ABP[0][subject_nr], signal_nr=0)
#vs.plot_onesignal_bwr_heatmap(all_IG_pressure_SBP[1], Segment1[3] , signal_nr=0)
#vs.plot_onesignal_bwr_heatmap(all_IG_pressure_SBP[1], Segment1[4] , signal_nr=1)
#vs.plot_onesignal_bwr_heatmap(all_IG_pressure_SBP[1], Segment1[5] , signal_nr=2)



for i in range(13,14): #3, 10, 14
    subject_nr = i
    Segment = np.load(path_main+quant_id[i])
    #IG_normalized = XAI.normalize_IG(all_IG_pressure_SBP[subject_nr], method='zero_mean')
    #IG_normalized = np.expand_dims(IG_normalized, axis=1)
    #vs.subplot_3_signals_bwr_heatmap(IG_normalized, Segment, colorbar='single', mode='ABP_multi')
    vs.subplot_3_signals_bwr_heatmap(all_IG_pressure_SBP_multi[i], Segment, colorbar='single', mode='ABP_multi')

'''
IG_test = all_IG_pressure_SBP_multi[13].copy()
IG_test[0][0][995] = 2.5
Segment14 = np.load(path_main+quant_id[13])
vs.subplot_3_signals_bwr_heatmap(IG_test, Segment, colorbar='single', mode='ABP_multi')
#vs.subplot_3_signals_bwr_heatmap(all_IG_pressure_DBP[subject_nr], Segment2, colorbar='single', mode='ABP_multi')
'''
#%% Test section
instance_tensor = XAI.make_tf_tensor_from_quant_id(path_main, quant_id[0], typ='ABP_single')

IG_pressure_SBP, IG_pressure_DBP = XAI.get_integrated_gradients(instance_tensor, baseline=None, num_steps=50, model=model_abp_uni)


Instance_np = XAI.make_instance(batch_data, index=1, batch_size=batch_size, n_signals=1)
Instance_tf = XAI.make_input_tensor(Instance_np, n_signals=1)
a1, a2 = XAI.get_integrated_gradients(Instance_tf[0], model_abp_uni, baseline=None, num_steps=50)

#%% Test Data Generator
#generator_train = DataGenerator(path_main, train_id, batch_size=batch_size, typ='ABP_multi', shuffle=False)
#generator_val = DataGenerator(path_main, val_id, batch_size=batch_size, typ='ABP_multi', shuffle=False)
generator_test = DataGenerator(path_main, files, batch_size=batch_size, typ='ABP_multi', shuffle=False)

old_path = 'C:/Biomedizinische Informationstechnik/2. Semester/Projektarbeit/Code/Data/'
old_files = os.listdir(old_path+'dev0/')
#old_generator_train = DGT.DataGenerator(old_path, train_id, batch_size=batch_size, shuffle=False)
#old_generator_val = DGT.DataGenerator(old_path, val_id, batch_size=batch_size, shuffle=False)
old_generator_test = DGT.DataGenerator_Pressure(old_path, old_files, batch_size=batch_size, shuffle=False)

length = generator_test.__len__()

for i in range(0,length):
    if i==0:
        old_batch_data0, old_temp_true0 = old_generator_test.__getitem__(0)
        batch_data0, temp_true0 = generator_test.__getitem__(0)
    if i ==12000:
        old_batch_data1, old_temp_true1 = old_generator_test.__getitem__(i)
        batch_data1, temp_true1 = generator_test.__getitem__(i)        
    if i == 2274:
        old_batch_data2, old_temp_true2 = old_generator_test.__getitem__(i)
        batch_data2, temp_true2 = generator_test.__getitem__(i)




    
    
