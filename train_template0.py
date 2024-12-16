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
config.gpu_options.per_process_gpu_memory_fraction = 0.2
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
'''


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
model = keras.models.load_model('C:/Biomedizinische Informationstechnik/2. Semester/Projektarbeit/Code/best_model_template0.h5', compile=False)
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
model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

model.fit(generator_train,
                validation_data=generator_val,
                epochs=1,
                verbose=1,
                callbacks=[es, mcp, reduce_lr])


#%% Make prediction
print('Prediction on test data')
#Define variables
all_mae_sbp, all_mae_dbp, subject_result, all_pred, all_r_sbp, all_r_dbp = [], [], [], [], [], []

nr_data = generator_test.__len__()  #nr_data equals number of batches
all_mae = np.zeros((nr_data,2))


for batch_index in range(0,nr_data): #Verwendung von Batch 9 aus Testdaten für die Projetkarbeit
    print(batch_index) 
    batch_data, temp_true = generator_test.__getitem__(batch_index)
    
    temp_pred = model.predict(batch_data, verbose=0, batch_size=batch_size)
    
    if batch_index==0:
        data_pred = temp_pred
        data_true = temp_true
        #continue        #reinschreiben für korrelationskoeffizient --> bessere Berechnung
    else:
        data_pred = np.concatenate((data_pred, temp_pred), axis=0)
        data_true = np.concatenate((data_true, temp_true), axis=0)
        

                

    mae_sbp_batch = mean_absolute_error(temp_pred[...,0], temp_true[:,0])
    mae_dbp_batch = mean_absolute_error(temp_pred[...,1], temp_true[:,1])

    all_mae[batch_index] = np.array([mae_sbp_batch, mae_dbp_batch])
r_sbp, _ = pearsonr(data_pred[:,0], data_true[:,0]) 
r_dbp, _ = pearsonr(data_pred[:,1], data_true[:,1]) 

mae_sbp = np.mean(all_mae[:,0])
mae_dbp = np.mean(all_mae[:,1])
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
model = keras.models.load_model('C:/Biomedizinische Informationstechnik/2. Semester/Projektarbeit/Code/best_model_template0.h5', compile=False)

all_IG_PPG_SBP = []
all_IG_PPG_DBP = []
#for i in range(len(quant_id)):
for i in range(0,1):
    print(f'Example No:{i+1}')
    segment_tensor = XAI.make_tf_tensor_from_quant_id(path_main, quant_id[i], typ='PPG')
    
    IG_PPG_SBP, IG_PPG_DBP = XAI.get_integrated_gradients(segment_tensor, model=model, baseline=None, num_steps=50)
    #all_IG_pressure_SBP_multi.append(IG_pressure_SBP_multi)
    #all_IG_pressure_DBP_multi.append(IG_pressure_DBP_multi)
    #np.save('C:/Biomedizinische Informationstechnik/3. Semester/Master-Studienarbeit/IG/TemplateNetPressure_Multi/SBP/'+str(i), IG_pressure_SBP_multi)
    #np.save('C:/Biomedizinische Informationstechnik/3. Semester/Master-Studienarbeit/IG/TemplateNetPressure_Multi/SBP/'+str(i), IG_pressure_DBP_multi)

#%% Calculate Metrics for all quant_ids
all_AOPC_SBP_PPG = []
all_AOPC_DBP_PPG = []
all_APT_SBP_PPG = []
all_APT_DBP_PPG = []

#for i in range(0,len(quant_id)):
for i in range(0,15):
    IG_SBP = np.load('C:/Biomedizinische Informationstechnik/3. Semester/Master-Studienarbeit/IG/TemplateNetPPG/SBP/quant_id_'+str(i)+'.npy')
    IG_DBP = np.load('C:/Biomedizinische Informationstechnik/3. Semester/Master-Studienarbeit/IG/TemplateNetPPG/DBP/quant_id_'+str(i)+'.npy')
    #IG_pressure_SBP = np.load('C:/Biomedizinische Informationstechnik/3. Semester/Master-Studienarbeit/IG/TemplateNetPressure_Multi/SBP/'+str(i))
    #IG_pressure_SBP = np.load('C:/Biomedizinische Informationstechnik/3. Semester/Master-Studienarbeit/IG/TemplateNetPressure_Multi/SBP/'+str(i))
    segment_tensor = XAI.make_tf_tensor_from_quant_id(path_main, quant_id[i], typ='PPG')
    #AOPC_SBP_multi = metrics.calculate_AOPC(segment_tensor ,IG_pressure_SBP , k=10, pattern='morf', window_length=5, replacement_strategy='global_mean', model=model_abp_multi)
    AOPC_SBP_PPG, y_pred_SBP_x = metrics.calculate_AOPC(segment_tensor, IG_SBP, k=15, pattern='morf', window_length=5, replacement_strategy='global_mean', model=model)
    AOPC_DBP_PPG, y_pred_DBP_x = metrics.calculate_AOPC(segment_tensor, IG_DBP, k=15, pattern='morf', window_length=5, replacement_strategy='global_mean', model=model)
    all_AOPC_SBP_PPG.append(AOPC_SBP_PPG)
    all_AOPC_DBP_PPG.append(AOPC_DBP_PPG)
    '''
    APT_SBP_PPG, k_SBP = metrics.calculate_APT(segment_tensor, all_IG_PPG_SBP[i], alpha=0.05 , pattern='morf', window_length=5, replacement_strategy='global_mean', model=model, mode='SBP')
    APT_DBP_PPG, k_DBP = metrics.calculate_APT(segment_tensor, all_IG_PPG_DBP[i], alpha=0.05 ,pattern='morf', window_length=5, replacement_strategy='global_mean', model=model, mode='DBP')
    all_APT_SBP_PPG.append(APT_SBP_PPG)
    all_APT_DBP_PPG.append(APT_DBP_PPG)
    '''
#%%
PPG = batch_data[0]
PPG1 = batch_data[1]
PPG2 = batch_data[2]
TemplatePPG = batch_data[3]
TemplatePPG1 = batch_data[4]
TemplatePPG2 = batch_data[5]

all_input_signals = []
for i in range(0,6):
    all_input_signals.append(batch_data[i])
        
    
    
#%%
Zero_Baseline = np.zeros((6,1000))
Zero_Baseline_Tensor = []
#Iterate over all 6 input signals for the baseline
for i in range(0,len(Zero_Baseline)):
    one_signal = Zero_Baseline[i]
    one_signal = np.expand_dims(one_signal, axis=0)
    Zero_Baseline_Tensor.append(tf.cast(one_signal, tf.float32))

    

#IG_SBP_normalized = normalize_IG(IG_zero_SBP[0], method='zero_mean')

#%%
#Make Input Data Tensors --> correct format for Integrated Gradients Algorithm
#all_instances = make_all_instances(batch_data, batch_size)
all_instances = XAI.make_input_tensors(batch_data, batch_size)
#Calculate one example
grads_SBP, grads_DBP = XAI.get_gradients(all_instances[1])
IG_SBP, IG_DBP = XAI.get_integrated_gradients(all_instances[1], baseline=None, num_steps=50)


#%% Calculate 100 examples with zero baseline

IG_SBP_examples_zero = []
IG_DBP_examples_zero = []
for i in range(100,300):
    print(f"Example No.{i+1} - Zero Baseline")
    #grads_SBP, grads_DBP = get_gradients(all_instances[i])
    IG_SBP, IG_DBP = XAI.get_integrated_gradients(all_instances[i], baseline=None, num_steps=50)
    IG_SBP_examples_zero.append(IG_SBP)
    IG_DBP_examples_zero.append(IG_DBP)

#np.save(target_path+'IG/IG_zero_SBP_100_300', IG_SBP_examples_zero)
#np.save(target_path+'IG/IG_zero_DBP_100_300', IG_DBP_examples_zero)  


#%%Load calculated IG examples
target_path = 'C:/Biomedizinische Informationstechnik/2. Semester/Projektarbeit/Code/Data/IG/'
#50 Interpolation steps - Zero Baseline - Segments 0-100
IG_zero_SBP_100 = np.load(target_path+'IG/IG_zero_SBP.npy')
IG_zero_DBP_100 = np.load(target_path+'IG/IG_zero_DBP.npy')
#50 Interpolation steps - Zero Baseline - Segments 100-300
IG_zero_SBP_300 = np.load(target_path+'IG/IG_zero_SBP_100_300.npy')
IG_zero_DBP_300 = np.load(target_path+'IG/IG_zero_DBP_100_300.npy')
#Concatenate arrays
IG_zero_SBP = np.concatenate((IG_zero_SBP_100, IG_zero_SBP_300), axis=0)
IG_zero_DBP = np.concatenate((IG_zero_DBP_100, IG_zero_DBP_300), axis=0)

#%% Subplot all input signals
vs.subplot_all_input_signals(batch_data, 22, n_input_signals=6)
subject_nr = 12
vs.plot_PPG_heatmap_scatter_subplot(PPG[subject_nr], PPG1[subject_nr], PPG2[subject_nr], IG_zero_SBP[subject_nr])

vs.subplot_3_signals_bwr_heatmap(IG_zero_SBP[subject_nr], all_input_signals, subject_nr, colorbar = 'single', mode='PPG')


#%% Visualize general results of TemplateNet

#np.save(target_path+'IG/y_pred', data_pred)  
#np.save(target_path+'IG/y_true', data_true)
y_pred = np.load(target_path+'IG/y_pred.npy')
y_true = np.load(target_path+'IG/y_true.npy')
y_pred_rounded = np.round(y_pred)
y_true_rounded = np.round(y_true)

n_bins_pred = len(np.unique(y_pred_rounded))
n_bins_true = len(np.unique(y_true_rounded))

plt.figure()
plt.hist(y_pred_rounded, n_bins_pred*2)
plt.xlim([40,200])
plt.ylim([0,34000])
plt.title('Histogram of the Predictions')
plt.xlabel('Blood Pressure')
plt.ylabel('Counts')
plt.grid()
plt.figure()
plt.hist(y_true_rounded, n_bins_true*2)
plt.xlim([40,200])
plt.ylim([0,34000])
plt.title('Histogram of the Ground Truth data')
plt.xlabel('Blood Pressure')
plt.ylabel('Counts')
plt.grid()



        
#%% Calculate AOPC and APT for all IG examples
all_AOPC_SBP = []
all_AOPC_DBP = []
all_APT_SBP = []
all_APT_DBP = []
all_k_SBP = []
all_k_SBP = []

for i in range(100,150):
    print(f'Segment: {i}')
    AOPC_SBP, all_f_x_k_SBP = metrics.calculate_AOPC(all_instances[i], IG_zero_SBP[i], k=10, pattern='morf', window_length=10, replacement_strategy='global_mean', model=model)
    AOPC_DBP, all_f_x_k_DBP = metrics.calculate_AOPC(all_instances[i], IG_zero_DBP[i], k=10, pattern='morf', window_length=10, replacement_strategy='global_mean', model=model)
    all_AOPC_SBP.append(AOPC_SBP[0][0])
    all_AOPC_DBP.append(AOPC_DBP[0][1])
    APT_SBP, k_SBP = metrics.calculate_APT(all_instances[i], IG_zero_SBP[i], alpha=0.05, pattern='morf', window_length=5, replacement_strategy='global_mean', model=model, mode='SBP')
    APT_DBP, k_DBP = metrics.calculate_APT(all_instances[i], IG_zero_DBP[i], alpha=0.05, pattern='morf', window_length=5, replacement_strategy='global_mean', model=model, mode='DBP')
    all_APT_SBP.append(APT_SBP)
    all_APT_DBP.append(APT_DBP)
    
AOPC_SBP_mean = np.mean(all_AOPC_SBP)
AOPC_DBP_mean = np.mean(all_AOPC_DBP)
APT_SBP_mean = np.mean(all_APT_SBP)
APT_DBP_mean = np.mean(all_APT_DBP)
    
#%% AOPC Metric
def rank_attributions(IG, pattern, window_length):
    IG_matrix = np.squeeze(IG)
    shape = np.shape(IG_matrix)
    n_input_signals = shape[0]
    #Calculate sums ove the window_length
    distances = np.arange(0,shape[1],window_length)
    summed_attributions_mat = np.zeros((n_input_signals,len(distances)))
    for i in range(0,n_input_signals):
        for j in range(0,len(distances)):
            #print(f'input_signals: {i}, distances: {distances[j]+window_length}')
            Sum = np.sum(np.abs(IG_matrix[i,distances[j]:distances[j]+window_length]))
            summed_attributions_mat[i,j] = Sum
    
    mat_shape = np.shape(summed_attributions_mat)
    
    if pattern == 'morf': #most relevant first
        sum_atts_vector = np.reshape(summed_attributions_mat,(1,mat_shape[0]*mat_shape[1]))
        vec_shape = np.shape(sum_atts_vector)

        #Sort summed attributions in descending order --> morf
        atts_sorted_indices = np.flip(np.argsort(sum_atts_vector))

        atts_sorted_vector = np.zeros(vec_shape)
        #rank the sorted attribution indices
        for i in range(0,vec_shape[1]):
            atts_sorted_vector[0,atts_sorted_indices[0,i]] = i  
        #Reshape from vector form to matrix form
        ranks = np.reshape(atts_sorted_vector, mat_shape)
    
    if pattern == 'lerf': #least relevant first
        #Reshape summed attributions from matrix form to vector form --> e.g. (6,100) to (1,600)
        sum_atts_vector = np.reshape(summed_attributions_mat,(1,mat_shape[0]*mat_shape[1]))
        vec_shape = np.shape(sum_atts_vector)

        #Sort summed attributions in ascending order --> lerf
        atts_sorted_indices = np.argsort(sum_atts_vector)

        atts_sorted_vector = np.zeros(vec_shape)
        #rank the sorted attribution indices
        for i in range(0,vec_shape[1]):
            atts_sorted_vector[0,atts_sorted_indices[0,i]] = i  
        #Reshape from vector form to matrix form
        ranks = np.reshape(atts_sorted_vector, mat_shape)
    
    return summed_attributions_mat, ranks

IG_sum_atts, IG_ranks = rank_attributions(IG_zero_SBP[0] , pattern='morf', window_length=10)

def replace_k_features(x, ranks, k, window_length, replacement_strategy):
    x_replaced = []
    x_temp = np.squeeze(np.array(x))
    #replace features with defined replacement_strategy
    if replacement_strategy == 'local_mean':
        #replace k features
        for i in range(0,k):
            #Get index of k-th feature
            rank_index = np.where(ranks==i)
            x_index = window_length * rank_index[1][0]
            #print(f'k: {k}, x_index:{x_index}, rank_index:{rank_index[0][0]}')
            #Calculate local mean
            local_mean = np.mean(x_temp[rank_index[0][0],x_index:x_index+window_length])
            #print(f'values:{x_temp[rank_index[0][0],x_index:x_index+window_length]}, local_mean: {local_mean}')
            #Replace k-th feature with local mean
            x_temp[rank_index[0][0],x_index:x_index+window_length] = local_mean
            #Append x_temp to x_replaced
            x_replaced.append(x_temp.copy())
            
    elif replacement_strategy == 'global_mean':
        x_copy = np.squeeze(x.copy())
        #replace k features
        for i in range(0,k):
            #Get index of k-th feature
            rank_index = np.where(ranks==i)
            x_index = window_length * rank_index[1][0]
            #print(f'k: {k}, x_index:{x_index}, rank_index:{rank_index[0][0]}')
            #Calculate local mean
            global_mean = np.mean(x_copy[rank_index[0][0],:])
            #print(f'values:{x_temp[rank_index[0][0],x_index:x_index+window_length]}, global_mean: {global_mean}')
            #Replace k-th feature with local mean
            x_temp[rank_index[0][0],x_index:x_index+window_length] = global_mean
            #Append x_temp to x_replaced
            x_replaced.append(x_temp.copy())
    
    elif replacement_strategy == 'zeros':
        for i in range(0,k):
            #Get index of k-th feature
            rank_index = np.where(ranks==i)
            x_index = window_length * rank_index[1][0]
            #print(f'k: {k}, x_index:{x_index}, rank_index:{rank_index[0][0]}')
            #Replace k-th feature with zeros
            x_temp[rank_index[0][0],x_index:x_index+window_length] = 0
            #Append x_temp to x_replaced
            x_replaced.append(x_temp.copy())
    
    return x_replaced

x_replaced = replace_k_features(all_instances[0], IG_ranks, k=50, window_length=10, replacement_strategy='local_mean')

vs.subplot_all_IG_attributions(x_replaced[1])
vs.subplot_all_IG_attributions(x_replaced[49])

def calculate_AOPC(x, IG, k, pattern, window_length, replacement_strategy, model):
    #Rank attributions
    IG_sum_atts, IG_ranks = rank_attributions(IG, pattern, window_length)
    matrix_shape = np.shape(IG_sum_atts)
    #Replace k features depending on the hyperparameters
    x_replaced = replace_k_features(x, IG_ranks, k, window_length, replacement_strategy)
    #Calculate AOPC with formula
    f_x = model.predict(x, verbose=0)
    summe = 0
    all_f_x_k = []
    for i in range(0,k):
        #print(f'k: {k}')
        #Make tf Tensor from numpy array
        x_k = [tf.cast(np.expand_dims(x_replaced[i][j],axis=0), tf.float32) for j in range(0,matrix_shape[0])]
        f_x_k = model.predict(x_k, verbose=0)
        diff = f_x - f_x_k
        summe = summe + diff
        all_f_x_k.append(f_x_k)
    AOPC = 1/k * summe
    
    return AOPC, all_f_x_k


#%% APT (Ablation Percentage Threshold) Metric

def replace_feature(x, x_ground_truth, ranks, k, window_length, replacement_strategy):
    x_temp = x.copy()
    #replace features with defined replacement_strategy
    if replacement_strategy == 'local_mean':
        #replace k-th feature
        #Get index of k-th feature
        rank_index = np.where(ranks==k)
        x_index = window_length * rank_index[1][0]
        #print(f'k: {k}, x_index:{x_index}, rank_index:{rank_index[0][0]}')
        #Calculate local mean
        local_mean = np.mean(x_temp[rank_index[0][0],x_index:x_index+window_length])
        #print(f'values:{x_temp[rank_index[0][0],x_index:x_index+window_length]}, local_mean: {local_mean}')
        #Replace k-th feature with local mean
        x_temp[rank_index[0][0],x_index:x_index+window_length] = local_mean
        #Append x_temp to x_replaced
        x_replaced = x_temp
            
    elif replacement_strategy == 'global_mean':
        x_copy = x_ground_truth.copy()
        #replace k-th feature
        #Get index of k-th feature
        rank_index = np.where(ranks==k)
        x_index = window_length * rank_index[1][0]
        #print(f'k: {k}, x_index:{x_index}, rank_index:{rank_index[0][0]}')
        #Calculate local mean
        global_mean = np.mean(x_copy[rank_index[0][0],:])
        #print(f'values:{x_temp[rank_index[0][0],x_index:x_index+window_length]}, global_mean: {global_mean}')
        #Replace k-th feature with local mean
        x_temp[rank_index[0][0],x_index:x_index+window_length] = global_mean
        #Append x_temp to x_replaced
        x_replaced = x_temp.copy()
    
    elif replacement_strategy == 'zeros':
        #replace k-th
        rank_index = np.where(ranks==k)
        x_index = window_length * rank_index[1][0]
        #print(f'k: {k}, x_index:{x_index}, rank_index:{rank_index[0][0]}')
        #Replace k-th feature with zeros
        x_temp[rank_index[0][0],x_index:x_index+window_length] = 0
        #Append x_temp to x_replaced
        x_replaced = x_temp
    
    return x_replaced
    
def calculate_APT(x, IG, alpha, pattern, window_length, replacement_strategy, model, mode):
    IG_sum_atts, IG_ranks = rank_attributions(IG, pattern, window_length)
    matrix_shape = np.shape(IG_sum_atts)
    x_temp = x.copy()
    x_temp = np.squeeze(np.array(x_temp))
    x_replaced = x_temp.copy()
    J = np.size(IG_ranks)
    k = 0
    condition = False
    if mode == 'SBP':
        #Calculate Ground Truth Value
        pred = model.predict(x, verbose=0)
        SBP_true = pred[0][0]
        #Calculate specific thresholds
        SBP_Up_lim = SBP_true + (alpha*SBP_true)
        SBP_Lo_lim = SBP_true - (alpha*SBP_true)
        while condition == False:
            x_replaced = replace_feature(x_replaced, x_temp, IG_ranks, k, window_length, replacement_strategy)
            #Make tf Tensor from numpy array
            x_replaced_tensor = [tf.cast(np.expand_dims(x_replaced[j],axis=0), tf.float32) for j in range(0,matrix_shape[0])]
            pred_replaced = model.predict(x_replaced_tensor, verbose=0)
            SBP_pred = pred_replaced[0][0]
            if SBP_pred > SBP_Up_lim:
                condition = True
            elif SBP_pred < SBP_Lo_lim:
                condition = True
            else:
                k = k+1
    elif mode == 'DBP':
        pred = model.predict(x, verbose=0)
        DBP_true = pred[0][1]
        #Calculate specific thresholds
        DBP_Up_lim = DBP_true + (alpha*DBP_true)
        DBP_Lo_lim = DBP_true - (alpha*DBP_true)
        while condition == False:
            #Make tf tensor from numpy array
            x_replaced_tensor = [tf.cast(np.expand_dims(x_replaced[j],axis=0), tf.float32) for j in range(0,matrix_shape[0])]
            x_replaced = replace_feature(x_replaced, x_temp, IG_ranks, k, window_length, replacement_strategy)
            pred_replaced = model.predict(x_replaced_tensor, verbose=0)
            DBP_pred = pred_replaced[0][1]
            if DBP_pred > DBP_Up_lim:
                condition = True
            elif DBP_pred < DBP_Lo_lim:
                condition = True
            else:
                k = k+1
    elif mode == 'ABP':
        pass
    
    k = k+1
    APT = k/J
    
    return APT, k

APT, k = calculate_APT(all_instances[0], IG_zero_SBP[0], alpha=0.05, pattern='morf', window_length=5, replacement_strategy='global_mean', model=model, mode='SBP')
print(f'APT: {APT*100}%, k: {k}')
#%%
subject = 50
k = 10
window_length = 5
AOPC_SBP, all_f_x_k_SBP = metrics.calculate_AOPC(all_instances[subject], IG_zero_SBP[subject], k=k, pattern='morf', window_length=window_length, replacement_strategy='global_mean', model=model)
AOPC_DBP, all_f_x_k_DBP = metrics.calculate_AOPC(all_instances[subject], IG_zero_DBP[subject], k=10, pattern='morf', window_length=10, replacement_strategy='global_mean', model=model)
print(f'Subject: {subject} k: {k} window_length: {window_length}--> AOPC_SBP: {AOPC_SBP[0][0]}, AOPC_DBP: {AOPC_DBP[0][1]}')

#%%Test Section
#IG_summed = sum_IG(IG_zero_SBP[0])
#vs.plot_signal_heatmap(PPG[0], IG_zero_SBP[0], signal_nr=0)
subject_nr = 151
IG_SBP_normalized = XAI.normalize_IG(IG_zero_SBP[subject_nr], method='zero_mean')
IG_SBP_normalized = np.expand_dims(IG_SBP_normalized, axis=1)
#vs.subplot_all_signals_bwr_heatmap(IG_zero_SBP[subject_nr], all_input_signals, subject_nr=subject_nr, colorbar='midpoint_norm')
#vs.subplot_all_signals_bwr_heatmap(IG_zero_SBP[subject_nr], all_input_signals, subject_nr=subject_nr, colorbar='single')
#vs.subplot_all_signals_bwr_heatmap(IG_zero_SBP[subject_nr], all_input_signals, subject_nr=subject_nr, colorbar='multi')
vs.subplot_all_signals_bwr_heatmap(IG_SBP_normalized, all_input_signals, subject_nr=subject_nr, colorbar ='single')

vs.subplot_3_signals_bwr_heatmap(IG_zero_SBP[subject_nr], all_input_signals, subject_nr, colorbar = 'single', mode='PPG')

#vs.AOPC_curve_plot(temp_pred[subject][0], all_f_x_k_SBP, mode='SBP')

signal_nr = 0
subject = 28
#vs.plot_onesignal_bwr_heatmap(IG_zero_SBP[subject], PPG[subject], signal_nr)



##########################################################Breite##########
###############################################################################























