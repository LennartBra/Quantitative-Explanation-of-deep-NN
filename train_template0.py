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

from DataGenerator_template import DataGenerator, DataGenerator_Pressure
from make_model_template import make_model, make_model_pressure

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

target_path = "C:/Biomedizinische Informationstechnik/2. Semester/Projektarbeit/Code/Data/"
#Make Matlab like plots
#%matplotlib qt

#%% Main code for training and testing neural network with PPG
#---------------------------------------------------------------------------------------------------------------------
# Initialize paths
#---------------------------------------------------------------------------------------------------------------------

# Main path of final preprocessed data
#path_main = "PulseDB/pulsedb0/"
path_main = "C:/Biomedizinische Informationstechnik/2. Semester/Projektarbeit/Code/Data/"
# IDs
files = os.listdir(path_main+"dev0/")
### Necessary for using subset ###
#files = files[:10]
##################################

#---------------------------------------------------------------------------------------------------------------------
# Initiliaze trainingsparameter
#---------------------------------------------------------------------------------------------------------------------
# loo = LeaveOneOut()
# loo.get_n_splits(files)

n_splits = 3
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
kfold.get_n_splits(files)

batch_size = 2048
    

if __name__=="__main__":
    print("Train TemplateNet with PulseDB")
    all_mae_sbp, all_mae_dbp, subject_result, all_pred, all_r_sbp, all_r_dbp = [], [], [], [], [], []
    for nr_fold, (train_index, test_index) in enumerate(kfold.split(files)):
        if nr_fold in [1,2]:continue

        # Separate training, validation and test ids
        train_index, val_index = train_test_split(train_index, test_size=0.1, random_state=42)
        
        train_id = [files[x] for x in train_index]
        val_id = [files[x] for x in val_index]
        test_id = [files[x] for x in test_index]
        
        #train_id = train_id[0:5]
        #val_id = val_id[0:1]
        
        # Generators
        print("Loading Datagenerator")
        generator_train = DataGenerator(path_main, train_id, batch_size=batch_size)
        generator_val = DataGenerator(path_main, val_id, batch_size=batch_size)
        generator_test = DataGenerator(path_main, test_id, batch_size=batch_size, shuffle=False)
        
        '''
        #Get Distribution of data --> train, val and test
        len_train = generator_train.__len__() * batch_size
        len_val = generator_val.__len__() * batch_size
        len_test = generator_test.__len__() * batch_size
        
        total = len_train + len_val + len_test
        percent_train = len_train / total
        percent_val = len_val / total
        percent_test = len_test / total
        
        print(f'nr_fold:{nr_fold}, percent_train:{percent_train},percent_val:{percent_val},percent_test:{percent_test}')
        '''
 
        #Load model with weights
        model = keras.models.load_model('C:/Biomedizinische Informationstechnik/2. Semester/Projektarbeit/Code/best_model_template0.h5', compile=False)
        
        '''
        # Training Code
        # Make training
        optimizer = optimizers.Adam(learning_rate=0.0001)

        es = EarlyStopping(monitor="mae", patience=10)
        mcp = ModelCheckpoint('best_model_template'+str(nr_fold)+'.h5', monitor='val_mae', save_best_only=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-8)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

        model.fit(generator_train,
                        validation_data=generator_val,
                        epochs=3,
                        verbose=1,
                        callbacks=[es, mcp, reduce_lr])
        '''
        # Make prediction on test data
        nr_data = generator_test.__len__()  #nr_data equals number of batches
        all_mae = np.zeros((nr_data,2))


        print('Prediction')
        #for batch_index in range(0,nr_data):
        for batch_index in range(0,10): #Verwendung von Batch 9 aus Testdaten für die Projetkarbeit
            print(batch_index) 
            batch_data, temp_true = generator_test.__getitem__(batch_index)
            

            temp_pred = model.predict(batch_data, verbose=0, batch_size=batch_size)
            if batch_index==0:
                data_pred = temp_pred
                data_true = temp_true
                #continue        #reinschreiben für korrelationskoeffizient --> bessere Berechnung

            if batch_index==nr_data-1:
                for i in range(len(temp_pred)):
                    if temp_pred[i,0]==0:
                        temp_pred = temp_pred[i-1] 
                        temp_true = temp_true[i-1]
                        
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

#%% Integrated Gradients Algorithm Implementation
def make_input_tensors(batch, batch_size):
    ''' 
    Computes the input signals as tf tensors for a batch of data
        
    Args:
        - batch: batch of data from the Data Generator
        - batch_size: --> predetermined in script
        
    Returns a batch of data as tf tensors
    '''
    #Define batch size and number of input signals for neuraln network as variables
    n_input_signals = 6
    l = batch_size
    
    #Make array for all instances
    all_instances = []
    
    #Iterate over all instances in the batch
    for i in range(0,l):
        instance = []
        #Iterate over all 6 input signals for an instance
        for j in range(0,n_input_signals):
            one_signal = batch[j][i]
            one_signal = np.expand_dims(one_signal, axis=0)
            instance.append(tf.cast(one_signal, tf.float32))
        all_instances.append(instance)
    
    return all_instances

def make_all_instances(batch, batch_size):
    ''' 
    Computes the input signals as np.array for a batch of data
        
    Args:
        - batch: batch of data from the Data Generator
        - batch_size: --> predetermined in script
        
    Returns a batch of data as list of numpy arrays
    '''
    #Define batch size and number of input signals for neural network as variables
    n_input_signals = 6
    l = batch_size
    
    #Make array for all instances
    all_instances = []
    
    #Iterate over all instances in the batch
    for i in range(0,l):
        instance = []
        #Iterate over all 6 input signals for an instance
        for j in range(0,n_input_signals):
            one_signal = batch[j][i]
            one_signal = np.expand_dims(one_signal, axis=0)
            #instance.append(tf.cast(one_signal, tf.float32))
            instance.append(one_signal)
        all_instances.append(instance)
    
    return all_instances


def get_gradients(instance):
    '''
    Computes the gradients of the output with respect to the input
    
    Args:
        - instance: list with all 6 input signals, input signals in the format of tf tensors
       
    Returns the gradients of the prediction with respect to the input signals
    for systolic and diastolic blood pressure
    '''
    
    #Array for all gradients
    grads = []
    
    #Get gradients with Tensorflow GradientTape class
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(instance)
        preds = model(instance)
        SBP = preds[:, 0]
        DBP = preds[:, 1]

    grads_SBP = tape.gradient(SBP, instance)
    grads_DBP = tape.gradient(DBP, instance)
    
    return grads_SBP,grads_DBP

#Gradients = grads[0].numpy()
    
def get_integrated_gradients(segment, baseline=None, num_steps=50):
    '''
    Computes the integrated gradients of one segment
    
    Args:
        - segment: one segment of data --> 10s segment
        - baseline: baseline to calculate the integrated gradients
        - num_steps: number of interpolations

    Returns the integrated gradients for the specified arguments

    '''
    shape = np.shape(segment)
    n_input_signals = shape[0]
    #print(f'inputsignals={n_input_signals}')
    #n_input_signals = 6
    
    if baseline == None:
        baseline = np.zeros((1,1000))
    elif baseline == 'Random_Signal':
        np.random.seed(0)
        baseline = np.random.default_rng().uniform(-1,1,(1,1000))
    elif baseline == 'Noise_Signal':
        noise = np.random.normal(0,1,1000)
        noise = np.expand_dims(noise, axis=0)
        baseline = segment + noise
    
    #Step 1: Do Interpolation
    interpolated_signals = []
    for step in range(num_steps+1):
        interpolated_signal = baseline +  (step/num_steps) * (segment - baseline)
        interpolated_signals.append(interpolated_signal)
    
    #Step 2: Get gradients from interpolated signals
    grads_SBP = []
    grads_DBP = []
    for i, signals in enumerate(interpolated_signals):
        print(f"Interpolated Signal {i}")
        #Make input tensor
        interpolated_signals_tensors = [tf.cast(signals[j], tf.float32) for j in range(0,n_input_signals)]
        #Get Gradients for one instance
        grad_SBP, grad_DBP = get_gradients(interpolated_signals_tensors)
        #Save gradients in list
        grads_SBP.append(grad_SBP)
        grads_DBP.append(grad_DBP)
        #print(f"i={i}, und signal={signal[0]")
        #interpolated_signals_tensors = []
        
    #Step 3: Approximate the integral
    #Make Numpy Arrays
    grads_SBP_numpy = np.array(grads_SBP)
    grads_DBP_numpy = np.array(grads_DBP)
    
    
    #Approximate the integral using the Riemann Sum/ Trapezoidal Rule
    #Riemann Sum
    sum_SBP = np.sum(grads_SBP_numpy, axis=0)
    sum_DBP = np.sum(grads_DBP_numpy, axis=0)
    #Trapezoidal Rule
    #grads_SBP_TR = grads_SBP_numpy[:-1] + grads_SBP_numpy[1:] / 2.0
    #grads_DBP_TR = grads_DBP_numpy[:-1] + grads_DBP_numpy[1:] / 2.0
    
    #Calculate Average Grads
    #Riemann Sum
    avg_grads_SBP = sum_SBP * (1/(num_steps))
    avg_grads_DBP = sum_DBP * (1/(num_steps))

    #Trapezoidal Rule
    #avg_grads_SBP = np.mean(grads_SBP_TR, axis=0)
    #avg_grads_DBP = np.mean(grads_DBP_TR, axis=0)
    
    #Step 4: Calculate integrated gradients and return
    integrated_grads_SBP = (segment - baseline) * avg_grads_SBP
    integrated_grads_DBP = (segment - baseline) * avg_grads_DBP
        
    return integrated_grads_SBP, integrated_grads_DBP


def normalize_IG(IG, method):
    IG_matrix = np.squeeze(IG.copy())
    matrix_shape = np.shape(IG_matrix)
    IG_vector = np.reshape(IG_matrix,(1,matrix_shape[0]*matrix_shape[1]))
    IG_max = np.max(IG_vector)
    IG_min = np.min(IG_vector)
    
    all_mean = []
    all_std = []
    for i in range(matrix_shape[0]):
        mean = np.mean(IG_matrix[i])
        std = np.std(IG_matrix[i])
        all_mean.append(mean)
        all_std.append(std)

    if method == 'min_max':
        IG = (IG-IG_min) / (IG_max-IG_min)
    elif method == 'max':
        IG = IG/IG_max
    elif method == 'zero_mean':
        IG = np.zeros((6,1000))
        for i in range(0,matrix_shape[0]):
            IG[i,:] = (IG_matrix[i,:]-all_mean[i])/all_std[i]
    
    return IG

def sum_IG(IG):
    IG_summed = np.sum(IG, axis=0)
    
    return IG_summed
    

#IG_SBP_normalized = normalize_IG(IG_zero_SBP[0], method='zero_mean')

#%%
#Make Input Data Tensors --> correct format for Integrated Gradients Algorithm
#all_instances = make_all_instances(batch_data, batch_size)
all_instances = make_input_tensors(batch_data, batch_size)
#Calculate one example
grads_SBP, grads_DBP = get_gradients(all_instances[1])
IG_SBP, IG_DBP = get_integrated_gradients(all_instances[1], baseline=None, num_steps=50)


#%% Calculate 100 examples with zero baseline

IG_SBP_examples_zero = []
IG_DBP_examples_zero = []
for i in range(100,300):
    print(f"Example No.{i+1} - Zero Baseline")
    #grads_SBP, grads_DBP = get_gradients(all_instances[i])
    IG_SBP, IG_DBP = get_integrated_gradients(all_instances[i], baseline=None, num_steps=50)
    IG_SBP_examples_zero.append(IG_SBP)
    IG_DBP_examples_zero.append(IG_DBP)

#np.save(target_path+'IG/IG_zero_SBP_100_300', IG_SBP_examples_zero)
#np.save(target_path+'IG/IG_zero_DBP_100_300', IG_DBP_examples_zero)

#%% Calculate 100 examples with Random baseline
IG_SBP_examples_random = []
IG_DBP_examples_random = []
for i in range(0,3):
    print(f"Example No.{i+1} - Random Baseline")
    #grads_SBP, grads_DBP = get_gradients(all_instances[i])
    IG_SBP, IG_DBP = get_integrated_gradients(all_instances[i], baseline='Random_Signal', num_steps=50)
    IG_SBP_examples_random.append(IG_SBP)
    IG_DBP_examples_random.append(IG_DBP)

#np.save(target_path+'IG/IG_random_SBP', IG_SBP_examples_random)  
#np.save(target_path+'IG/IG_random_DBP', IG_DBP_examples_random)  


#%%Load calculated IG examples
#50 Interpolation steps - Zero Baseline - Segments 0-100
IG_zero_SBP_100 = np.load(target_path+'IG/IG_zero_SBP.npy')
IG_zero_DBP_100 = np.load(target_path+'IG/IG_zero_DBP.npy')
#50 Interpolation steps - Zero Baseline - Segments 100-300
IG_zero_SBP_300 = np.load(target_path+'IG/IG_zero_SBP_100_300.npy')
IG_zero_DBP_300 = np.load(target_path+'IG/IG_zero_DBP_100_300.npy')
#Concatenate arrays
IG_zero_SBP = np.concatenate((IG_zero_SBP_100, IG_zero_SBP_300), axis=0)
IG_zero_DBP = np.concatenate((IG_zero_DBP_100, IG_zero_DBP_300), axis=0)

#50 Interpolation steps - Random Baseline - 100 segments
IG_random_SBP = np.load(target_path+'IG/IG_random_SBP.npy')
IG_random_DBP = np.load(target_path+'IG/IG_random_DBP.npy')
#25 Interpolation Steps - Zero Baseline - 100 segments
IG_zero_SBP_25steps = np.load(target_path+'IG/IG_zero_SBP_new.npy')
IG_zero_DBP_25steps = np.load(target_path+'IG/IG_zero_DBP_new.npy')

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



#%% Calculate Integrated Gradients for certain number of batches
#Define number of batches to calculate IG for
nr_batches = 10
all_IG_SBP = []
all_IG_DBP = []

for batch_index in range(0,nr_batches):
    batch_data, temp_true = generator_test.__getitem__(batch_index)
    all_instances = make_input_tensors(batch_data, batch_size)
    
    for i in range(0,len(all_instances)):
        print(f"Example No.{i+1} - Zero Baseline")
        #grads_SBP, grads_DBP = get_gradients(all_instances[i])
        IG_SBP, IG_DBP = get_integrated_gradients(all_instances[i], baseline=None, num_steps=50)
        all_IG_SBP.append(IG_SBP)
        all_IG_DBP.append(IG_DBP)
        
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
IG_SBP_normalized = normalize_IG(IG_zero_SBP[subject_nr], method='zero_mean')
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























