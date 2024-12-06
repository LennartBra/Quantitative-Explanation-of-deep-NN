"""
Master Studienarbeit im Master Studiengang: Biomedizinische Informationstechnik
Titel: Quantitative Erklärbarkeit tiefer neuronaler Netze in der Analyse von Biosignalen
Autor: Lennart Brakelmann
FH Dortmund
"""
# ---------------------------------------------------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------------------------------------------------
from sklearn.model_selection import StratifiedKFold
import numpy as np
import os
import math
#os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

#from DataGenerator_template import DataGenerator, DataGenerator_Pressure, DataGenerator_Pressure_onesignal
from DataGenerator_Implementation import DataGenerator
from make_model_template import make_model, make_model_pressure, make_model_pressure_onesignal

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
#path_main = "PulseDB/pulsedb0/"
path_main = "C:/Biomedizinische Informationstechnik/2. Semester/Projektarbeit/Code/Data/"
# IDs
files = os.listdir(path_main+"dev0_abp/")

target_path = "C:/Biomedizinische Informationstechnik/2. Semester/Projektarbeit/Code/Data/"
# Make Matlab like plots
# %matplotlib qt

# %%Train neural network without k-fold cross validation - multivariate time series
# Main path of final preprocessed data
#path_main = "PulseDB/pulsedb0/"
#path_main = "C:/Biomedizinische Informationstechnik/2. Semester/Projektarbeit/Code/Data/"
path_main = "C:/Biomedizinische Informationstechnik/3. Semester/Master-Studienarbeit/Data/"
# IDs
files = os.listdir(path_main)

batch_size = 64

indexes = []
for c in range(0, len(files)):
    indexes.append(c)
indexes = np.array(indexes)


print("Train TemplateNet with PulseDB")
all_mae_sbp, all_mae_dbp, subject_result, all_pred, all_r_sbp, all_r_dbp = [
], [], [], [], [], []

# Separate training, validation and test ids
train_index, test_index = train_test_split(
    indexes, test_size=0.1, random_state=42)
train_index, val_index = train_test_split(
    train_index, test_size=0.11111, random_state=42)
train_id = [files[x] for x in train_index]
val_id = [files[x] for x in val_index]
test_id = [files[x] for x in test_index]

#train_id = train_id[0:5]
#val_id = val_id[0:1]

# Generators
print("Loading Datagenerator")
#generator_train = DataGenerator_Pressure(path_main, train_id, batch_size=batch_size, shuffle=False)
#generator_val = DataGenerator_Pressure(path_main, val_id, batch_size=batch_size, shuffle=False)
#generator_test = DataGenerator_Pressure(path_main, test_id, batch_size=batch_size, shuffle=False)
generator_train = DataGenerator(
    path_main, train_id, batch_size=batch_size, typ='ABP_multi', shuffle=False)
generator_val = DataGenerator(
    path_main, val_id, batch_size=batch_size, typ='ABP_multi', shuffle=False)
generator_test = DataGenerator(
    path_main, test_id, batch_size=batch_size, typ='ABP_multi', shuffle=False)

# Load model with weights
#model_abp = keras.models.load_model('C:/Biomedizinische Informationstechnik/3. Semester/Master-Studienarbeit/Code/best_model_template_pressure.h5', compile=False)

# %% Start Training
model_abp = make_model_pressure()


# Training Code
# Make training
optimizer = optimizers.Adam(learning_rate=0.0001)

es = EarlyStopping(monitor="mae", patience=10)
mcp = ModelCheckpoint('best_model_template_pressure_multivariate-TS'+'.h5',
                      monitor='val_mae', save_best_only=True)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', factor=0.1, patience=5, min_lr=1e-8)
model_abp.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

model_abp.fit(generator_train,
              validation_data=generator_val,
              epochs=1,
              verbose=1,
              callbacks=[es, mcp, reduce_lr])

# Make prediction
nr_data = generator_test.__len__()
all_mae = np.zeros((nr_data, 2))

print('Prediction')
for batch_index in range(0, nr_data):
    # for batch_index in range(0,1):
    batch_data, temp_true = generator_test.__getitem__(batch_index)
    if batch_index == 1:
        batch_data1 = batch_data
        temp_true1 = temp_true
    if batch_index == 2:
        batch_data2 = batch_data
        temp_true2 = temp_true

    print(f'batch_index: {batch_index}')

    temp_pred = model_abp.predict(batch_data, verbose=0, batch_size=batch_size)
    if batch_index == 0:
        data_pred = temp_pred
        data_true = temp_true
        # continue        #reinschreiben für korrelationskoeffizient --> bessere Berechnung

    if batch_index == nr_data-1:
        for i in range(len(temp_pred)):
            if temp_pred[i, 0] == 0:
                temp_pred = temp_pred[i-1]
                temp_true = temp_true[i-1]

    data_pred = np.concatenate((data_pred, temp_pred), axis=0)
    data_true = np.concatenate((data_true, temp_true), axis=0)

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


# %%Train neural network without k-fold cross validation - univariate time series
# Main path of final preprocessed data
#path_main = "PulseDB/pulsedb0/"
path_main = "C:/Biomedizinische Informationstechnik/3. Semester/Master-Studienarbeit/Data/"
# IDs
files = os.listdir(path_main)

batch_size = 64

indexes = []
for c in range(0, len(files)):
    indexes.append(c)
indexes = np.array(indexes)


print("Train TemplateNet with PulseDB")
all_mae_sbp, all_mae_dbp, subject_result, all_pred, all_r_sbp, all_r_dbp = [
], [], [], [], [], []

# Separate training, validation and test ids
train_index, test_index = train_test_split(
    indexes, test_size=0.1, random_state=42)
train_index, val_index = train_test_split(
    train_index, test_size=0.11111, random_state=42)
train_id = [files[x] for x in train_index]
val_id = [files[x] for x in val_index]
test_id = [files[x] for x in test_index]

#train_id = train_id[0:5]
#val_id = val_id[0:1]

# Generators
print("Loading Datagenerator")
generator_train = DataGenerator(
    path_main, train_id, batch_size=batch_size, typ='ABP_single', shuffle=False)
generator_val = DataGenerator(
    path_main, val_id, batch_size=batch_size, typ='ABP_single', shuffle=False)
generator_test = DataGenerator(
    path_main, test_id, batch_size=batch_size, typ='ABP_single', shuffle=False)

#batch_data, temp_true = generator_test.__getitem__(0)
# Load model with weights
#model_abp = keras.models.load_model('C:/Biomedizinische Informationstechnik/3. Semester/Master-Studienarbeit/Code/best_model_template_pressure.h5', compile=False)

# %% Start Training
#model_abp = make_model_pressure_onesignal()

# Training Code
# Make training
optimizer = optimizers.Adam(learning_rate=0.0001)

es = EarlyStopping(monitor="mae", patience=10)
mcp = ModelCheckpoint('best_model_template_pressure_univariate-TS_session2' +
                      '.h5', monitor='val_mae', save_best_only=True)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', factor=0.1, patience=5, min_lr=1e-8)
model_abp.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

model_abp.fit(generator_train,  # generator_train
              validation_data=generator_val,
              epochs=4,
              verbose=1,
              callbacks=[es, mcp, reduce_lr])

# Make prediction
nr_data = generator_test.__len__()  # nr_data equals number of batches
all_mae = np.zeros((nr_data, 2))

print('Prediction')
for batch_index in range(0, nr_data):
    # for batch_index in range(0,1):
    batch_data, temp_true = generator_test.__getitem__(batch_index)
    #print(f'batch_index: {batch_index}')

    temp_pred = model_abp.predict(batch_data, verbose=0, batch_size=batch_size)
    if batch_index == 0:
        data_pred = temp_pred
        data_true = temp_true
        # continue        #reinschreiben für korrelationskoeffizient --> bessere Berechnung

    if batch_index == nr_data-1:
        for i in range(len(temp_pred)):
            if temp_pred[i, 0] == 0:
                temp_pred = temp_pred[i-1]
                temp_true = temp_true[i-1]

    data_pred = np.concatenate((data_pred, temp_pred), axis=0)
    data_true = np.concatenate((data_true, temp_true), axis=0)

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


# %% Make distribution of data
nr_data = generator_test.__len__()  # nr_data equals number of batches

generator_test = DataGenerator(path_main, test_id, batch_size=batch_size, shuffle=False)
batch_data0, temp_true = generator_test.__getitem__(1)
batch_data1, temp_true = generator_test.__getitem__(1)
batch_data2, temp_true = generator_test.__getitem__(2)

p1v = np.load(path_main+"dev0_abp/p002031v.npy", allow_pickle=True)
p1v_dev0 = p1v[0]

p1v_batches = []

total_nr_segments = 0
for i in range(len(test_id)):
    dev0_abp = np.load(path_main+"dev0_abp/"+test_id[i], allow_pickle=True)
    nr_segments = len(dev0_abp)
    total_nr_segments = total_nr_segments + nr_segments


nr_batches = int(np.floor(total_nr_segments/batch_size))


# %%
print('Prediction')
for batch_index in range(0, nr_data):
    # print(batch_index)
    batch_data, temp_true = generator_test.__getitem__(batch_index)

    if batch_index == 0:
        data_true = temp_true

    data_true = np.concatenate((data_true, temp_true), axis=0)

data_true_rounded = np.round(data_true, decimals=0)

n_bins_true = len(np.unique(data_true_rounded))

plt.figure()
plt.hist(data_true_rounded, n_bins_true*2)
plt.xlim([20, 200])
plt.ylim([0, 5000])
plt.title('Histogram of the Ground Truth data - whole dataset')
plt.xlabel('Blood Pressure')
plt.ylabel('Counts')
plt.grid()


data_true_SBP = data_true_rounded[:, 0]
data_true_DBP = data_true_rounded[:, 1]

values_SBP, indices_SBP, counts_SBP = np.unique(
    data_true_SBP, return_index=True, return_counts=True, axis=0)
values_DBP, indices_DBP, counts_DBP = np.unique(
    data_true_DBP, return_index=True, return_counts=True, axis=0)


# Make distribution of data with StratifiedKFold

indexes = []
for c in range(0, len(data_true)):
    indexes.append(c)
indexes = np.array(indexes)


skf = StratifiedKFold(n_splits=10, shuffle=False)
skf.get_n_splits(indexes, data_true_SBP)

#train_index_quant, test_index_quant = skf.split(indexes, data_true_rounded[:,0])
for i, (train_index, test_index) in enumerate(skf.split(indexes, data_true_SBP)):
    print(f"Fold {i}:")
    print(f"  Train: index={train_index}")
    print(f"  Test:  index={test_index}")
    if i == 0:
        quant_indexes = test_index

data_true_rounded_SBP = data_true_rounded[quant_indexes, 0]
data_true_rounded_DBP = data_true_rounded[quant_indexes, 1]

n_bins_true = len(np.unique(data_true_rounded_SBP))

plt.figure()
plt.hist(data_true_rounded_SBP, n_bins_true*3)
plt.hist(data_true_rounded_DBP, n_bins_true*3)
plt.xlim([20, 200])
plt.ylim([0, 500])
plt.title('Histogram of the Ground Truth data - Sample')
plt.xlabel('Blood Pressure')
plt.ylabel('Counts')
plt.grid()


def get_DataGen_index(index, batch_size):
    quotient = index/batch_size
    batch_index = math.floor(quotient)
    index_nr = index - (batch_index * batch_size)

    return batch_index, index_nr


batch_index, index_nr = get_DataGen_index(11906, 64)
print(f'batch_index: {batch_index}, index_nr: {index_nr}')

quantitative_dataset = []

'''
#for i in range(0,len(quant_indexes)):
for i in range(0,5):
    index = quant_indexes[i]
    batch_index, index_nr = get_DataGen_index(index, batch_size)
    batch_data, temp_true = generator_test.__getitem__(batch_index)
    liste = []
    ABP0 = batch_data[0][index_nr][:]
    liste.append(ABP0)
    ABP1 = batch_data[1][index_nr][:]
    liste.append(ABP1)
    ABP2 = batch_data[2][index_nr][:]
    liste.append(ABP2)
    quantitative_dataset.append(liste)
 '''

# %% Integrated Gradients Algorithm Implementation


def make_input_tensors(batch, batch_size, n_signals):
    ''' 
    Computes the input signals as tf tensors for a batch of data

    Args:
        - batch: batch of data from the Data Generator
        - batch_size: --> predetermined in script

    Returns a batch of data as tf tensors
    '''
    # Define batch size and number of input signals for neural network as variables
    n_input_signals = n_signals
    l = batch_size

    # Make array for all instances
    all_instances = []

    # Iterate over all instances in the batch
    for i in range(0, l):
        instance = []
        # Iterate over all 6 input signals for an instance
        for j in range(0, n_input_signals):
            one_signal = batch[j][i]
            one_signal = np.expand_dims(one_signal, axis=0)
            instance.append(tf.cast(one_signal, tf.float32))
        all_instances.append(instance)

    return all_instances


def make_all_instances(batch, batch_size, n_signals):
    ''' 
    Computes the input signals as np.array for a batch of data

    Args:
        - batch: batch of data from the Data Generator
        - batch_size: --> predetermined in script

    Returns a batch of data as list of numpy arrays
    '''
    # Define batch size and number of input signals for neural network as variables
    n_input_signals = n_signals
    l = batch_size

    # Make array for all instances
    all_instances = []

    # Iterate over all instances in the batch
    for i in range(0, l):
        instance = []
        # Iterate over all 6 input signals for an instance
        for j in range(0, n_input_signals):
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

    # Array for all gradients
    grads = []

    # Get gradients with Tensorflow GradientTape class
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(instance)
        preds = model_abp(instance)
        SBP = preds[:, 0]
        DBP = preds[:, 1]

    grads_SBP = tape.gradient(SBP, instance)
    grads_DBP = tape.gradient(DBP, instance)

    return grads_SBP, grads_DBP

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
    # print(f'inputsignals={n_input_signals}')
    #n_input_signals = 6

    if baseline == None:
        baseline = np.zeros((1, 1000))
    elif baseline == 'Random_Signal':
        np.random.seed(0)
        baseline = np.random.default_rng().uniform(-1, 1, (1, 1000))
    elif baseline == 'Noise_Signal':
        noise = np.random.normal(0, 1, 1000)
        noise = np.expand_dims(noise, axis=0)
        baseline = segment + noise

    # Step 1: Do Interpolation
    interpolated_signals = []
    for step in range(num_steps+1):
        interpolated_signal = baseline + \
            (step/num_steps) * (segment - baseline)
        interpolated_signals.append(interpolated_signal)

    # Step 2: Get gradients from interpolated signals
    grads_SBP = []
    grads_DBP = []
    for i, signals in enumerate(interpolated_signals):
        print(f"Interpolated Signal {i}")
        # Make input tensor
        interpolated_signals_tensors = [
            tf.cast(signals[j], tf.float32) for j in range(0, n_input_signals)]
        # Get Gradients for one instance
        grad_SBP, grad_DBP = get_gradients(interpolated_signals_tensors)
        # Save gradients in list
        grads_SBP.append(grad_SBP)
        grads_DBP.append(grad_DBP)
        # print(f"i={i}, und signal={signal[0]")
        #interpolated_signals_tensors = []

    # Step 3: Approximate the integral
    # Make Numpy Arrays
    grads_SBP_numpy = np.array(grads_SBP)
    grads_DBP_numpy = np.array(grads_DBP)

    # Approximate the integral using the Riemann Sum/ Trapezoidal Rule
    # Riemann Sum
    sum_SBP = np.sum(grads_SBP_numpy, axis=0)
    sum_DBP = np.sum(grads_DBP_numpy, axis=0)
    # Trapezoidal Rule
    #grads_SBP_TR = grads_SBP_numpy[:-1] + grads_SBP_numpy[1:] / 2.0
    #grads_DBP_TR = grads_DBP_numpy[:-1] + grads_DBP_numpy[1:] / 2.0

    # Calculate Average Grads
    # Riemann Sum
    avg_grads_SBP = sum_SBP * (1/(num_steps))
    avg_grads_DBP = sum_DBP * (1/(num_steps))

    # Trapezoidal Rule
    #avg_grads_SBP = np.mean(grads_SBP_TR, axis=0)
    #avg_grads_DBP = np.mean(grads_DBP_TR, axis=0)

    # Step 4: Calculate integrated gradients and return
    integrated_grads_SBP = (segment - baseline) * avg_grads_SBP
    integrated_grads_DBP = (segment - baseline) * avg_grads_DBP

    return integrated_grads_SBP, integrated_grads_DBP


def sum_IG(IG):
    IG_summed = np.sum(IG, axis=0)

    return IG_summed


def normalize_IG(IG, method):
    IG_matrix = np.squeeze(IG.copy())
    matrix_shape = np.shape(IG_matrix)
    IG_vector = np.reshape(IG_matrix, (1, matrix_shape[0]*matrix_shape[1]))
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
        IG = np.zeros((6, 1000))
        for i in range(0, matrix_shape[0]):
            IG[i, :] = (IG_matrix[i, :]-all_mean[i])/all_std[i]

    return IG


# %% Calculate Integrated Gradients for certain number of batches
# Define number of batches to calculate IG for
nr_batches = 1
#all_IG_pressure_SBP = []
#all_IG_pressure_DBP = []

for batch_index in range(0, nr_batches):
    batch_data, temp_true = generator_test.__getitem__(batch_index)
    all_instances = make_input_tensors(batch_data, batch_size, n_signals=1)

    all_input_signals = []
    for i in range(0, 1):
        all_input_signals.append(batch_data[i])

    # for i in range(0,len(all_instances)):
    for i in range(2, 10):
        print(f"Batch: {batch_index} - Example No:{i+1} - Zero Baseline")
        #grads_SBP, grads_DBP = get_gradients(all_instances[i])
        IG_pressure_SBP, IG_pressure_DBP = get_integrated_gradients(
            all_instances[i], baseline=None, num_steps=50)
        all_IG_pressure_SBP.append(IG_pressure_SBP)
        all_IG_pressure_DBP.append(IG_pressure_DBP)


#np.save(target_path+'IG/IG_0_1600_SBP', all_IG_pressure_SBP)
#np.save(target_path+'IG/IG_0_1600_DBP', all_IG_pressure_DBP)

# %% Visualize Results
subject_nr = 4
#IG_normalized = normalize_IG(all_IG_pressure_SBP[subject_nr], method='zero_mean')

#vs.subplot_3_signals_bwr_heatmap(all_IG_pressure_SBP[subject_nr], all_input_signals, subject_nr, colorbar='single', mode='PPG')
#vs.subplot_3_signals_bwr_heatmap(IG_normalized, all_input_signals, subject_nr, colorbar='single', mode='PPG')

vs.plot_onesignal_bwr_heatmap(all_IG_pressure_SBP[subject_nr], ABP[0][subject_nr], signal_nr=0)
