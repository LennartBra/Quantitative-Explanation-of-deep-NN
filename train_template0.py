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

from DataGenerator_template import DataGenerator
from make_model_template import make_model

import tensorflow as tf
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split, LeaveOneOut, KFold
from keras import optimizers
from keras.models import clone_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow import keras
import matplotlib.pyplot as plt

keras.backend.clear_session()

if tf.test.gpu_device_name():
    print("Default GPU Device: {}".format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")

target_path = "C:/Biomedizinische Informationstechnik/2. Semester/Projektarbeit/Code/Data/"
#Make Matlab like plots
#%matplotlib qt

#%% Main code
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
        train_index, val_index = train_test_split(train_index, test_size=0.1)
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
        # Make prediction
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
    n_input_signals = 6
    
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

#Make Input Data Tensors --> correct format for Integrated Gradients Algorithm
#all_instances = make_all_instances(batch_data, batch_size)
all_instances = make_input_tensors(batch_data, batch_size)
#Calculate one example
grads_SBP, grads_DBP = get_gradients(all_instances[1])
IG_SBP, IG_DBP = get_integrated_gradients(all_instances[1], baseline=None, num_steps=25)


#%% Calculate 100 examples with zero baseline

IG_SBP_examples_zero = []
IG_DBP_examples_zero = []
for i in range(0,3):
    print(f"Example No.{i+1} - Zero Baseline")
    #grads_SBP, grads_DBP = get_gradients(all_instances[i])
    IG_SBP, IG_DBP = get_integrated_gradients(all_instances[i], baseline=None, num_steps=50)
    IG_SBP_examples_zero.append(IG_SBP)
    IG_DBP_examples_zero.append(IG_DBP)

#np.save(target_path+'IG/IG_zero_SBP', IG_SBP_examples_zero)
#np.save(target_path+'IG/IG_zero_DBP', IG_DBP_examples_zero)

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
#50 Interpolation steps - Zero Baseline - 100 segments
IG_zero_SBP = np.load(target_path+'IG/IG_zero_SBP.npy')
IG_zero_DBP = np.load(target_path+'IG/IG_zero_DBP.npy')
#50 Interpolation steps - Random Baseline - 100 segments
IG_random_SBP = np.load(target_path+'IG/IG_random_SBP.npy')
IG_random_DBP = np.load(target_path+'IG/IG_random_DBP.npy')
#25 Interpolation Steps - Zero Baseline - 100 segments
IG_zero_SBP_25steps = np.load(target_path+'IG/IG_zero_SBP_new.npy')
IG_zero_DBP_25steps = np.load(target_path+'IG/IG_zero_DBP_new.npy')


#%% Visualizing Results
def subplot_all_input_signals(batch_data, segment_nr):
    '''
    Subplots all input signals of a segment from a given batch over the samples
    
    Args:
        - batch_data = One batch of 2048 segments
        - segment_nr = One segment
       
    '''
    def get_instance(batch_data, instance_nr):
        signal_nr = 6
        instance = []
    
        for i in range(0,signal_nr):
            signal =  batch_data[i][instance_nr][:]
            instance.append(signal)
        
        return instance
    
    instance = get_instance(batch_data,segment_nr)
    
    fig, axs = plt.subplots(6, sharex=True)
    fig.suptitle('All Input Signals')
    fig.supxlabel('Samples')
    fig.supylabel('Amplitude')
    axs[0].plot(range(0,1000),instance[0])
    axs[1].plot(range(0,1000),instance[1])
    axs[2].plot(range(0,1000),instance[2])
    axs[3].plot(range(0,1000),instance[3])
    axs[4].plot(range(0,1000),instance[4])
    axs[5].plot(range(0,1000),instance[5])  
    plt.setp(axs, xlim=(0,1000))
    
def subplot_all_IG_attributions(IG):
    '''
    Subplot all six IG signals of a segment over the samples
    
    Args:
        - IG: IG attributions of one segment for all six input signals
       
    '''
    fig, axs = plt.subplots(6, sharex=True)
    fig.suptitle('Attributions for all Input signals')
    fig.supxlabel('Samples')
    fig.supylabel('Attributions')
    axs[0].plot(range(0,1000), np.squeeze(IG[0]))
    axs[0].grid()
    axs[1].plot(range(0,1000), np.squeeze(IG[1]))
    axs[1].grid()
    axs[2].plot(range(0,1000), np.squeeze(IG[2]))
    axs[2].grid()
    axs[3].plot(range(0,1000), np.squeeze(IG[3]))
    axs[3].grid()
    axs[4].plot(range(0,1000), np.squeeze(IG[4]))
    axs[4].grid()
    axs[5].plot(range(0,1000), np.squeeze(IG[5]))
    axs[5].grid()

    
def plot_signal_heatmap(signal, IG, signal_nr):
    '''
    Plot one time signal with the corresponding IG attributions as a heatmap over the samples
    
    Args:
        - signal: one time signal of the segment (PPG, PPG1, PPG2, TemplatePPG, TemplatePPG1,
                                                  TemplatePPG2)
        - IG: IG attributions of one segment
        - signal_nr: specifies the desired signal number for the IG attributions; must be the same
                     as signal (0 = PPG, 1 = PPG1, 2 = PPG2, 3 = TemplatePPG, 4 = TemplatePPG2,
                                5 = TemplatePPG2)
       
    '''
    IG = IG[signal_nr][0][0:1000].reshape((1,1000))
    t = range(0,1000)
    plt.figure()
    s = plt.scatter(t, signal, c=IG, cmap='jet')#, ec='k')
    if signal_nr == 0:
        plt.title('PPG0 Signal + Attributions as heatmap and signal')
        #plt.plot(t,np.squeeze(IG), linewidth=1, color='k')
    elif signal_nr == 1:
        plt.title('PPG1 Signal + Attributions as heatmap')
        #plt.plot(t,np.squeeze(IG), linewidth=0.8, color='k')
    elif signal_nr == 2:
        plt.title('PPG2 Signal + Attributions as heatmap')
    elif signal_nr == 3:
        plt.title('Template Signal 1 + Attributions as heatmap and signal')
    elif signal_nr == 4:
        plt.title('Template Signal 2 + Attributions as heatmap and signal')
    elif signal_nr == 5:
        plt.title('Template Signal 3 + Attributions as heatmap and signal')
    plt.plot(t,signal, linewidth=0.8, color='k')
    plt.xlim([0,1000])
    plt.xlabel('Samples')
    plt.ylabel('Amplitude/Attributions')
    plt.colorbar(s)
    plt.grid()
    plt.show()
    
def plot_signal_with_IG(signal,IG, signal_nr):
    '''
    Plots a signal and the IG attributions over the samples in the same plot
    
    Args:
        - IG: IG attributions of one segment for all six input signals
        - signal: one time signal
        - signal_nr: specifies the desired signal number for the IG attributions; must be the same
                     as signal (0 = PPG, 1 = PPG1, 2 = PPG2, 3 = TemplatePPG, 4 = TemplatePPG2,
                                5 = TemplatePPG2)
       
    '''
    IG = IG[signal_nr]
    plt.figure()
    plt.xlabel('Samples')
    plt.ylabel('Amplitude/Attributions')
    if signal_nr == 0:
        plt.title('PPG0 Signal + Attributions')
    elif signal_nr == 1:
        plt.title('PPG1 Signal + Attributions')
    elif signal_nr == 2:
        plt.title('PPG2 Signal + Attributions')
    elif signal_nr == 3:
        plt.title('Template Signal 1 + Attributions')
    elif signal_nr == 4:
        plt.title('Template Signal 2 + Attributions')
    elif signal_nr == 5:
        plt.title('Template Signal 3 + Attributions')
    plt.plot(range(0,1000),signal)
    plt.plot(range(0,1000),np.squeeze(IG))
    plt.grid()
    plt.show()
    
def plot_PPG_heatmap_scatter_subplot(PPG, PPG1, PPG2, IG):
    '''
    Subplot PPG, PPG1 and PPG2 with the corresponding IG attributions as a heatmap over the samples
    
    Args:
        - PPG: PPG time signal
        - PPG1: first derivative of PPG
        - PPG2: second derivative of PPG
        - IG: IG attributions of one segment for all six input signals
       
    '''
    IG0 = IG[0][0][0:1000].reshape((1,1000))
    IG1 = IG[1][0][0:1000].reshape((1,1000))
    IG2 = IG[2][0][0:1000].reshape((1,1000))
    t = range(0,1000)
    
    fig, axs = plt.subplots(3, sharex=True)
    fig.suptitle('PPG, PPG1 and PPG2 + Attributions as heatmap')
    fig.supxlabel('Samples')
    fig.supylabel('Amplitude/Attributions')
    s0 = axs[0].scatter(t, PPG, c=IG0, cmap='jet')#, ec='k')
    axs[0].plot(range(0,1000),PPG, linewidth=0.5, color='k')
    axs[0].grid()
    s1 = axs[1].scatter(t, PPG1, c=IG1, cmap='jet')#, ec='k')
    axs[1].plot(range(0,1000),PPG1, linewidth=0.5, color='k')
    axs[1].grid()
    s2 = axs[2].scatter(t, PPG2, c=IG2, cmap='jet')#, ec='k')
    axs[2].plot(range(0,1000),PPG2, linewidth=0.5, color='k')
    axs[2].grid()
    #Use multiple colorbars
    fig.colorbar(s0)
    fig.colorbar(s1)
    fig.colorbar(s2)
    plt.setp(axs, xlim=(0,1000))
    #Use One Colorbar
    #fig.subplots_adjust(right=0.8)
    #cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    #fig.colorbar(s0, cax=cbar_ax)

def plot_templates_heatmap_scatter_subplot(TPPG, TPPG1, TPPG2, IG):
    '''
    Subplot Template PPG, Template PPG1 and Template PPG2 with the corresponding IG attributions
    as a heatmap over the samples
    
    Args:
        - TPPG: Template signal for PPG
        - PPG1: Template signal for first derivative of PPG
        - PPG2: Template signal for second derivative of PPG
        - IG: IG attributions of one segment for all six input signals
       
    '''
    IG0 = IG[3][0][0:1000].reshape((1,1000))
    IG1 = IG[4][0][0:1000].reshape((1,1000))
    IG2 = IG[5][0][0:1000].reshape((1,1000))
    t = range(0,1000)
    
    fig, axs = plt.subplots(3, sharex=True)
    fig.suptitle('Template signals + Attributions as heatmap')
    fig.supxlabel('Samples')
    fig.supylabel('Amplitude/Attributions')
    s0 = axs[0].scatter(t, TPPG, c=IG0, cmap='jet')
    axs[0].plot(range(0,1000),TPPG, linewidth=0.5, color='k')
    axs[0].grid()
    s1 = axs[1].scatter(t, TPPG1, c=IG1, cmap='jet')
    axs[1].plot(range(0,1000),TPPG1, linewidth=0.5, color='k')
    axs[1].grid()
    s2 = axs[2].scatter(t, TPPG2, c=IG2, cmap='jet')
    axs[2].plot(range(0,1000),TPPG2, linewidth=0.5, color='k')
    axs[2].grid()
    #Use multiple colorbars --> one for each subplot 
    fig.colorbar(s0)
    fig.colorbar(s1)
    fig.colorbar(s2)
    plt.setp(axs, xlim=(0,1000))
    #Use One Colorbar --> one for all subplots
    #fig.subplots_adjust(right=0.8)
    #cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    #fig.colorbar(s0, cax=cbar_ax)
    
def subplot_heatmap_and_IG(PPG, IG, signal_nr):
    '''
    Subplot one time signal and the corresponding IG attributions over the samples
    as a heatmap
    
    Args:
        - TPPG: Template signal for PPG
        - PPG1: Template signal for first derivative of PPG
        - PPG2: Template signal for second derivative of PPG
        - IG: IG attributions of one segment for all six input signals
       
    '''
    
    IG0 = IG[signal_nr][0][0:1000].reshape((1,1000))
    t = range(0,1000)
    
    fig, axs = plt.subplots(2, sharex=True)
    if signal_nr == 0:
        fig.suptitle('PPG0 Signal + Attributions as heatmap and signal')
    elif signal_nr == 1:
        fig.suptitle('PPG1 Signal + Attributions as heatmap and signal')
    elif signal_nr == 2:
        fig.suptitle('PPG2 Signal + Attributions as heatmap and signal')
    elif signal_nr == 3:
        fig.suptitle('Template Signal 1 + Attributions as heatmap and signal')
    elif signal_nr == 4:
        fig.suptitle('Template Signal 2 + Attributions as heatmap and signal')
    elif signal_nr == 5:
        fig.suptitle('Template Signal 3 + Attributions as heatmap and signal')
    fig.supxlabel('Samples')
    fig.supylabel('Amplitude/Attributions')
    s0 = axs[0].scatter(t, PPG, c=IG0, cmap='jet', ec='k')
    axs[0].plot(range(0,1000),PPG, linewidth=0.5)
    axs[1].plot(range(0,1000),np.squeeze(IG0))
    
    fig.colorbar(s0)

    
def subplot_many_templates_in_one(IG):
    '''
    Subplot IG attributions of 20 segments for TemplatePPG, TemplatePPG1 and TemplatePPG2 over
    the samples
    
    Args:
        - IG: IG attributions of 20 segments for all six input signals
       
    '''
    t = range(0,1000)
    fig, axs = plt.subplots(3)
    fig.suptitle('All Templatesignal Attributions')
    fig.supxlabel('Samples')
    fig.supylabel('Attributions')
    for i in range(0,20):
        axs[0].plot(t, np.squeeze(IG[i][3]))
        axs[1].plot(t, np.squeeze(IG[i][4]))
        axs[2].plot(t, np.squeeze(IG[i][5]))
    axs[0].grid()
    axs[1].grid()
    axs[2].grid()
    
def plot_IG_for_interpolation_steps(IG25, IG75, IG175):
    '''
    Subplot 3 different IG attributions over the samples for all six time signals
    
    Args:
        - IG25: IG attributions for one segment calculated with 25 interpolation steps
        - IG75: IG attributions for one segment calculated with 75 interpolation steps
        - IG175: IG attributions for one segment calculated with 175 interpolation steps
       
    '''
    fig, axs = plt.subplots(6, sharex=True)
    fig.suptitle('All Signals + Attributions with different step size')
    fig.supxlabel('Samples')
    fig.supylabel('Attributions')
    axs[0].plot(range(0,1000), np.squeeze(IG25[0]), label='25')
    axs[0].plot(range(0,1000), np.squeeze(IG75[0]), label='75')
    axs[0].plot(range(0,1000), np.squeeze(IG175[0]), label='175')
    axs[0].legend()
    axs[0].grid()
    axs[1].plot(range(0,1000), np.squeeze(IG25[1]))
    axs[1].plot(range(0,1000), np.squeeze(IG75[1]))
    axs[1].plot(range(0,1000), np.squeeze(IG175[1]))
    axs[1].grid()
    axs[2].plot(range(0,1000), np.squeeze(IG25[2]))
    axs[2].plot(range(0,1000), np.squeeze(IG75[2]))
    axs[2].plot(range(0,1000), np.squeeze(IG175[2]))
    axs[2].grid()
    axs[3].plot(range(0,1000), np.squeeze(IG25[3]))
    axs[3].plot(range(0,1000), np.squeeze(IG75[3]))
    axs[3].plot(range(0,1000), np.squeeze(IG175[3]))
    axs[3].grid()
    axs[4].plot(range(0,1000), np.squeeze(IG25[4]))
    axs[4].plot(range(0,1000), np.squeeze(IG75[4]))
    axs[4].plot(range(0,1000), np.squeeze(IG175[4]))
    axs[4].grid()
    axs[5].plot(range(0,1000), np.squeeze(IG25[5]))
    axs[5].plot(range(0,1000), np.squeeze(IG75[5]))
    axs[5].plot(range(0,1000), np.squeeze(IG175[5]))
    axs[5].grid()
        
def plot_3PPG_heatmap_scatter_subplot(PPG, PPG1, PPG2, IG, IG1, IG2, signal):
    '''
    Subplot PPG, PPG1 and PPG2 with the corresponding IG attributions as a heatmap over the samples
    
    Args:
        - PPG: PPG time signal 1
        - PPG1: PPG time signal 2
        - PPG2: PPG time signal 3
        
        - IG: IG attributions 1 of one segment for all six input signals
        - IG: IG attributions 2 of one segment for all six input signals
        - IG: IG attributions 3 of one segment for all six input signals
       
    '''
    
    if signal == 1:
        signal_nr = 0
    elif signal == 2:
        signal_nr = 1
    elif signal == 3:
        signal_nr = 2
    elif signal == 4:
        signal_nr = 3
    elif signal == 5:
        signal_nr = 4
    elif signal == 6:
        signal_nr = 5
        
    IG_0 = IG[signal_nr][0][0:1000].reshape((1,1000))
    IG_1 = IG1[signal_nr][0][0:1000].reshape((1,1000))
    IG_2 = IG2[signal_nr][0][0:1000].reshape((1,1000))
    t = range(0,1000)
    
    fig, axs = plt.subplots(3, sharex=True)
    if signal_nr == 0:
        fig.suptitle('PPG from 3 time segments + Attributions as heatmap')
    elif signal_nr == 1:
        fig.suptitle('PPG1 from 3 time segments + Attributions as heatmap')
    elif signal_nr == 2:
        fig.suptitle('PPG2 from 3 time segments + Attributions as heatmap')
    elif signal_nr == 3:
        fig.suptitle('Template PPG from 3 time segments + Attributions as heatmap')
    elif signal_nr == 4:
        fig.suptitle('Template PPG1 from 3 time segments + Attributions as heatmap')
    elif signal_nr == 5:
        fig.suptitle('Template PPG2 from 3 time segments + Attributions as heatmap')
    fig.supxlabel('Samples')
    fig.supylabel('Amplitude/Attributions')
    s0 = axs[0].scatter(t, PPG, c=IG_0, cmap='jet')#, ec='k')
    axs[0].plot(range(0,1000),PPG, linewidth=0.5, color='k')
    axs[0].grid()
    s1 = axs[1].scatter(t, PPG1, c=IG_1, cmap='jet')#, ec='k')
    axs[1].plot(range(0,1000),PPG1, linewidth=0.5, color='k')
    axs[1].grid()
    s2 = axs[2].scatter(t, PPG2, c=IG_2, cmap='jet')#, ec='k')
    axs[2].plot(range(0,1000),PPG2, linewidth=0.5, color='k')
    axs[2].grid()
    #Use multiple colorbars
    fig.colorbar(s0)
    fig.colorbar(s1)
    fig.colorbar(s2)
    plt.setp(axs, xlim=(0,1000))
    #Use One Colorbar
    #fig.subplots_adjust(right=0.8)
    #cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    #fig.colorbar(s0, cax=cbar_ax)
#%% Subplot all input signals
subplot_all_input_signals(batch_data, 22)

#%% Visualize some zero baseline examples for SBP 
#Beispielsegment 1 SBP PPG0 --> Instanz 80
#Beispielsegment 2 SBP PPG0 --> Instanz 81
#Beispielsegment 1 SBP PPG1 --> Instanz 80
#Beispielsegment 2 SBP PPG1 --> Instanz 81
#Beispielsegment 1 SBP PPG2 --> Instanz 80
#Beispielsegment 2 SBP PPG2 --> Instanz 81
#Define segment_nr and signal_nr and create plots
segment_nr = 80
signal_nr = 0

#Plot all IG attributions for one segment
subplot_all_IG_attributions(IG_zero_SBP[segment_nr])
#Plot Signal with IG in one plot
#plot_signal_with_IG(PPG[segment_nr], IG_zero_SBP[segment_nr], signal_nr)
#Plot Signal with attributions as heatmap
plot_signal_heatmap(PPG[segment_nr], IG_zero_SBP[segment_nr], signal_nr)
#Subplot Signal with attributions as heatmap and plot IG as signal
#subplot_heatmap_and_IG(PPG[segment_nr], IG_zero_SBP[segment_nr], signal_nr)
#Plot PPG0,PPG1,PPG2 with attributions as heatmap in one subplot
plot_PPG_heatmap_scatter_subplot(PPG[segment_nr], PPG1[segment_nr], PPG2[segment_nr], IG_zero_SBP[segment_nr])
#Plot all Attributions for the template signals
#plot_templates_heatmap_scatter_subplot(TemplatePPG[segment_nr], TemplatePPG1[segment_nr], TemplatePPG2[segment_nr], IG_zero_SBP[segment_nr])
#Plot many Templates in one
#subplot_many_templates_in_one(IG_zero_SBP)

#%% Visualize some zero baseline examples for DBP
#Beispielsegment 1 DBP PPG0 --> Instanz 44
#Beispielsegment 2 DBP PPG0 --> Instanz 54
#Beispielsegment 1 DBP PPG1 --> Instanz 44
#Beispielsegment 2 DBP PPG1 --> Instanz 54
#Beispielsegment 1 DBP PPG2 --> Instanz 44
#Beispielsegment 2 DBP PPG2 --> Instanz 54
#Define segment_nr and signal_nr and create plots
segment_nr = 44
signal_nr = 2

#Plot all IG attributions for one segment
subplot_all_IG_attributions(IG_zero_DBP[segment_nr])
#Plot Signal with IG in one plot
#plot_signal_with_IG(PPG2[segment_nr], IG_zero_DBP[segment_nr], signal_nr)
#Plot Signal with attributions as heatmap
plot_signal_heatmap(PPG2[segment_nr], IG_zero_DBP[segment_nr], signal_nr)
#Subplot Signal with attributions as heatmap and plot IG as signal
#subplot_heatmap_and_IG(PPG[segment_nr], IG_zero_DBP[segment_nr], signal_nr)
#Plot PPG0,PPG1,PPG2 with attributions as heatmap in one subplot
plot_PPG_heatmap_scatter_subplot(PPG[segment_nr], PPG1[segment_nr], PPG2[segment_nr], IG_zero_DBP[segment_nr])
#Plot all Attributions for the template signals
#plot_templates_heatmap_scatter_subplot(TemplatePPG[segment_nr], TemplatePPG1[segment_nr], TemplatePPG2[segment_nr], IG_zero_DBP[segment_nr])

#Plot many Templates in one
#subplot_many_templates_in_one(IG_zero_DBP)

#%% Einfluss der Interpolationsschritte
segment_nr = 90
#Calculate Attributions with different num_steps
IG_25steps_SBP, IG_25steps_DBP = get_integrated_gradients(all_instances[segment_nr], baseline=None, num_steps=25)
IG_75steps_SBP, IG_75steps_DBP = get_integrated_gradients(all_instances[segment_nr], baseline=None, num_steps=75)
IG_175steps_SBP, IG_175steps_DBP = get_integrated_gradients(all_instances[segment_nr], baseline=None, num_steps=175)

#Plot all IG attributions with different step sizes for SBP
plot_IG_for_interpolation_steps(IG_25steps_SBP, IG_75steps_SBP, IG_175steps_SBP)
#Plot all IG attributions with different step sizes for SBP
plot_IG_for_interpolation_steps(IG_25steps_DBP, IG_75steps_DBP, IG_175steps_DBP)


#%% Einfluss der Baseline
Instanz_E1 = 80
#Calculate Attributions with different num_steps
IG_zero_SBP_E1, IG_zero_DBP_E1 = get_integrated_gradients(all_instances[Instanz_E1], baseline=None, num_steps=50)
IG_random_SBP_E1, IG_random_DBP_E1 = get_integrated_gradients(all_instances[Instanz_E1], baseline='Random_Signal', num_steps=50)
Instanz_E2 = 54
IG_zero_SBP_E2, IG_zero_DBP_E2 = get_integrated_gradients(all_instances[Instanz_E2], baseline=None, num_steps=50)
IG_random_SBP_E2, IG_random_DBP_E2 = get_integrated_gradients(all_instances[Instanz_E2], baseline='Random_Signal', num_steps=50)


#Subplot PPG Attributions for PPG0,PPG1 and PPG2 for Uniform and Zero Baseline for SBP Example
plot_PPG_heatmap_scatter_subplot(PPG[Instanz_E1], PPG1[Instanz_E1], PPG2[Instanz_E1], IG_random_SBP_E1)
plot_PPG_heatmap_scatter_subplot(PPG[Instanz_E1], PPG1[Instanz_E1], PPG2[Instanz_E1], IG_zero_SBP[Instanz_E1])
#Plot Templatesignal attributions
plot_templates_heatmap_scatter_subplot(TemplatePPG[Instanz_E1], TemplatePPG1[Instanz_E1], TemplatePPG2[Instanz_E1],IG_random_SBP_E1,)
plot_templates_heatmap_scatter_subplot(TemplatePPG[Instanz_E1], TemplatePPG1[Instanz_E1], TemplatePPG2[Instanz_E1],IG_zero_SBP[Instanz_E1],)

#Subplot PPG Attributions for PPG0,PPG1 and PPG2 for Uniform and Zero Baseline for DBP Example
plot_PPG_heatmap_scatter_subplot(PPG[Instanz_E1], PPG1[Instanz_E1], PPG2[Instanz_E1], IG_random_SBP_E1)
plot_PPG_heatmap_scatter_subplot(PPG[Instanz_E1], PPG1[Instanz_E1], PPG2[Instanz_E1], IG_zero_SBP[Instanz_E1])
#Plot Templatesignal attributions
plot_templates_heatmap_scatter_subplot(TemplatePPG[Instanz_E1], TemplatePPG1[Instanz_E1], TemplatePPG2[Instanz_E1],IG_random_SBP_E1,)
plot_templates_heatmap_scatter_subplot(TemplatePPG[Instanz_E1], TemplatePPG1[Instanz_E1], TemplatePPG2[Instanz_E1],IG_zero_SBP[Instanz_E1],)


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


#%%Visualisierungen für die Powerpoint
#PPG
Instanz1 = 80
Instanz2= 44
Instanz3= 81
#Plot 3 PPG and their integrated gradients
plot_3PPG_heatmap_scatter_subplot(PPG[Instanz1], PPG[Instanz2], PPG[Instanz3], IG_zero_SBP[Instanz1], IG_zero_DBP[Instanz2], IG_zero_SBP[Instanz3],1)

#Erste Ableitung
plot_3PPG_heatmap_scatter_subplot(PPG1[Instanz1], PPG1[Instanz2], PPG1[Instanz3], IG_zero_SBP[Instanz1], IG_zero_DBP[Instanz2], IG_zero_SBP[Instanz3],2)

#Zweite Ableitung
plot_3PPG_heatmap_scatter_subplot(PPG2[Instanz1], PPG2[Instanz2], PPG2[Instanz3], IG_zero_SBP[Instanz1], IG_zero_DBP[Instanz2], IG_zero_SBP[Instanz3],3)

#PPG Template
plot_3PPG_heatmap_scatter_subplot(TemplatePPG[Instanz1], TemplatePPG[Instanz2], TemplatePPG[Instanz3], IG_zero_SBP[Instanz1], IG_zero_DBP[Instanz2], IG_zero_SBP[Instanz3],4)

#Erste Ableitung Template
plot_3PPG_heatmap_scatter_subplot(TemplatePPG1[Instanz1], TemplatePPG1[Instanz2], TemplatePPG1[Instanz3], IG_zero_SBP[Instanz1], IG_zero_DBP[Instanz2], IG_zero_SBP[Instanz3],5)

#Zweite Ableitung Template
plot_3PPG_heatmap_scatter_subplot(TemplatePPG2[Instanz1], TemplatePPG2[Instanz2], TemplatePPG2[Instanz3], IG_zero_SBP[Instanz1], IG_zero_DBP[Instanz2], IG_zero_SBP[Instanz3],6)

##########################################################Breite##########
###############################################################################























