# -*- coding: utf-8 -*-
"""
Master-Studienarbeit
Autor: Lennart Brakelmann
Thema: Quantitative Erklärbarkeit tiefer neuronaler Netze
Skript: XAI-Methode Integrated Gradients
"""
import tensorflow as tf
import numpy as np

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

def make_input_tensor(segment, n_signals):
    ''' 
    Computes the input signals as tf tensors for one segment

    Args:
        - segment: one segment of data from the Data Generator
        - n_signals: number of signals in one segment

    Returns one segment of data as tf tensors
    '''
    # Define batch size and number of input signals for neural network as variables
    n_input_signals = n_signals

    # Make array for all instances
    instance = []

    # Iterate over all input signals for an instance
    for j in range(0, n_input_signals):
        one_signal = segment[j]
        one_signal = np.expand_dims(one_signal, axis=0)
        instance.append(tf.cast(one_signal, tf.float32))

    return instance


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

def make_instance(batch, index, batch_size, n_signals):
    ''' 
    Computes the input signals as np.array for one segment

    Args:
        - batch: batch of data from the Data Generator
        - batch_size: --> predetermined in script
        - n_signals: number of signals in one segment

    Returns a instance of data as list of numpy arrays
    '''
    # Define batch size and number of input signals for neural network as variables
    n_input_signals = n_signals
    l = batch_size

    # Make array for all instances
    instance = []

    # Iterate over all 6 input signals for an instance
    for j in range(0, n_input_signals):
        one_signal = batch[j][index]
        one_signal = np.expand_dims(one_signal, axis=0)
        #instance.append(tf.cast(one_signal, tf.float32))
        instance.append(one_signal)

    return instance

def make_tf_tensor_from_Segment(Segment, typ):
    
    if typ == 'ABP_single':
        n_signals = 1 
        ABP = Segment[3]
        
        all_signals = np.array(ABP)
        
        # Make array for all instances
        instance = []

        # Iterate over all input signals for an instance
        for j in range(0, n_signals):
            #one_signal = all_signals[j]
            one_signal = np.expand_dims(all_signals, axis=0)
            instance.append(tf.cast(one_signal, tf.float32))
        
    elif typ == 'ABP_multi':
        n_signals = 3
        ABP0 = Segment[3]
        ABP1 = Segment[4]
        ABP2 = Segment[5]
        
        all_signals = np.array((ABP0, ABP1, ABP2))
        
        # Make array for all instances
        instance = []

        # Iterate over all input signals for an instance
        for j in range(0, n_signals):
            one_signal = all_signals[j]
            one_signal = np.expand_dims(one_signal, axis=0)
            instance.append(tf.cast(one_signal, tf.float32))
            
    elif typ == 'ABP_multi_900':
        n_signals = 3
        ABP0 = Segment[3][50:950]
        ABP1 = Segment[4][50:950]
        ABP2 = Segment[5][50:950]
        
        all_signals = np.array((ABP0, ABP1, ABP2))
        
        # Make array for all instances
        instance = []

        # Iterate over all input signals for an instance
        for j in range(0, n_signals):
            one_signal = all_signals[j]
            one_signal = np.expand_dims(one_signal, axis=0)
            instance.append(tf.cast(one_signal, tf.float32))
        
    
    elif typ == 'PPG':
        n_signals = 6
        PPG0 = Segment[0]
        PPG1 = Segment[1]
        PPG2 = Segment[2]
        temp0 = Segment[6]
        temp1 = Segment[7]
        temp2 = Segment[8]
        
        all_signals = np.array((PPG0, PPG1, PPG2, temp0, temp1, temp2))
        
        # Make array for all instances
        instance = []

        # Iterate over all input signals for an instance
        for j in range(0, n_signals):
            one_signal = all_signals[j]
            one_signal = np.expand_dims(one_signal, axis=0)
            instance.append(tf.cast(one_signal, tf.float32))
    
    return instance
    
def make_tf_tensor_from_quant_id(path_main, quant_id, typ):
    Segment = np.load(path_main+quant_id)
    
    if typ == 'ABP_single':
        n_signals = 1 
        ABP = Segment[3]
        
        all_signals = np.array(ABP)
        
        # Make array for all instances
        instance = []

        # Iterate over all input signals for an instance
        for j in range(0, n_signals):
            #one_signal = all_signals[j]
            one_signal = np.expand_dims(all_signals, axis=0)
            instance.append(tf.cast(one_signal, tf.float32))
        
    elif typ == 'ABP_multi':
        n_signals = 3
        ABP0 = Segment[3]
        ABP1 = Segment[4]
        ABP2 = Segment[5]
        
        all_signals = np.array((ABP0, ABP1, ABP2))
        
        # Make array for all instances
        instance = []

        # Iterate over all input signals for an instance
        for j in range(0, n_signals):
            one_signal = all_signals[j]
            one_signal = np.expand_dims(one_signal, axis=0)
            instance.append(tf.cast(one_signal, tf.float32))

    elif typ == 'ABP_multi_900':
        n_signals = 3
        ABP0 = Segment[3][50:950]
        ABP1 = Segment[4][50:950]
        ABP2 = Segment[5][50:950]
        
        all_signals = np.array((ABP0, ABP1, ABP2))
        
        # Make array for all instances
        instance = []

        # Iterate over all input signals for an instance
        for j in range(0, n_signals):
            one_signal = all_signals[j]
            one_signal = np.expand_dims(one_signal, axis=0)
            instance.append(tf.cast(one_signal, tf.float32))
        
    
    elif typ == 'PPG':
        n_signals = 6
        PPG0 = Segment[0]
        PPG1 = Segment[1]
        PPG2 = Segment[2]
        temp0 = Segment[6]
        temp1 = Segment[7]
        temp2 = Segment[8]
        
        all_signals = np.array((PPG0, PPG1, PPG2, temp0, temp1, temp2))
        
        # Make array for all instances
        instance = []

        # Iterate over all input signals for an instance
        for j in range(0, n_signals):
            one_signal = all_signals[j]
            one_signal = np.expand_dims(one_signal, axis=0)
            instance.append(tf.cast(one_signal, tf.float32))
    
    return instance

def make_np_array_from_quant_id(path_main, quant_id, typ):
    Segment = np.load(path_main+quant_id)
    
    if typ == 'ABP_single':
        n_signals = 1 
        
        instance = np.array((n_signals,1,1000))
        
        instance[n_signals][0][:] = Segment[3]
        
    elif typ == 'ABP_multi':
        n_signals = 3
        print(Segment.shape)
        
        instance = np.array((n_signals,1,1000))
        
        instance[0][0][:] = Segment[3]
        instance[1][0][:] = Segment[4]
        instance[2][0][:] = Segment[5]
        
    
    elif typ == 'PPG':
        n_signals = 6
        
        instance = np.array((6,1,1000))
        
        instance[0][0][:] = Segment[0]
        instance[1][0][:] = Segment[1]
        instance[2][0][:] = Segment[2]
        instance[3][0][:] = Segment[6]
        instance[4][0][:] = Segment[7]
        instance[5][0][:] = Segment[8]
    
    return instance
    
def make_segment_from_quant_id(path_main, quant_id):
    Segment = np.load(path_main+quant_id)
    
    return Segment


def get_gradients(instance, model):
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
        preds = model(instance)
        SBP = preds[:, 0]
        DBP = preds[:, 1]

    grads_SBP = tape.gradient(SBP, instance)
    grads_DBP = tape.gradient(DBP, instance)

    return grads_SBP, grads_DBP

#Gradients = grads[0].numpy()


def get_integrated_gradients(segment, model, baseline=None, num_steps=50):
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
        
        if shape[2] == 900:
            baseline = np.zeros((1, 900))
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
        grad_SBP, grad_DBP = get_gradients(interpolated_signals_tensors, model)
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
        IG = np.zeros((3, 1000))
        for i in range(0, matrix_shape[0]):
            IG[i, :] = (IG_matrix[i, :]-all_mean[i])/all_std[i]

    return IG
