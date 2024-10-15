"""
Master Studienarbeit im Master Studiengang: Biomedizinische Informationstechnik
Titel: Quantitative Erklärbarkeit tiefer neuronaler Netze in der Analyse von Biosignalen
Autor: Lennart Brakelmann
FH Dortmund
"""
import numpy as np
import tensorflow as tf

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


def calculate_AOPC(x, IG, k, pattern, window_length, replacement_strategy, model):
    #Rank attributions
    IG_sum_atts, IG_ranks = rank_attributions(IG, pattern, window_length)
    #Replace k features depending on the hyperparameters
    x_replaced = replace_k_features(x, IG_ranks, k, window_length, replacement_strategy)
    #Calculate AOPC with formula
    f_x = model.predict(x)
    summe = 0
    all_f_x_k = []
    for i in range(0,k):
        #Make tf Tensor from numpy array
        x_k = [tf.cast(np.expand_dims(x_replaced[i][j],axis=0), tf.float32) for j in range(0,6)]
        f_x_k = model.predict(x_k)
        diff = f_x - f_x_k
        summe = summe + diff
        all_f_x_k.append(f_x_k)
    AOPC = 1/k * summe
    
    return AOPC, all_f_x_k